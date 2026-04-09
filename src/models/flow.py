"""Conditional Glow normalizing flow for 125-band hyperspectral images.

Multi-scale architecture (3 scales x K steps). Each step:
    ActNorm -> Invertible 1x1 Conv (LU) -> Affine Coupling
Class-conditional via label embedding injected into coupling layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


class ActNorm(nn.Module):
    """Activation normalization with data-dependent init (zero mean, unit var)."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.register_buffer("initialized", torch.tensor(False))

    def _initialize(self, x: torch.Tensor):
        with torch.no_grad():
            mean = x.mean(dim=[0, 2, 3], keepdim=True)
            std = x.std(dim=[0, 2, 3], keepdim=True) + 1e-6
            self.bias.data.copy_(-mean)
            self.log_scale.data.copy_(-torch.log(std))
            self.initialized.fill_(True)

    def forward(
        self, x: torch.Tensor, reverse: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.initialized and not reverse:
            self._initialize(x)

        _, _, H, W = x.shape
        # log_det is the same for every sample in the batch (scalar, broadcasts)
        log_det = H * W * self.log_scale.sum()

        if not reverse:
            return x * torch.exp(self.log_scale) + self.bias, log_det
        else:
            return (x - self.bias) * torch.exp(-self.log_scale), -log_det


class Invertible1x1Conv(nn.Module):
    """Invertible 1x1 conv with LU decomposition. log|det| = sum(log_s)."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

        # Initialise from a random orthogonal matrix
        W = torch.linalg.qr(torch.randn(num_channels, num_channels))[0]
        P, L, U = torch.linalg.lu(W)
        s = torch.diag(U)

        self.register_buffer("P", P)
        self.register_buffer("sign_s", torch.sign(s))
        self.register_buffer(
            "L_mask", torch.tril(torch.ones_like(L), diagonal=-1)
        )
        self.register_buffer(
            "U_mask", torch.triu(torch.ones_like(U), diagonal=1)
        )
        self.register_buffer("eye", torch.eye(num_channels))

        self.log_s = nn.Parameter(torch.log(torch.abs(s)))
        self.L = nn.Parameter(L)
        self.U = nn.Parameter(U)

    def _weight(self) -> torch.Tensor:
        L = self.L * self.L_mask + self.eye
        U = self.U * self.U_mask + torch.diag(
            self.sign_s * torch.exp(self.log_s)
        )
        return self.P @ L @ U

    def forward(
        self, x: torch.Tensor, reverse: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, H, W = x.shape
        log_det = H * W * self.log_s.sum()

        if not reverse:
            weight = self._weight()
            return F.conv2d(x, weight.unsqueeze(-1).unsqueeze(-1)), log_det
        else:
            inv_weight = torch.inverse(self._weight())
            return F.conv2d(x, inv_weight.unsqueeze(-1).unsqueeze(-1)), -log_det


class CouplingNetwork(nn.Module):
    """Small CNN for coupling params. Zero-init output; class embed as spatial bias."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        class_embed_dim: int,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(hidden_channels, out_channels, 3, padding=1)
        self.class_proj = nn.Linear(class_embed_dim, hidden_channels)

        nn.init.zeros_(self.out_conv.weight)
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, c_embed: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        h = h + self.class_proj(c_embed).unsqueeze(-1).unsqueeze(-1)
        return self.out_conv(h)


class AffineCoupling(nn.Module):
    """Affine coupling: split channels, one half parameterises scale+shift for the other."""

    def __init__(
        self, num_channels: int, hidden_channels: int, class_embed_dim: int
    ):
        super().__init__()
        half = num_channels // 2
        self.nn = CouplingNetwork(
            half, half * 2, hidden_channels, class_embed_dim
        )

    def forward(
        self,
        x: torch.Tensor,
        c_embed: torch.Tensor,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_a, x_b = x.chunk(2, dim=1)

        h = self.nn(x_b, c_embed)
        log_s, t = h.chunk(2, dim=1)
        log_s = torch.tanh(log_s) * 2.0  # clamp to [-2, 2]

        if not reverse:
            y_a = x_a * torch.exp(log_s) + t
            log_det = log_s.sum(dim=[1, 2, 3])  # per-sample (B,)
        else:
            y_a = (x_a - t) * torch.exp(-log_s)
            log_det = -log_s.sum(dim=[1, 2, 3])

        return torch.cat([y_a, x_b], dim=1), log_det


class FlowStep(nn.Module):
    """Single Glow step: ActNorm -> Inv1x1Conv -> AffineCoupling."""

    def __init__(
        self, num_channels: int, hidden_channels: int, class_embed_dim: int
    ):
        super().__init__()
        self.actnorm = ActNorm(num_channels)
        self.inv1x1 = Invertible1x1Conv(num_channels)
        self.coupling = AffineCoupling(
            num_channels, hidden_channels, class_embed_dim
        )

    def forward(
        self,
        x: torch.Tensor,
        c_embed: torch.Tensor,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_det = torch.zeros(x.size(0), device=x.device)
        layers = [
            lambda x: self.actnorm(x, reverse=reverse),
            lambda x: self.inv1x1(x, reverse=reverse),
            lambda x: self.coupling(x, c_embed, reverse=reverse),
        ]
        for layer in (reversed(layers) if reverse else layers):
            x, ld = layer(x)
            log_det = log_det + ld
        return x, log_det


def squeeze(x: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, 4C, H/2, W/2)."""
    B, C, H, W = x.shape
    x = x.view(B, C, H // 2, 2, W // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    return x.view(B, C * 4, H // 2, W // 2)


def unsqueeze(x: torch.Tensor) -> torch.Tensor:
    """(B, 4C, H/2, W/2) -> (B, C, H, W)."""
    B, C4, H2, W2 = x.shape
    C = C4 // 4
    x = x.view(B, C, 2, 2, H2, W2)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    return x.view(B, C, H2 * 2, W2 * 2)


class ConditionalGlow(nn.Module):
    """Multi-scale conditional Glow normalizing flow."""

    def __init__(
        self,
        in_channels: int = 125,
        num_classes: int = 10,
        num_scales: int = 3,
        num_steps: int = 4,
        hidden_channels: int = 256,
        class_embed_dim: int = 64,
        use_grad_checkpoint: bool = False,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.use_grad_checkpoint = use_grad_checkpoint
        self.in_channels = in_channels

        self.class_embed = nn.Embedding(num_classes, class_embed_dim)

        self.flow_scales = nn.ModuleList()
        C = in_channels
        for s in range(num_scales):
            C *= 4
            steps = nn.ModuleList(
                [FlowStep(C, hidden_channels, class_embed_dim) for _ in range(num_steps)]
            )
            self.flow_scales.append(steps)
            if s < num_scales - 1:
                C //= 2

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Encode image (B,125,128,128) to latent z's + log_det."""
        c_embed = self.class_embed(labels)
        z_list: list[torch.Tensor] = []
        total_log_det = torch.zeros(x.size(0), device=x.device)

        h = x
        for s, steps in enumerate(self.flow_scales):
            h = squeeze(h)
            for step in steps:
                if self.use_grad_checkpoint and self.training:
                    h, ld = grad_checkpoint(  # type: ignore[misc]
                        step, h, c_embed, False, use_reentrant=False
                    )
                else:
                    h, ld = step(h, c_embed)
                total_log_det = total_log_det + ld

            if s < self.num_scales - 1:
                z, h = h.chunk(2, dim=1)
                z_list.append(z)
            else:
                z_list.append(h)

        return z_list, total_log_det

    @torch.no_grad()
    def reverse(
        self, z_list: list[torch.Tensor], labels: torch.Tensor
    ) -> torch.Tensor:
        """Decode latent z's back to image space."""
        c_embed = self.class_embed(labels)
        h = z_list[-1]

        for s in reversed(range(self.num_scales)):
            if s < self.num_scales - 1:
                h = torch.cat([z_list[s], h], dim=1)
            for step in reversed(list(self.flow_scales[s])):
                h, _ = step(h, c_embed, reverse=True)
            h = unsqueeze(h)

        return h

    @torch.no_grad()
    def generate(
        self,
        labels: torch.Tensor,
        temperature: float = 0.7,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Sample images: z ~ N(0, T²I) -> reverse -> clamp to [0,1]."""
        if device is None:
            device = next(self.parameters()).device
        labels = labels.to(device)

        z_list: list[torch.Tensor] = []
        C = self.in_channels
        H, W = 128, 128

        for s in range(self.num_scales):
            C, H, W = C * 4, H // 2, W // 2
            if s < self.num_scales - 1:
                z_list.append(
                    torch.randn(labels.size(0), C // 2, H, W, device=device)
                    * temperature
                )
                C = C // 2
            else:
                z_list.append(
                    torch.randn(labels.size(0), C, H, W, device=device)
                    * temperature
                )

        return torch.clamp(self.reverse(z_list, labels), 0.0, 1.0)


def glow_nll_loss(
    z_list: list[torch.Tensor], log_det: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """NLL = 0.5*sum(z²) - log_det, averaged per dimension."""
    total_dims = sum(z.shape[1] * z.shape[2] * z.shape[3] for z in z_list)

    prior: torch.Tensor = sum(  # type: ignore[assignment]
        (0.5 * (z**2).sum(dim=[1, 2, 3]) for z in z_list),
        torch.tensor(0.0),
    )
    nll = (prior - log_det) / total_dims  # (B,) nats/dim

    return (
        nll.mean(),
        prior.mean() / total_dims,
        log_det.mean() / total_dims,
    )
