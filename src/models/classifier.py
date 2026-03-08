"""
Hybrid CNN + Transformer classifier for hyperspectral crop disease severity.
"""

import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe: torch.Tensor = self.pe  # type: ignore[assignment]
        return x + pe[:, : x.size(1)]


class HybridCNNTransformer(nn.Module):
    """CNN backbone → spatial tokens → Transformer encoder → classification.

    Args:
        in_channels: Number of spectral bands (default 125).
        num_classes: Number of disease severity classes (default 10).
        nhead: Transformer attention heads.
        num_layers: Transformer encoder layers.
        dim_feedforward: Transformer FFN hidden dimension.
        dropout: Dropout rate used throughout.
    """

    def __init__(
        self,
        in_channels: int = 125,
        num_classes: int = 10,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        # CNN feature extractor (125-ch → 256-ch, 128×128 → 16×16)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),  # → 64×64
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),  # → 32×32
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.MaxPool2d(2),  # → 16×16
        )
        self.cnn_dropout = nn.Dropout2d(dropout)

        # Transformer on spatial tokens
        self.feature_dim = 256
        self.layer_norm = nn.LayerNorm(self.feature_dim)
        self.pos_encoder = PositionalEncoding(self.feature_dim, max_len=16 * 16)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Classifier head
        self.head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, C=125, H=128, W=128) — already in CHW from the Dataset.
        Returns:
            logits: (batch, num_classes)
        """
        x = self.cnn(x)  # (B, 256, 16, 16)
        x = self.cnn_dropout(x)
        x = x.flatten(2).transpose(1, 2)  # (B, 256_tokens, 256_dim)
        x = self.layer_norm(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)  # (B, 256, 256)
        x = x.mean(dim=1)  # global average pool → (B, 256)
        return self.head(x)
