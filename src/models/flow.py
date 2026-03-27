"""
Conditional Normalizing Flow for hyperspectral data generation.
"""
"""
Conditional Normalizing Flow for hyperspectral data generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from pathlib import Path
import random

# ======================
# Dataset
# ======================

class HyperSpectralDataset(Dataset):

    def __init__(self, root):
        self.samples = []
        root = Path(root)

        for class_dir in sorted(root.iterdir()):
            if class_dir.is_dir():
                label = int(class_dir.name)
                for file in class_dir.glob("*.npy"):
                    self.samples.append((file, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        file, label = self.samples[idx]

        data = np.load(file).astype(np.float32) / 4000.0
        data = data[:32, :32, :]  # crop

        data = torch.tensor(data).permute(2,0,1)

        return data, label


# ======================
# Subset (30%)
# ======================

def get_subset(dataset, percent=0.3):

    size = int(len(dataset) * percent)

    indices = random.sample(range(len(dataset)), size)

    return Subset(dataset, indices)


# ======================
# DataLoader
# ======================

data_dir = Path("../data/beyond-visible-spectrum-ai-for-agriculture-2025p2/Train")

dataset = HyperSpectralDataset(data_dir)

small_dataset = get_subset(dataset, percent=0.3)

train_loader = DataLoader(
    small_dataset,
    batch_size=4,
    shuffle=True
)

print("Subset size:", len(small_dataset))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================================================
# 1️⃣ Conditional VAE
# ====================================================

class VAE(nn.Module):

    def __init__(self, latent_dim=128):

        super().__init__()

        input_dim = 125*32*32

        self.encoder = nn.Sequential(
            nn.Linear(input_dim+10,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU()
        )

        self.mu = nn.Linear(512,latent_dim)
        self.logvar = nn.Linear(512,latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim+10,512),
            nn.ReLU(),
            nn.Linear(512,1024),
            nn.ReLU(),
            nn.Linear(1024,input_dim)
        )

    def forward(self,x,c):

        x = x.view(x.size(0),-1)

        h = torch.cat([x,c],dim=1)

        h = self.encoder(h)

        mu = self.mu(h)
        logvar = self.logvar(h)

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        z = mu + eps*std

        z = torch.cat([z,c],dim=1)

        recon = self.decoder(z)

        return recon, mu, logvar

    def loss(self,x,labels):

        x = x.to(device)

        c = F.one_hot(labels,10).float().to(device)

        recon,mu,logvar = self.forward(x,c)

        x = x.view(x.size(0),-1)

        recon_loss = F.mse_loss(recon,x)

        kl = -0.5 * torch.mean(1+logvar-mu.pow(2)-logvar.exp())

        return recon_loss + kl


# ====================================================
# 2️⃣ GAN
# ====================================================

class Generator(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(128+10,512),
            nn.ReLU(),
            nn.Linear(512,1024),
            nn.ReLU(),
            nn.Linear(1024,125*32*32)
        )

    def forward(self,z,c):

        x = torch.cat([z,c],dim=1)

        return self.net(x)


class Discriminator(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(125*32*32+10,512),
            nn.ReLU(),
            nn.Linear(512,1)
        )

    def forward(self,x,c):

        x = x.view(x.size(0),-1)

        x = torch.cat([x,c],dim=1)

        return self.net(x)


# ====================================================
# 3️⃣ Normalizing Flow (simplified)
# ====================================================

class SimpleFlow(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(125*32*32+10,1024),
            nn.ReLU(),
            nn.Linear(1024,125*32*32)
        )

    def loss(self,x,labels):

        x = x.to(device)

        c = F.one_hot(labels,10).float().to(device)

        x = x.view(x.size(0),-1)

        inp = torch.cat([x,c],dim=1)

        z = self.net(inp)

        loss = torch.mean(z**2)

        return loss


# ====================================================
# TRAINER
# ====================================================

def train_model(model,loader,name):

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

    print("\nTraining",name)

    for epoch in range(1):

        total_loss = 0

        for data,labels in loader:

            optimizer.zero_grad()

            loss = model.loss(data,labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print("Loss:",total_loss/len(loader))


# ====================================================
# RUN EXPERIMENTS
# ====================================================

vae_model = VAE()

flow_model = SimpleFlow()

train_model(vae_model,train_loader,"VAE")

train_model(flow_model,train_loader,"FLOW")
