"""import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from gan.gan import Generator, Discriminator, LATENT_DIM
from preprocessing.preprocessing import gan_transform

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dataset = datasets.ImageFolder(
    "DataSet_Final/train",
    transform=gan_transform()
)

loader = DataLoader(dataset, batch_size=16, shuffle=True)

G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)

criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=0.0002)
opt_D = optim.Adam(D.parameters(), lr=0.0002)

for epoch in range(20):
    for imgs, _ in loader:
        imgs = imgs.to(DEVICE)
        bs = imgs.size(0)

        real = torch.ones(bs, 1).to(DEVICE)
        fake = torch.zeros(bs, 1).to(DEVICE)

        # ---- Train Discriminator ----
        noise = torch.randn(bs, LATENT_DIM, 1, 1).to(DEVICE)
        fake_imgs = G(noise)

        loss_real = criterion(D(imgs), real)
        loss_fake = criterion(D(fake_imgs.detach()), fake)
        d_loss = loss_real + loss_fake

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        # ---- Train Generator ----
        g_loss = criterion(D(fake_imgs), real)
        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

    print(f"Epoch {epoch+1} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

torch.save(G.state_dict(), "models/gan_generator.pth")
print("✅ GAN Generator saved")"""


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from gan.gan import Generator, Discriminator, LATENT_DIM

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 20
LR = 0.0002
IMG_SIZE = 64

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

# ---------------- DATASET ----------------
dataset = datasets.ImageFolder(
    "DataSet_Final/train",
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# ---------------- MODELS ----------------
G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)

# ---------------- LOSS & OPTIMIZERS ----------------
criterion = nn.BCELoss()

optimizer_G = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

# ---------------- TRAINING LOOP ----------------
for epoch in range(EPOCHS):
    for real_imgs, _ in loader:

        real_imgs = real_imgs.to(DEVICE)
        batch_size = real_imgs.size(0)

        # Real and fake labels
        real_labels = torch.ones(batch_size).to(DEVICE)
        fake_labels = torch.zeros(batch_size).to(DEVICE)

        # -------- Train Discriminator --------
        noise = torch.randn(batch_size, LATENT_DIM, 1, 1).to(DEVICE)
        fake_imgs = G(noise)

        D_real = D(real_imgs)
        D_fake = D(fake_imgs.detach())

        loss_real = criterion(D_real, real_labels)
        loss_fake = criterion(D_fake, fake_labels)

        loss_D = (loss_real + loss_fake) / 2

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # -------- Train Generator --------
        noise = torch.randn(batch_size, LATENT_DIM, 1, 1).to(DEVICE)
        fake_imgs = G(noise)

        output = D(fake_imgs)
        loss_G = criterion(output, real_labels)

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

    print(f"Epoch {epoch+1} | D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f}")

# ---------------- SAVE GENERATOR ----------------
torch.save(G.state_dict(), "models/gan_generator.pth")
print("✅ GAN Generator saved successfully")

