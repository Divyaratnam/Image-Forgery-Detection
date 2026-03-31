"""import torch.nn as nn

LATENT_DIM = 100

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM, 256, 4, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)



        return self.classifier(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),   # -> 64 x 32 x 32
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1), # -> 128 x 16 x 16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1), # -> 256 x 8 x 8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        # 🔥 KEY FIX: adaptive pooling
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # -> 256 x 1 x 1
            nn.Flatten(),              # -> 256
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)"""

import torch
import torch.nn as nn

LATENT_DIM = 100
CHANNELS = 3
FEATURES_GEN = 64
FEATURES_DISC = 64

# ---------------- GENERATOR ----------------
class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(

            # input: Z (100 x 1 x 1)

            nn.ConvTranspose2d(LATENT_DIM, FEATURES_GEN * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(FEATURES_GEN * 8),
            nn.ReLU(True),

            # 4x4

            nn.ConvTranspose2d(FEATURES_GEN * 8, FEATURES_GEN * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURES_GEN * 4),
            nn.ReLU(True),

            # 8x8

            nn.ConvTranspose2d(FEATURES_GEN * 4, FEATURES_GEN * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURES_GEN * 2),
            nn.ReLU(True),

            # 16x16

            nn.ConvTranspose2d(FEATURES_GEN * 2, FEATURES_GEN, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURES_GEN),
            nn.ReLU(True),

            # 32x32

            nn.ConvTranspose2d(FEATURES_GEN, CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh()

            # Output: 64x64
        )

    def forward(self, x):
        return self.net(x)


# ---------------- DISCRIMINATOR ----------------
class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(

            # input: 3x64x64

            nn.Conv2d(CHANNELS, FEATURES_DISC, 4, 2, 1),
            nn.LeakyReLU(0.2),

            # 32x32

            nn.Conv2d(FEATURES_DISC, FEATURES_DISC * 2, 4, 2, 1),
            nn.BatchNorm2d(FEATURES_DISC * 2),
            nn.LeakyReLU(0.2),

            # 16x16

            nn.Conv2d(FEATURES_DISC * 2, FEATURES_DISC * 4, 4, 2, 1),
            nn.BatchNorm2d(FEATURES_DISC * 4),
            nn.LeakyReLU(0.2),

            # 8x8

            nn.Conv2d(FEATURES_DISC * 4, FEATURES_DISC * 8, 4, 2, 1),
            nn.BatchNorm2d(FEATURES_DISC * 8),
            nn.LeakyReLU(0.2),

            # 4x4

            nn.Conv2d(FEATURES_DISC * 8, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1)
    