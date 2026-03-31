"""import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.models import swin_t, Swin_T_Weights

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Image transforms
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Dataset
train_ds = datasets.ImageFolder("DataSet_Final/train", transform=transform)
val_ds = datasets.ImageFolder("DataSet_Final/validation", transform=transform)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8)

# Load pretrained Swin Transformer
model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
model.head = nn.Linear(model.head.in_features, 2)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training
for epoch in range(5):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} completed")

torch.save(model.state_dict(), "models/swin_transformer.pth")
print("✅ Swin Transformer model saved")
"""
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.models import swin_t, Swin_T_Weights
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader
from preprocessing.preprocessing import swin_transform

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize([0.485,0.456,0.406],
              [0.229,0.224,0.225])
])

train_ds = datasets.ImageFolder(
    "DataSet_Final/train",
    transform=swin_transform()
)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)

model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
model.head = nn.Linear(model.head.in_features, 2)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(20):
    model.train()
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    print(f"Swin Epoch {epoch+1} done")

torch.save(model.state_dict(), "models/swin_transformer.pth")
print("✅ Swin saved")

"""

"""   84 
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.models import swin_t, Swin_T_Weights
from torch.utils.data import DataLoader
from preprocessing.preprocessing import swin_transform

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- DATASET ----------------
train_ds = datasets.ImageFolder(
    "DataSet_Final/train",
    transform=swin_transform()
)

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)

val_ds = datasets.ImageFolder(
    "DataSet_Final/validation",
    transform=swin_transform()
)

val_loader = DataLoader(val_ds, batch_size=8)

# ---------------- MODEL ----------------
model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)

# ✅ Replace classifier head FIRST 🔥
model.head = nn.Linear(model.head.in_features, 2)

# ✅ Freeze entire backbone
for param in model.parameters():
    param.requires_grad = False

# ✅ Unfreeze HEAD 🔥🔥🔥 (CRITICAL)
for param in model.head.parameters():
    param.requires_grad = True

# ✅ Fine-tune LAST Swin stage 🔥
for param in model.features[-1].parameters():
    param.requires_grad = True

model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

# ✅ Optimizer ONLY trainable params 🔥
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=5e-5
)

# ---------------- TRAINING ----------------
best_acc = 0

for epoch in range(5):

    # ---------------- TRAIN ----------------
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(x)

        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        preds = outputs.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    train_acc = correct / total

    # ---------------- VALIDATION ----------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            outputs = model(x)
            preds = outputs.argmax(1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    val_acc = correct / total

    print(f"Epoch {epoch+1} | Train Loss: {running_loss:.4f} | Train Acc : {train_acc:.4f} | Val Acc   : {val_acc:.4f}\n")


    # ---------------- SAVE BEST MODEL 🔥 ----------------
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "models/best_swin.pth")
        print("✅ Best Swin Model Updated")

# ---------------- FINAL ----------------
print(f"\n🔥 Best Validation Accuracy: {best_acc:.4f}")

torch.save(model.state_dict(), "models/swin_transformer.pth")
print("✅ Final Swin Transformer Saved")
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.models import swin_t, Swin_T_Weights
from torch.utils.data import DataLoader
from preprocessing.preprocessing import swin_transform
from torch.optim.lr_scheduler import StepLR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- DATASET ----------------
train_ds = datasets.ImageFolder("DataSet_Final/train", transform=swin_transform())
val_ds   = datasets.ImageFolder("DataSet_Final/validation", transform=swin_transform())

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=8)

# ---------------- MODEL ----------------
model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
model.head = nn.Linear(model.head.in_features, 2)

# Freeze everything
for p in model.parameters():
    p.requires_grad = False

# 🔥 Unfreeze HEAD
for p in model.head.parameters():
    p.requires_grad = True

# 🔥 Unfreeze LAST TWO stages (IMPORTANT)
for p in model.features[-2:].parameters():
    p.requires_grad = True

model.to(DEVICE)

# 🔥 Class weights (boost REAL recall)
class_weights = torch.tensor([1.2, 1.0]).to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.1
)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=3e-5   # slightly smaller for deeper tuning
)

scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

# ---------------- EARLY STOPPING ----------------
best_acc = 0
patience = 6
counter = 0

# ---------------- TRAINING ----------------
for epoch in range(30):
    model.train()
    running_loss, correct, total = 0, 0, 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(x)

        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        preds = outputs.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    train_acc = correct / total
    train_loss = running_loss / len(train_loader)
    # -------- VALIDATION --------
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    val_acc = correct / total

    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        counter = 0
        torch.save(model.state_dict(), "models/best_swin.pth")
        print("✅ Best Swin Updated")
    else:
        counter += 1

    if counter >= patience:
        print("🔥 Early stopping triggered")
        break

    scheduler.step()

print(f"\n🔥 Best Validation Accuracy: {best_acc:.4f}")
torch.save(model.state_dict(), "models/swin_transformer.pth")
print("✅ Final Swin Saved")