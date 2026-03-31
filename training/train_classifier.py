"""import torch
import torch.nn as nn
from torchvision import datasets, models
from torch.utils.data import DataLoader

from preprocessing.preprocessing import classifier_transform

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_ds = datasets.ImageFolder(
    "DataSet_Final/train",
    transform=classifier_transform()
)

val_ds = datasets.ImageFolder(
    "DataSet_Final/validation",
    transform=classifier_transform()
)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16)

model = models.efficientnet_b0(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(1280, 2)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(10):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} completed")

torch.save(model.state_dict(), "models/efficientnet.pth")
print("✅ EfficientNet model saved")
"""

"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torchvision import datasets, models
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader
from preprocessing.preprocessing import classifier_transform


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = Compose([
    Resize((224,224)),
    ToTensor(),
    Normalize([0.485,0.456,0.406],
              [0.229,0.224,0.225])
])

train_ds = datasets.ImageFolder(
    "DataSet_Final/train",
    transform=classifier_transform()
)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

model = models.efficientnet_b0(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(1280, 2)
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
    print(f"EfficientNet Epoch {epoch+1} done")

torch.save(model.state_dict(), "models/efficientnet.pth")
print("✅ EfficientNet saved")
"""

"""  84
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torchvision import datasets, models
from torch.utils.data import DataLoader
from preprocessing.preprocessing import classifier_transform

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- DATASET ----------------
train_ds = datasets.ImageFolder(
    "DataSet_Final/train",
    transform=classifier_transform()
)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)

val_ds = datasets.ImageFolder(
    "DataSet_Final/validation",
    transform=classifier_transform()
)

val_loader = DataLoader(val_ds, batch_size=16)

# ---------------- MODEL ----------------
model = models.efficientnet_b0(weights="IMAGENET1K_V1")

# ✅ Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# ✅ Replace classifier
model.classifier[1] = nn.Linear(1280, 2)

# ✅ Fine-tune LAST block 🔥
for param in model.features[-2].parameters():
    param.requires_grad = True

model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

# ✅ Optimizer ONLY trainable params 🔥
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=8e-5
)

# ---------------- TRAINING ----------------
best_acc = 0  # ✅ Track best validation accuracy

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
        torch.save(model.state_dict(), "models/best_efficientnet.pth")
        print("✅ Best EfficientNet Model Updated")

# ---------------- SAVE ----------------
print(f"\n🔥 Best Validation Accuracy: {best_acc:.4f}")

torch.save(model.state_dict(), "models/efficientnet.pth")
print("✅ Final EfficientNet Saved")

"""


import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torchvision import datasets, models
from torch.utils.data import DataLoader
from preprocessing.preprocessing import classifier_transform
from torch.optim.lr_scheduler import StepLR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- DATASET ----------------
train_ds = datasets.ImageFolder("DataSet_Final/train", transform=classifier_transform())
val_ds   = datasets.ImageFolder("DataSet_Final/validation", transform=classifier_transform())

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=16)

# ---------------- MODEL ----------------
model = models.efficientnet_b0(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(1280, 2)

# Freeze all
for p in model.parameters():
    p.requires_grad = False

# 🔥 Unfreeze classifier
for p in model.classifier.parameters():
    p.requires_grad = True

# 🔥 Unfreeze LAST THREE blocks
for p in model.features[-3:].parameters():
    p.requires_grad = True

model.to(DEVICE)

# 🔥 Class weights
class_weights = torch.tensor([1.2, 1.0]).to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.1
)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=6e-5
)

scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

best_acc = 0
patience = 6
counter = 0

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
        torch.save(model.state_dict(), "models/best_efficientnet.pth")
        print("✅ Best EfficientNet Updated")
    else:
        counter += 1

    if counter >= patience:
        print("🔥 Early stopping triggered")
        break

    scheduler.step()

print(f"\n🔥 Best Validation Accuracy: {best_acc:.4f}")
torch.save(model.state_dict(), "models/efficientnet.pth")
print("✅ Final EfficientNet Saved")