""" 84 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.models import swin_t, efficientnet_b0
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report

from preprocessing.preprocessing import inference_transform

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset
test_ds = datasets.ImageFolder(
    "DataSet_Final/test",
    transform=inference_transform()
)

test_loader = DataLoader(test_ds, batch_size=16)

# Load Swin
swin = swin_t()
swin.head = nn.Linear(swin.head.in_features, 2)
swin.load_state_dict(torch.load("models/swin_transformer.pth", map_location=DEVICE))
swin.to(DEVICE).eval()

# Load EfficientNet
eff = efficientnet_b0()
eff.classifier[1] = nn.Linear(1280, 2)
eff.load_state_dict(torch.load("models/efficientnet.pth", map_location=DEVICE))
eff.to(DEVICE).eval()

y_true, y_pred = [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)

        pswin = torch.softmax(swin(imgs), dim=1)
        peff  = torch.softmax(eff(imgs), dim=1)

        final_prob = 0.8 * pswin + 0.2 * peff
        preds = final_prob.argmax(1).cpu()

        y_pred.extend(preds)
        y_true.extend(labels)

print("Hybrid Accuracy:", accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred))

"""


import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

from torchvision import datasets
from torchvision.models import swin_t, efficientnet_b0
from torch.utils.data import DataLoader
from preprocessing.preprocessing import inference_transform

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- DATASET ----------------
test_ds = datasets.ImageFolder(
    "DataSet_Final/test",
    transform=inference_transform()
)

test_loader = DataLoader(test_ds, batch_size=16)

# ---------------- LOAD SWIN ----------------
swin = swin_t()
swin.head = nn.Linear(swin.head.in_features, 2)
swin.load_state_dict(torch.load("models/best_swin.pth", map_location=DEVICE))
swin.to(DEVICE).eval()

# ---------------- LOAD EFFICIENTNET ----------------
eff = efficientnet_b0()
eff.classifier[1] = nn.Linear(1280, 2)
eff.load_state_dict(torch.load("models/best_efficientnet.pth", map_location=DEVICE))
eff.to(DEVICE).eval()

y_true, y_pred, y_scores = [], [], []

# ---------------- INFERENCE ----------------
with torch.no_grad():
    for imgs, labels in test_loader:

        imgs = imgs.to(DEVICE)

        out1 = swin(imgs)
        out2 = eff(imgs)

        p1 = torch.softmax(out1, dim=1)
        p2 = torch.softmax(out2, dim=1)

        conf1 = torch.max(p1, dim=1).values
        conf2 = torch.max(p2, dim=1).values

        eps = 1e-8
        total = conf1 + conf2 + eps

        w1 = conf1 / total
        w2 = conf2 / total

        final_logits = w1.unsqueeze(1) * out1 + w2.unsqueeze(1) * out2
        final_prob = torch.softmax(final_logits, dim=1)

        preds = final_prob.argmax(1).cpu()

        y_pred.extend(preds)
        y_true.extend(labels)

        # store FAKE probability for ROC
        y_scores.extend(final_prob[:, 1].cpu())

# ---------------- METRICS ----------------
print("\n🔥 HYBRID MODEL PERFORMANCE")
print("Hybrid Accuracy:", accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred))

# ---------------- CONFUSION MATRIX 🔥 ----------------
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["REAL", "FAKE"]
)

disp.plot()
plt.title("Confusion Matrix")
plt.show()


# ---------------- ROC CURVE  ----------------
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# ---------------- ERROR ANALYSIS  ----------------
#print("\n ERROR ANALYSIS")

#for i in range(len(y_pred)):

#    if y_pred[i] != y_true[i]:
#
#        print("\n❌ Misclassified Sample")
#        print("True Label :", "REAL" if y_true[i]==0 else "FAKE")
#        print("Predicted  :", "REAL" if y_pred[i]==0 else "FAKE")



"""
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

class_names = test_ds.classes
image_paths = [path for path, _ in test_ds.samples]

# ---------------- LOAD MODELS ----------------
swin = swin_t()
swin.head = nn.Linear(swin.head.in_features, 2)
swin.load_state_dict(torch.load("models/best_swin.pth", map_location=DEVICE))
swin.to(DEVICE).eval()

eff = efficientnet_b0()
eff.classifier[1] = nn.Linear(1280, 2)
eff.load_state_dict(torch.load("models/best_efficientnet.pth", map_location=DEVICE))
eff.to(DEVICE).eval()

print("\n--- TEST RESULTS ---\n")

# ---------------- TEST LOOP ----------------
for i, (img, label) in enumerate(test_loader):

    img = img.to(DEVICE)

    with torch.no_grad():

        pswin = torch.softmax(swin(img), dim=1)
        peff = torch.softmax(eff(img), dim=1)

        final_prob = 0.6 * pswin + 0.4 * peff

        pred = final_prob.argmax(1).item()

    actual_label = class_names[label.item()]
    predicted_label = class_names[pred]

    image_name = os.path.basename(image_paths[i])

    # ---------------- PRINT RESULT ----------------
    if pred != label.item():

        print(f"❌ {image_name} → Predicted: {predicted_label.upper()} | Actual: {actual_label.upper()} ") #pred} {label} ")

    else:

        print(f"✅ {image_name} → Correct ({actual_label.upper()})")
    
    print(image_name, final_prob)

    """