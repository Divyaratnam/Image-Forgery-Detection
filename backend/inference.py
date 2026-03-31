"""import torch
from torchvision.models import swin_t
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load Swin Transformer model
model = swin_t()
model.head = torch.nn.Linear(model.head.in_features, 2)
model.load_state_dict(
    torch.load("models/swin_transformer.pth", map_location=DEVICE)
)
model.to(DEVICE)
model.eval()

# Image preprocessing
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

def predict_image(image_file):
    img = Image.open(image_file).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(img).argmax(1).item()

    return "FAKE" if pred == 1 else "REAL"
"""

"""  84
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torchvision.models import swin_t, efficientnet_b0
from PIL import Image
from preprocessing.preprocessing import inference_transform

# ---------------- DEVICE ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- PATH HANDLING ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# SWIN_MODEL_PATH = os.path.join(BASE_DIR, "models", "swin_transformer.pth")
 # EFF_MODEL_PATH  = os.path.join(BASE_DIR, "models", "efficientnet.pth")

SWIN_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_swin.pth")
EFF_MODEL_PATH  = os.path.join(BASE_DIR, "models", "best_efficientnet.pth")
# ---------------- IMAGE TRANSFORM ----------------
transform = inference_transform()

# ---------------- LOAD SWIN ----------------
swin = swin_t()
swin.head = nn.Linear(swin.head.in_features, 2)
swin.load_state_dict(torch.load(SWIN_MODEL_PATH, map_location=DEVICE))
swin.to(DEVICE).eval()

# ---------------- LOAD EFFICIENTNET ----------------
eff = efficientnet_b0()
eff.classifier[1] = nn.Linear(1280, 2)
eff.load_state_dict(torch.load(EFF_MODEL_PATH, map_location=DEVICE))
eff.to(DEVICE).eval()

# ---------------- HYBRID PREDICTION ----------------
def hybrid_predict(image_file):

    img = Image.open(image_file).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        swin_prob = torch.softmax(swin(img), dim=1)
        eff_prob  = torch.softmax(eff(img), dim=1)

    # ✅ Fusion Weights (tunable 🔥)
    final_prob = 0.8* swin_prob + 0.2 * eff_prob

    pred_class = final_prob.argmax(dim=1).item()
    confidence = final_prob[0][pred_class].item()

    # ✅ Confidence Threshold 🔥🔥🔥
    if confidence < 0.60:
        return "UNCERTAIN", round(confidence * 100, 2)

    label = "FAKE" if pred_class == 1 else "REAL"

    return label, round(confidence * 100, 2)

    """

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torchvision.models import swin_t, efficientnet_b0
from PIL import Image
from preprocessing.preprocessing import inference_transform

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SWIN_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_swin.pth")
EFF_MODEL_PATH  = os.path.join(BASE_DIR, "models", "best_efficientnet.pth")

transform = inference_transform()

# ---------------- LOAD SWIN ----------------
swin = swin_t()
swin.head = nn.Linear(swin.head.in_features, 2)
swin.load_state_dict(torch.load(SWIN_MODEL_PATH, map_location=DEVICE))
swin.to(DEVICE).eval()

# ---------------- LOAD EFFICIENTNET ----------------
eff = efficientnet_b0()
eff.classifier[1] = nn.Linear(1280, 2)
eff.load_state_dict(torch.load(EFF_MODEL_PATH, map_location=DEVICE))
eff.to(DEVICE).eval()

print("✅ Models loaded successfully")


# ---------------- HYBRID PREDICTION ----------------
def hybrid_predict(image_file):

    img = Image.open(image_file).convert("RGB")
    img = transform(img)
    img = img.unsqueeze(0).to(DEVICE)
   # print("Image size:", img.shape)

    with torch.no_grad():
        
        def tta(model, img):
            preds = []
            preds.append(model(img))
            preds.append(model(torch.flip(img, dims=[3])))
            return torch.mean(torch.stack(preds), dim=0)

        out1 = tta(swin, img)
        out2 = tta(eff, img)

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

        pred = final_prob.argmax(1).item()
        confidence = final_prob[0][pred].item()

    class_names = ["fake", "real"]
    label = class_names[pred].upper()

    return label, round(confidence * 100, 2)