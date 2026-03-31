"""from torchvision.transforms import *

# ✅ Standard ImageNet Normalization (CRITICAL)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ----------------------------------------------------
# ✅ Swin Transformer Transform (Very Mild)
# ----------------------------------------------------
def swin_transform():
    return Compose([
        Resize((224, 224)),

        # ✅ VERY mild augmentation (Transformers sensitive)
        RandomHorizontalFlip(p=0.5),
        RandomRotation(3),

        ColorJitter(
            brightness=0.15,
            contrast=0.15
        ),

        ToTensor(),
        Normalize(MEAN, STD)
    ])


# ----------------------------------------------------
# ✅ EfficientNet Transform (Moderate)
# ----------------------------------------------------
def classifier_transform():
    return Compose([
        Resize((224, 224)),

        # ✅ Moderate augmentation (avoid destroying artifacts)
        RandomHorizontalFlip(p=0.5),
        RandomRotation(7),

        RandomAffine(
            degrees=7,
            translate=(0.03, 0.03)
        ),

        ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.15
        ),

        # ✅ Gentle blur ONLY
        GaussianBlur(kernel_size=3),

        ToTensor(),
        Normalize(MEAN, STD)
    ])


# ----------------------------------------------------
# ✅ Inference Transform (STRICT ⚠️)
# ----------------------------------------------------
def inference_transform():
    return Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(MEAN, STD)
    ])
"""

from torchvision.transforms import *

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def swin_transform():
    return Compose([
        Resize((224, 224)),
        RandomHorizontalFlip(p=0.5),
        RandomRotation(3),
        ColorJitter(brightness=0.15, contrast=0.15),
        ToTensor(),
        Normalize(MEAN, STD)
    ])

def classifier_transform():
    return Compose([
        Resize((224, 224)),
        RandomHorizontalFlip(p=0.5),
        RandomRotation(7),
        RandomAffine(degrees=7, translate=(0.03, 0.03)),
        ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
        GaussianBlur(3),
        ToTensor(),
        Normalize(MEAN, STD)
    ])

"""def inference_transform():
    return Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(MEAN, STD)
    ])"""

# import torchvision.transforms as transforms

def inference_transform():
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])