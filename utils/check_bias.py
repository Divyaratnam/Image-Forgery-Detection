import os
from PIL import Image
import numpy as np
import cv2

# ==============================
# CONFIG — CHANGE IF NEEDED
# ==============================
REAL_PATH = r"D:\Final Year Project\Image-Forgery-Detection\DataSet_Final\test\real"
FAKE_PATH = r"D:\Final Year Project\Image-Forgery-Detection\DataSet_Final\test\fake"

# ==============================
# HELPER — LOAD IMAGES
# ==============================
def load_images(folder):
    images = []

    for root, _, files in os.walk(folder):
        for f in files:
            path = os.path.join(root, f)
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
            except:
                pass

    return images

# ==============================
# 1️⃣ RESOLUTION CHECK
# ==============================
def analyze_resolutions(images, label):
    widths, heights = [], []

    for img in images:
        w, h = img.size
        widths.append(w)
        heights.append(h)

    print(f"\n📏 {label} RESOLUTION STATS")
    print("Avg Width :", round(np.mean(widths), 2))
    print("Avg Height:", round(np.mean(heights), 2))
    print("Min Width :", np.min(widths))
    print("Max Width :", np.max(widths))

# ==============================
# 2️⃣ BLUR CHECK (Variance of Laplacian)
# ==============================
def analyze_blur(images, label):
    blur_scores = []

    for img in images:
        img_cv = np.array(img)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_scores.append(score)

    print(f"\n🌫 {label} BLUR STATS")
    print("Avg Blur Score:", round(np.mean(blur_scores), 2))
    print("Min Blur Score:", round(np.min(blur_scores), 2))
    print("Max Blur Score:", round(np.max(blur_scores), 2))

# ==============================
# 3️⃣ BRIGHTNESS CHECK
# ==============================
def analyze_brightness(images, label):
    brightness_vals = []

    for img in images:
        img_np = np.array(img)
        brightness_vals.append(img_np.mean())

    print(f"\n💡 {label} BRIGHTNESS STATS")
    print("Avg Brightness:", round(np.mean(brightness_vals), 2))
    print("Min Brightness:", round(np.min(brightness_vals), 2))
    print("Max Brightness:", round(np.max(brightness_vals), 2))

# ==============================
# RUN ANALYSIS
# ==============================
print("🚀 Starting Dataset Bias Analysis...")

real_images = load_images(REAL_PATH)
fake_images = load_images(FAKE_PATH)

print(f"\nLoaded REAL images: {len(real_images)}")
print(f"Loaded FAKE images: {len(fake_images)}")

# Resolution
analyze_resolutions(real_images, "REAL")
analyze_resolutions(fake_images, "FAKE")

# Blur
analyze_blur(real_images, "REAL")
analyze_blur(fake_images, "FAKE")

# Brightness
analyze_brightness(real_images, "REAL")
analyze_brightness(fake_images, "FAKE")

print("\n✅ Bias Analysis Completed")