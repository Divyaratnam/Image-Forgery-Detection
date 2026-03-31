import os
import random
import shutil

# ===== PATHS (CHANGE ONLY IF YOUR FOLDER NAME IS DIFFERENT) =====
SOURCE_DIR = r"D:\Final Year Project\Image-Forgery-Detection\DataSet"
TARGET_DIR = r"D:\Final Year Project\Image-Forgery-Detection\DataSet_Final"

TEST_RATIO = 0.15   # 15% of training data goes to TEST

# ===== FUNCTION TO SPLIT TRAINING INTO TRAIN + TEST =====
def split_training(src_class, dst_class):
    src_path = os.path.join(SOURCE_DIR, "TRAINING", src_class)
    images = os.listdir(src_path)
    random.shuffle(images)

    test_count = int(len(images) * TEST_RATIO)
    test_images = images[:test_count]
    train_images = images[test_count:]

    train_dir = os.path.join(TARGET_DIR, "train", dst_class)
    test_dir = os.path.join(TARGET_DIR, "test", dst_class)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for img in train_images:
        shutil.copy(os.path.join(src_path, img), os.path.join(train_dir, img))

    for img in test_images:
        shutil.copy(os.path.join(src_path, img), os.path.join(test_dir, img))

# ===== FUNCTION TO COPY VALIDATION AS-IS =====
def copy_validation(src_class, dst_class):
    src_path = os.path.join(SOURCE_DIR, "VALIDATION", src_class)
    val_dir = os.path.join(TARGET_DIR, "validation", dst_class)

    os.makedirs(val_dir, exist_ok=True)

    for img in os.listdir(src_path):
        shutil.copy(os.path.join(src_path, img), os.path.join(val_dir, img))

# ===== RUN FOR BOTH CLASSES =====
# ORIGINAL → real
split_training("ORIGINAL", "real")
copy_validation("ORIGINAL", "real")

# TAMPERED → fake
split_training("TAMPERED", "fake")
copy_validation("TAMPERED", "fake")

print("✅ Train, Validation, and Test datasets created successfully!")
