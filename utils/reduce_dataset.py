import os
import random
import shutil

# ---------------- SETTINGS ----------------
SOURCE_DIR = "LargeDataset"      # CHANGE if needed
TARGET_DIR = "Dataset_20K"

TARGET_TOTAL = 20000
random.seed(42)

# ---------------- SAFETY CHECK ----------------
def check_folder(path):
    if not os.path.exists(path):
        raise Exception(f"Folder NOT FOUND: {path}")

# ---------------- COPY FUNCTION ----------------
def reduce_split(split_name):

    print(f"\nProcessing split -> {split_name}")

    for class_name in ["real", "fake"]:

        src_path = os.path.join(SOURCE_DIR, split_name, class_name)
        check_folder(src_path)

        images = os.listdir(src_path)
        random.shuffle(images)

        total_images = len(images)
        keep_count = int(total_images * REDUCTION_RATIO)

        selected_imgs = images[:keep_count]

        dst_folder = os.path.join(TARGET_DIR, split_name, class_name)
        os.makedirs(dst_folder, exist_ok=True)

        for img in selected_imgs:
            shutil.copy(
                os.path.join(src_path, img),
                os.path.join(dst_folder, img)
            )

        print(f"{split_name}/{class_name} -> kept {keep_count} images")

# ---------------- MAIN ----------------
if __name__ == "__main__":

    print("Starting dataset reduction...\n")

    check_folder(SOURCE_DIR)

    total = 0
    for split in ["train", "validation", "test"]:
        for cls in ["real", "fake"]:
            path = os.path.join(SOURCE_DIR, split, cls)
            check_folder(path)
            total += len(os.listdir(path))

    print(f"Original Dataset Size -> {total} images")

    global REDUCTION_RATIO
    REDUCTION_RATIO = TARGET_TOTAL / total

    print(f"Reduction Ratio -> {REDUCTION_RATIO:.3f}")

    if os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)

    for split in ["train", "validation", "test"]:
        reduce_split(split)

    print("\nSUCCESS -> Dataset_20K created safely!")
