import os

base = "DataSet_Final"

for split in ["train", "validation", "test"]:
    for cls in ["real", "fake"]:
        path = os.path.join(base, split, cls)
        print(f"{split}/{cls}: {len(os.listdir(path))}")
