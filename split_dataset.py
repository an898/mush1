import os
import random
import shutil
from pathlib import Path

random.seed(42)

# CHANGE THIS if needed
SOURCE_DIR = "crops"
OUTPUT_DIR = "data"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

CLASSES = ["edible", "poisonous"]


def make_dir(path):
    os.makedirs(path, exist_ok=True)


def split_files(file_list, train_ratio, val_ratio, test_ratio):
    random.shuffle(file_list)

    total = len(file_list)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_files = file_list[:train_end]
    val_files = file_list[train_end:val_end]
    test_files = file_list[val_end:]

    return train_files, val_files, test_files


def copy_files(files, destination_folder):
    make_dir(destination_folder)
    for file_path in files:
        shutil.copy(file_path, destination_folder)


def main():
    print("Starting dataset split...")

    # Create output folders
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            make_dir(os.path.join(OUTPUT_DIR, split, cls))

    for cls in CLASSES:
        class_dir = Path(SOURCE_DIR) / cls
        files = [str(p) for p in class_dir.iterdir() if p.is_file()]

        print(f"{cls}: found {len(files)} images")

        train_files, val_files, test_files = split_files(
            files, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
        )

        copy_files(train_files, os.path.join(OUTPUT_DIR, "train", cls))
        copy_files(val_files, os.path.join(OUTPUT_DIR, "val", cls))
        copy_files(test_files, os.path.join(OUTPUT_DIR, "test", cls))

        print(f"{cls}:")
        print(f"  train = {len(train_files)}")
        print(f"  val   = {len(val_files)}")
        print(f"  test  = {len(test_files)}")

    print("Done! Dataset split created inside 'data/'")


if __name__ == "__main__":
    main()