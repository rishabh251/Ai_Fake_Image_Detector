import os
import shutil
import csv
import random

# ========= CONFIG =========
RAW_DATA_DIR = "datasets/train"
OUT_IMAGE_DIR = "dataset"
OUT_CSV_DIR = "data"
TRAIN_SPLIT = 0.8
VALID_EXTS = (".jpg", ".jpeg", ".png")

IGNORE_PATTERNS = ["(2)"]  # skip duplicate Windows files

LABEL_MAP = {
    "REAL": 0,
    "FAKE": 1
}

# ==========================


def should_ignore(filename: str) -> bool:
    return any(p in filename for p in IGNORE_PATTERNS)


def collect_images():
    samples = []

    for class_name, label in LABEL_MAP.items():
        src_dir = os.path.join(RAW_DATA_DIR, class_name)
        if not os.path.isdir(src_dir):
            raise FileNotFoundError(f"Missing folder: {src_dir}")

        for root, _, files in os.walk(src_dir):
            for f in files:
                if not f.lower().endswith(VALID_EXTS):
                    continue
                if should_ignore(f):
                    continue

                src_path = os.path.join(root, f)
                samples.append((src_path, label))

    return samples


def prepare_folders():
    for cls in ["real", "ai"]:
        os.makedirs(os.path.join(OUT_IMAGE_DIR, cls), exist_ok=True)
    os.makedirs(OUT_CSV_DIR, exist_ok=True)


def copy_images(samples):
    new_samples = []

    for src_path, label in samples:
        cls_name = "real" if label == 0 else "ai"
        fname = os.path.basename(src_path)

        dst_path = os.path.join(OUT_IMAGE_DIR, cls_name, fname)
        shutil.copy2(src_path, dst_path)

        new_samples.append((dst_path, label))

    return new_samples


def write_csv(train, val):
    train_csv = os.path.join(OUT_CSV_DIR, "train.csv")
    val_csv = os.path.join(OUT_CSV_DIR, "val.csv")

    with open(train_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(train)

    with open(val_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(val)

    print(f"[OK] train.csv: {len(train)} samples")
    print(f"[OK] val.csv  : {len(val)} samples")


def main():
    print("[1/4] Preparing folders...")
    prepare_folders()

    print("[2/4] Collecting raw images...")
    samples = collect_images()
    print(f"    Found {len(samples)} valid images")

    print("[3/4] Copying images to dataset/ ...")
    samples = copy_images(samples)

    print("[4/4] Creating CSV files...")
    random.shuffle(samples)

    split_idx = int(len(samples) * TRAIN_SPLIT)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    write_csv(train_samples, val_samples)

    print("\n✅ DATA PREPARATION COMPLETE")
    print("You can now train the CNN.")


if __name__ == "__main__":
    main()
