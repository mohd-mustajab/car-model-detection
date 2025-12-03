import os
import shutil
import random
from glob import glob
from datetime import datetime
import argparse

# ---------- Config (change if you want) ----------
RAW_DIR = 'data/raw'
OUT_DIR = 'data'
CLASSES_FILE = 'classes.txt'
SEED = 42

TRAIN_RATIO = 0.75
VAL_RATIO = 0.15
TEST_RATIO = 0.10

IMAGE_EXTENSIONS = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')

# ---------- Helper functions ----------
def list_class_folders(directory):
    if not os.path.isdir(directory):
        return []
    return sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])

def safe_backup(path):
    """Move an existing folder to a timestamped backup (if it exists)."""
    if os.path.exists(path):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = f"{path}_backup_{ts}"
        print(f"[BACKUP] Moving '{path}' -> '{backup_path}'")
        shutil.move(path, backup_path)

def find_images_in_dir(dirpath):
    imgs = []
    for pat in IMAGE_EXTENSIONS:
        imgs.extend(glob(os.path.join(dirpath, pat)))
    # filter zero-size files
    imgs = [p for p in imgs if os.path.getsize(p) > 0]
    return imgs

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def unique_destination(dst_dir, fname):
    """Return a destination path that does not overwrite existing file by adding suffix if needed."""
    dst = os.path.join(dst_dir, fname)
    if not os.path.exists(dst):
        return dst
    base, ext = os.path.splitext(fname)
    i = 1
    while True:
        candidate = os.path.join(dst_dir, f"{base}_copy{i}{ext}")
        if not os.path.exists(candidate):
            return candidate
        i += 1

# ---------- Core actions ----------
def split_and_copy_from_raw(raw_dir, out_dir, classes):
    random.seed(SEED)
    for cls in classes:
        src_dir = os.path.join(raw_dir, cls)
        if not os.path.isdir(src_dir):
            print(f"[WARN] expected class folder missing: {src_dir} (skipping)")
            continue

        images = find_images_in_dir(src_dir)
        if not images:
            print(f"[WARN] no images found in {src_dir}; skipping class '{cls}'")
            continue

        random.shuffle(images)
        n = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        # guard: ensure at least 1 image in train if possible
        if n_train == 0 and n > 0:
            n_train = 1
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0

        splits = {
            'train': images[:n_train],
            'val': images[n_train:n_train + n_val],
            'test': images[n_train + n_val:]
        }

        for split_name, files in splits.items():
            out_cls_dir = os.path.join(out_dir, split_name, cls)
            ensure_dir(out_cls_dir)
            for p in files:
                fname = os.path.basename(p)
                dst = unique_destination(out_cls_dir, fname)
                shutil.copy2(p, dst)
        print(f"[INFO] {cls}: total={n}, train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

def write_classes_file(classes, classes_file):
    with open(classes_file, 'w', encoding='utf-8') as f:
        for c in classes:
            f.write(c + '\n')
    print(f"[INFO] wrote classes list to {classes_file} ({len(classes)} classes)")

def rebuild_classes_from_existing(out_dir):
    found = set()
    for split in ('train', 'val', 'test'):
        p = os.path.join(out_dir, split)
        if os.path.isdir(p):
            for d in os.listdir(p):
                full = os.path.join(p, d)
                if os.path.isdir(full):
                    found.add(d)
    return sorted(found)

# ---------- Main ----------
def main(args):
    print("=== prepare_data.py (safe mode) ===")

    raw_exists = os.path.isdir(RAW_DIR)
    any_splits_exist = any(os.path.isdir(os.path.join(OUT_DIR, s)) for s in ('train', 'val', 'test'))

    # If raw does not exist but there are already splits, only rebuild classes.txt and exit.
    if not raw_exists and any_splits_exist:
        print("[INFO] data/raw/ not found, but train/val/test exist. Rebuilding classes.txt from existing splits (no copying).")
        classes = rebuild_classes_from_existing(OUT_DIR)
        if not classes:
            print("[WARN] No class folders found in train/val/test. Nothing to do.")
            return
        write_classes_file(classes, CLASSES_FILE)
        print("[DONE] Rebuilt classes.txt. Exiting.")
        return

    # If raw does not exist and no splits, nothing to do.
    if not raw_exists and not any_splits_exist:
        print("[ERROR] data/raw/ not found and train/val/test do not exist. Put your dataset in data/raw/<class>/ and re-run.")
        return

    # raw exists -> prepare splits, but first backup any existing train/val/test
    print(f"[INFO] data/raw found. Preparing splits from '{RAW_DIR}' into '{OUT_DIR}/train','{OUT_DIR}/val','{OUT_DIR}/test'.")
    classes = list_class_folders(RAW_DIR)
    if not classes:
        print("[ERROR] No class folders found inside data/raw/. Please put your images under data/raw/<class_name>/")
        return

    # Backup existing splits safely (move them)
    for s in ('train', 'val', 'test'):
        folder = os.path.join(OUT_DIR, s)
        if os.path.exists(folder):
            safe_backup(folder)
        ensure_dir(os.path.join(OUT_DIR, s))

    # copy/split
    split_and_copy_from_raw(RAW_DIR, OUT_DIR, classes)

    # write classes file
    write_classes_file(classes, CLASSES_FILE)

    print("[DONE] Dataset preparation complete. Existing split folders were backed up (if present).")

# ---------- CLI ----------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Safe prepare_data script. See file header for behavior.")
    parser.add_argument('--raw_dir', type=str, default=RAW_DIR, help='Source raw data directory (default: data/raw)')
    parser.add_argument('--out_dir', type=str, default=OUT_DIR, help='Output data directory (default: data)')
    parser.add_argument('--classes_file', type=str, default=CLASSES_FILE, help='Path to classes.txt to write (default: classes.txt)')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed for shuffling (default: 42)')
    parser.add_argument('--train_ratio', type=float, default=TRAIN_RATIO, help='Train split ratio (default: 0.75)')
    parser.add_argument('--val_ratio', type=float, default=VAL_RATIO, help='Val split ratio (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=TEST_RATIO, help='Test split ratio (default: 0.10)')
    args = parser.parse_args()

    # apply CLI overrides
    RAW_DIR = args.raw_dir
    OUT_DIR = args.out_dir
    CLASSES_FILE = args.classes_file
    SEED = args.seed
    TRAIN_RATIO = args.train_ratio
    VAL_RATIO = args.val_ratio
    TEST_RATIO = args.test_ratio

    main(args)
