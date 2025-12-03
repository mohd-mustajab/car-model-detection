import os
import shutil
import random
from glob import glob

# Paths (edit only if needed)
TRAIN_DIR = 'data/train'
VAL_DIR = 'data/val'

# 15% of training images will be copied to val/
VAL_SPLIT = 0.15

SEED = 42
random.seed(SEED)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_images(folder):
    exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    imgs = []
    for e in exts:
        imgs.extend(glob(os.path.join(folder, e)))
    return [i for i in imgs if os.path.getsize(i) > 0]

def main():
    if not os.path.isdir(TRAIN_DIR):
        print(f"ERROR: Training directory '{TRAIN_DIR}' does not exist.")
        return

    ensure_dir(VAL_DIR)

    class_folders = [d for d in os.listdir(TRAIN_DIR)
                     if os.path.isdir(os.path.join(TRAIN_DIR, d))]
    
    if not class_folders:
        print("ERROR: No class folders found inside data/train/")
        return

    print(f"Found {len(class_folders)} classes. Creating validation split...\n")

    for cls in class_folders:
        train_cls_path = os.path.join(TRAIN_DIR, cls)
        val_cls_path = os.path.join(VAL_DIR, cls)
        ensure_dir(val_cls_path)

        images = get_images(train_cls_path)
        if len(images) == 0:
            print(f"[WARN] No images found for class '{cls}', skipping.")
            continue

        k = max(1, int(len(images) * VAL_SPLIT))
        selected = random.sample(images, k)

        for idx, img_path in enumerate(selected):
            fname = os.path.basename(img_path)
            dst = os.path.join(val_cls_path, fname)
            
            # prevent overwriting
            if os.path.exists(dst):
                base, ext = os.path.splitext(fname)
                dst = os.path.join(val_cls_path, f"{base}_copy{idx}{ext}")

            shutil.copy2(img_path, dst)

        print(f"{cls}: copied {k} images to validation set.")

    print("\n[✓] Validation dataset created successfully!")
    print(f"[✓] Check here: {VAL_DIR}")

if __name__ == "__main__":
    main()
