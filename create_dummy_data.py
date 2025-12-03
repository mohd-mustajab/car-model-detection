# create_dummy_data.py
import os
from PIL import Image, ImageDraw, ImageFont

CLASS_FILE = 'classes.txt'
TRAIN_DIR = 'data/train'
VAL_DIR = 'data/val'

def read_classes(fn):
    if not os.path.exists(fn):
        with open(fn, 'w') as f:
            f.write('toyota_corolla\nhonda_civic\nmaruti_swift\n')
    with open(fn, 'r') as f:
        return [l.strip() for l in f if l.strip()]

def make_image(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.new('RGB', (320, 240), color=(200, 200, 200))
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.load_default()
    except:
        f = None
    d.text((10, 10), text, fill=(10,10,10), font=f)
    img.save(path, quality=85)

def main():
    classes = read_classes(CLASS_FILE)
    print("Using classes:", classes)
    for cls in classes:
        for i in range(5):
            make_image(os.path.join(TRAIN_DIR, cls, f'{cls}_train_{i}.jpg'), f'{cls} train {i}')
        for i in range(2):
            make_image(os.path.join(VAL_DIR, cls, f'{cls}_val_{i}.jpg'), f'{cls} val {i}')
    print("Dummy data created under data/train and data/val")

if __name__ == '__main__':
    main()
