# src/data_utils.py
import os
from glob import glob
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from torchvision import transforms

class CarClassificationDataset(Dataset):
    """
    Simple dataset for classification where folder layout is:
    data/train/<class_name>/*.jpg
    data/val/<class_name>/*.jpg
    """
    def __init__(self, root_dir, classes_file, transform=None):
        self.root_dir = root_dir
        # read classes list
        with open(classes_file, 'r') as f:
            self.classes = [c.strip() for c in f if c.strip()]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for p in glob(os.path.join(cls_dir, '*')):
                if p.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((p, self.class_to_idx[cls]))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = np.array(Image.open(path).convert('RGB'))
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            # fallback: simple to-tensor via torchvision
            image = transforms.ToTensor()(Image.fromarray(image))
        return image, label
