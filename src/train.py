import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score

# local imports
# ensure project root is on sys.path when running directly (safety)
proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from src.data_utils import CarClassificationDataset
from src.model import create_classification_model

def get_transforms(image_size=224):
    train_tf = A.Compose([
        A.Resize(image_size, image_size),
        A.RandomCrop(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.4),
        A.Normalize(),
        ToTensorV2(),
    ])
    val_tf = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(),
        ToTensorV2(),
    ])
    return train_tf, val_tf

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='data/train')
    parser.add_argument('--val_dir', type=str, default='data/val')
    parser.add_argument('--classes', type=str, default='classes.txt')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--image_size', type=int, default=224)
    return parser.parse_args()

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)

    # read classes file
    if not os.path.exists(args.classes):
        raise SystemExit(f"ERROR: classes file not found: {args.classes}")
    with open(args.classes, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f if line.strip()]
    num_classes = len(classes)
    print(f"Loaded classes ({num_classes}) from {args.classes}")

    # transforms
    train_tf, val_tf = get_transforms(args.image_size)

    # build datasets (this may enumerate files — print debug)
    print("Creating datasets...")
    train_ds = CarClassificationDataset(args.train_dir, args.classes, transform=train_tf)
    val_ds = CarClassificationDataset(args.val_dir, args.classes, transform=val_tf)

    print(f"Dataset sizes -> train: {len(train_ds)} , val: {len(val_ds)}")
    if len(train_ds) == 0:
        raise SystemExit("ERROR: training dataset is empty. Check data/train and classes.txt.")
    if len(val_ds) == 0:
        print("WARNING: validation dataset is empty. You can proceed but evaluation won't run properly.")

    # choose num_workers safely: use 0 on Windows or if running on CPU to avoid spawn hang
    num_workers = 0 if (os.name == 'nt' or device == 'cpu') else 4
    print(f"Using DataLoader num_workers={num_workers}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    # model
    print(f"Creating model: {args.backbone} with {num_classes} classes")
    model = create_classification_model(num_classes=num_classes, backbone_name=args.backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_acc = 0.0
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        model.train()
        running_loss = 0.0
        processed = 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            preds = model(imgs)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            processed += imgs.size(0)
            if processed % (args.batch_size * 10) == 0:
                print(f"  processed {processed} samples, current loss: {loss.item():.4f}")

        train_loss = running_loss / len(train_ds)
        print(f"Train loss: {train_loss:.4f}")

        # validation
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                preds = model(imgs)
                pred_labels = preds.argmax(dim=1).cpu().numpy()
                all_preds.extend(pred_labels.tolist())
                all_targets.extend(labels.numpy().tolist())
        if len(all_targets) > 0:
            val_acc = accuracy_score(all_targets, all_preds)
        else:
            val_acc = 0.0
        print(f"Val acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = f'checkpoints/best_{args.backbone}.pth'
            torch.save(model.state_dict(), ckpt)
            print("Saved best model:", ckpt)

if __name__ == '__main__':
    args = parse_args()
    train(args)
