# train.py 
import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score
from torchvision import models

# ensure project root is on sys.path when running directly
proj_root = os.path.dirname(os.path.abspath(__file__))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from src.data_utils import CarClassificationDataset
from src.model import create_classification_model  # fallback small net

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def get_transforms(image_size=224):
    train_tf = A.Compose([
        A.Resize(image_size, image_size),
        A.RandomCrop(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.08, rotate_limit=12, p=0.4),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    val_tf = A.Compose([
        A.Resize(image_size, image_size),
        A.CenterCrop(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    return train_tf, val_tf

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='data/train')
    parser.add_argument('--val_dir', type=str, default='data/val')
    parser.add_argument('--classes', type=str, default='classes.txt')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--backbone', type=str, default='resnet50', help="e.g. resnet50 or small")
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--use_pretrained', action='store_true', help='Use pretrained weights for supported backbones')
    parser.add_argument('--freeze_epochs', type=int, default=3, help='Freeze backbone for first N epochs (set 0 to not freeze)')
    parser.add_argument('--workers', type=int, default=None, help='num_workers for DataLoader (auto if not provided)')
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    return parser.parse_args()

def compute_class_weights(train_dir, classes_file):
    # inverse frequency weighting
    from glob import glob
    classes = [l.strip() for l in open(classes_file) if l.strip()]
    counts = []
    for c in classes:
        n = len(glob(os.path.join(train_dir, c, "*")))
        counts.append(n if n>0 else 0)
    total = sum(counts) + 1e-6
    weights = [total/(c+1e-6) for c in counts]
    # normalize to mean 1
    mean_w = sum(weights)/len(weights)
    weights = [w/mean_w for w in weights]
    return torch.tensor(weights, dtype=torch.float)

def build_model(backbone_name, num_classes, use_pretrained=False, device='cpu'):
    if backbone_name and backbone_name.lower().startswith('resnet'):
        # support resnet50, resnet18 etc.
        if '50' in backbone_name:
            model = models.resnet50(pretrained=use_pretrained)
        elif '18' in backbone_name:
            model = models.resnet18(pretrained=use_pretrained)
        else:
            model = models.resnet50(pretrained=use_pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        print(f"Built torchvision {backbone_name} (pretrained={use_pretrained}) -> fc={in_features}->{num_classes}")
        return model.to(device)
    else:
        # fallback to your small convnet factory
        print("Using custom small convnet (create_classification_model)")
        model = create_classification_model(num_classes=num_classes, backbone_name=backbone_name)
        return model.to(device)

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

    # build datasets
    print("Creating datasets...")
    train_ds = CarClassificationDataset(args.train_dir, args.classes, transform=train_tf)
    val_ds = CarClassificationDataset(args.val_dir, args.classes, transform=val_tf)

    print(f"Dataset sizes -> train: {len(train_ds)} , val: {len(val_ds)}")
    if len(train_ds) == 0:
        raise SystemExit("ERROR: training dataset is empty. Check data/train and classes.txt.")
    if len(val_ds) == 0:
        print("WARNING: validation dataset is empty. You can proceed but evaluation won't run properly.")

    # DataLoader workers
    if args.workers is not None:
        num_workers = args.workers
    else:
        # safe defaults: 0 on windows / cpu; else 4
        num_workers = 0 if (os.name == 'nt' or device == 'cpu') else 4
    print(f"Using DataLoader num_workers={num_workers}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device!='cpu'))
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device!='cpu'))

    # model
    print(f"Creating model: {args.backbone} with {num_classes} classes")
    model = build_model(args.backbone, num_classes, use_pretrained=args.use_pretrained, device=device)

    # optionally freeze backbone parameters
    freeze_epochs = args.freeze_epochs if args.freeze_epochs is not None else 0
    if freeze_epochs > 0 and args.backbone.lower().startswith('resnet'):
        for name, p in model.named_parameters():
            if not name.startswith('fc'):
                p.requires_grad = False
        print(f"Backbone frozen for first {freeze_epochs} epochs (only fc will train)")

    # loss + class weights
    try:
        class_weights = compute_class_weights(args.train_dir, args.classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using class weights in loss")
    except Exception as ex:
        print("Could not compute class weights:", ex)
        criterion = nn.CrossEntropyLoss()

    # optimizer and scheduler
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_acc = 0.0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
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

        train_loss = running_loss / len(train_ds)
        print(f"Train loss: {train_loss:.4f}  processed: {processed}  time: {time.time()-t0:.1f}s")

        # unfreeze if we've completed freeze_epochs
        if freeze_epochs and epoch == freeze_epochs:
            print("Unfreezing backbone for fine-tuning...")
            for name, p in model.named_parameters():
                p.requires_grad = True
            # reset optimizer to include all params (lower lr)
            optimizer = optim.AdamW(model.parameters(), lr=args.lr * 0.2, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
            print("New optimizer created for full fine-tune; lr scaled down")

        # validation
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                pred_labels = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(pred_labels.tolist())
                all_targets.extend(labels.numpy().tolist())
        if len(all_targets) > 0:
            val_acc = accuracy_score(all_targets, all_preds)
        else:
            val_acc = 0.0
        print(f"Val acc: {val_acc:.4f}")

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = os.path.join(args.output_dir, f'best_{args.backbone}.pth')
            torch.save(model.state_dict(), ckpt)
            print("Saved best model:", ckpt)

        scheduler.step()

    print("Training finished. Best val acc:", best_val_acc)

if __name__ == '__main__':
    args = parse_args()
    train(args)
