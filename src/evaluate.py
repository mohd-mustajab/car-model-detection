# src/evaluate.py
import os
import csv
import torch
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, top_k_accuracy_score
import matplotlib.pyplot as plt
import itertools

# ensure project root import works
import sys
proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from src.data_utils import CarClassificationDataset
from src.model import create_classification_model
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transform(image_size=224):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(),
        ToTensorV2(),
    ])

def plot_confusion_matrix(cm, classes, out_path, normalize=False, cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix' + (' (normalized)' if normalize else ''))
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=6)
    plt.yticks(tick_marks, classes, fontsize=6)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        val = cm[i, j]
        if normalize:
            s = f"{val:.2f}"
        else:
            s = f"{int(val)}"
        plt.text(j, i, s, horizontalalignment="center", verticalalignment="center",
                 fontsize=4, color="white" if val > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(out_path, dpi=200)
    plt.close()

def evaluate(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)

    classes = [c.strip() for c in open(args.classes, 'r', encoding='utf-8') if c.strip()]
    num_classes = len(classes)
    print("Num classes:", num_classes)

    transform = get_transform(args.image_size)
    test_ds = CarClassificationDataset(args.test_dir, args.classes, transform=transform)
    print("Test samples:", len(test_ds))
    if len(test_ds) == 0:
        raise SystemExit("No test samples found. Check test_dir and classes.txt")

    # dataloader
    num_workers = 0 if (os.name == 'nt' or device == 'cpu') else 4
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    # model
    model = create_classification_model(num_classes=num_classes, backbone_name=args.backbone)
    ckpt = args.checkpoint
    print("Loading checkpoint:", ckpt)
    state = torch.load(ckpt, map_location=device)
    # if saved whole model object, handle that:
    if isinstance(state, dict) and 'state_dict' in state:
        model.load_state_dict(state['state_dict'])
    elif isinstance(state, dict):
        # assume state is state_dict
        model.load_state_dict(state)
    else:
        # maybe the file is a full model saved via torch.save(model)
        model = state
    model.to(device)
    model.eval()

    all_targets = []
    all_preds = []
    all_probs = []

    misclassified = []

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())
            all_targets.extend(labels.numpy().tolist())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    top1 = (all_preds == all_targets).mean()
    try:
        top5 = top_k_accuracy_score(all_targets, all_probs, k=5, labels=range(num_classes))
    except Exception:
        top5 = None

    print(f"Top-1 accuracy: {top1:.4f}")
    if top5 is not None:
        print(f"Top-5 accuracy: {top5:.4f}")

    # confusion matrix
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
    print("Confusion matrix computed.")

    # per-class precision/recall
    prec, rec, f1, sup = precision_recall_fscore_support(all_targets, all_preds, labels=list(range(num_classes)), zero_division=0)
    # write per-class CSV
    perclass_csv = os.path.join(args.output_dir, "per_class_metrics.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(perclass_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["class_index", "class_name", "precision", "recall", "f1", "support"])
        for i, cname in enumerate(classes):
            writer.writerow([i, cname, f"{prec[i]:.4f}", f"{rec[i]:.4f}", f"{f1[i]:.4f}", int(sup[i])])
    print("Wrote per-class metrics to", perclass_csv)

    # misclassified examples: iterate test dataset to get filenames
    # test_ds.samples contains tuples (path, label)
    mis_csv = os.path.join(args.output_dir, "misclassified.csv")
    with open(mis_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "true_label_idx", "true_label", "pred_label_idx", "pred_label", "pred_prob"])
        for idx, (path, true_label) in enumerate(test_ds.samples):
            pred = all_preds[idx]
            prob = all_probs[idx][pred]
            if pred != true_label:
                writer.writerow([path, true_label, classes[true_label], pred, classes[pred], f"{prob:.4f}"])
    print("Wrote misclassified examples to", mis_csv)

    # save confusion matrices (raw and normalized)
    plot_confusion_matrix(cm, classes, os.path.join(args.output_dir, "confusion_matrix.png"), normalize=False)
    plot_confusion_matrix(cm, classes, os.path.join(args.output_dir, "confusion_matrix_norm.png"), normalize=True)
    print("Saved confusion matrix images in", args.output_dir)

    # summary file
    with open(os.path.join(args.output_dir, "summary.txt"), 'w', encoding='utf-8') as f:
        f.write(f"Top1: {top1:.6f}\n")
        if top5 is not None:
            f.write(f"Top5: {top5:.6f}\n")
        f.write(f"Num test samples: {len(test_ds)}\n")
    print("Evaluation complete. Summary written.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, default='data/test')
    parser.add_argument('--classes', type=str, default='classes.txt')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_resnet50.pth')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output_dir', type=str, default='experiments/eval')
    args = parser.parse_args()
    evaluate(args)
