# src/infer_single.py
import torch, sys, os
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from src.model import create_classification_model

def get_transform(image_size=224):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(),
        ToTensorV2(),
    ])

def load_model(path, classes_file, backbone='resnet50'):
    classes = [c.strip() for c in open(classes_file, 'r', encoding='utf-8') if c.strip()]
    num_classes = len(classes)
    model = create_classification_model(num_classes=num_classes, backbone_name=backbone)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model, classes

def predict(image_path, model, transform, classes):
    img = np.array(Image.open(image_path).convert('RGB'))
    input = transform(image=img)['image'].unsqueeze(0)
    with torch.no_grad():
        out = model(input)
        probs = torch.nn.functional.softmax(out, dim=1)[0].cpu().numpy()
    top_idx = int(np.argmax(probs))
    return classes[top_idx], float(probs[top_idx]), probs

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str)
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_resnet50.pth')
    parser.add_argument('--classes', type=str, default='classes.txt')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--image_size', type=int, default=224)
    args = parser.parse_args()

    model, classes = load_model(args.checkpoint, args.classes, backbone=args.backbone)
    transform = get_transform(args.image_size)
    label, score, probs = predict(args.image, model, transform, classes)
    print("Pred:", label, "score:", score)
