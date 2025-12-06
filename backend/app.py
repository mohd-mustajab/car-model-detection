# backend/app.py (compact)
import sys
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as tv_models

# make project root importable for `from src...`
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import create_classification_model  # local small model factory

# config
CANDIDATES = [
    HERE / "checkpoints" / "best_resnet50.pth",
    PROJECT_ROOT / "checkpoints" / "best_resnet50.pth",
    PROJECT_ROOT / "checkpoints" / "best_small.pth",
    HERE / "checkpoints" / "best_small.pth",
]
CLASSES_FILE = PROJECT_ROOT / "./classes.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224

app = Flask(__name__)
CORS(app)  # allow CORS for local dev

# utils
def find_checkpoint():
    for p in CANDIDATES:
        try:
            if p.exists() and p.is_file() and p.stat().st_size > 100:
                return p
        except Exception:
            pass
    return None

def load_classes(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def extract_state_dict(loaded):
    if isinstance(loaded, dict):
        for k in ("state_dict", "model_state", "model_state_dict", "model"):
            if k in loaded and isinstance(loaded[k], dict):
                sd = loaded[k]; break
        else:
            sd = loaded
    else:
        raise RuntimeError("Unsupported checkpoint format")
    return { (k.replace("module.", "") if isinstance(k, str) else k): v for k,v in sd.items() }

def looks_like_resnet(sd):
    return any(k.startswith("layer1.") or k.startswith("layer2.") or k.startswith("layer3.") or k.startswith("layer4.") for k in sd.keys())

def load_matched_weights(model: nn.Module, sd: dict):
    md = model.state_dict()
    matched = {k: v for k, v in sd.items() if k in md and tuple(md[k].shape) == tuple(v.shape)}
    if not matched:
        raise RuntimeError("No matching params between checkpoint and model.")
    md.update(matched)
    model.load_state_dict(md)
    return model

# prepare model
CKPT = find_checkpoint()
if CKPT is None:
    raise SystemExit(f"No checkpoint found. Looked at: {CANDIDATES}")

CLASSES = load_classes(CLASSES_FILE)
NUM_CLASSES = len(CLASSES)

raw = torch.load(str(CKPT), map_location="cpu")
sd = extract_state_dict(raw)

if looks_like_resnet(sd):
    model = tv_models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
else:
    model = create_classification_model(num_classes=NUM_CLASSES, backbone_name="small")

# try to load matching weights (shape-checked)
try:
    model = load_matched_weights(model, sd)
except Exception:
    # fallback: try non-strict load (may still raise)
    try:
        model.load_state_dict(sd, strict=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")

MODEL = model.to(DEVICE)
MODEL.eval()

# preprocessing
transform = T.Compose([
    T.Resize(IMAGE_SIZE),
    T.CenterCrop(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def preprocess_image_file(f):
    img = Image.open(f).convert("RGB")
    return transform(img).unsqueeze(0)

# route
@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "empty filename"}), 400

    try:
        tensor = preprocess_image_file(f.stream).to(DEVICE)

        with torch.no_grad():
            logits = MODEL(tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()

            # --- TOP-3 PREDICTIONS ---
            top3_idx = probs.argsort()[-3:][::-1]  # descending order
            results = []
            for idx in top3_idx:
                label = CLASSES[idx] if idx < len(CLASSES) else f"class_{idx}"
                confidence = float(probs[idx])
                results.append({"label": label, "confidence": confidence})

        return jsonify({"predictions": results})

    except Exception as e:
        return jsonify({"error": "prediction failed", "detail": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
