# backend/app.py
import os
import sys
import io
import traceback
from typing import List, Tuple
from flask import Flask, request, jsonify, render_template_string
from PIL import Image
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from flask_cors import CORS


# ---- make project root importable so `src` package works ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ---- imports from your src/ module (expects src/model.py to exist) ----
try:
    from src.model import create_classification_model
except Exception as e:
    raise ImportError(f"Unable to import src.model: {e}") from e

# ---- config (can be overridden using env vars) ----
CHECKPOINT = os.environ.get("CHECKPOINT", os.path.join(BASE_DIR, "checkpoints", "best_resnet50.pth"))
CLASSES_FILE = os.environ.get("CLASSES_FILE", os.path.join(BASE_DIR, "classes.txt"))
BACKBONE = os.environ.get("BACKBONE", "resnet50")
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "224"))
TOPK = int(os.environ.get("TOPK", "3"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- helper: load classes.txt from project root ----
if not os.path.exists(CLASSES_FILE):
    raise FileNotFoundError(f"classes file not found at: {CLASSES_FILE}")
CLASSES: List[str] = [line.strip() for line in open(CLASSES_FILE, "r", encoding="utf-8") if line.strip()]
NUM_CLASSES = len(CLASSES)
if NUM_CLASSES == 0:
    raise RuntimeError(f"classes.txt is empty: {CLASSES_FILE}")

# ---- transforms ----
def get_transform(image_size=224):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(),
        ToTensorV2(),
    ])

TRANSFORM = get_transform(IMAGE_SIZE)

# ---- robust checkpoint loader ----
def load_checkpoint_safe(checkpoint_path: str, num_classes: int, backbone: str):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    size = os.path.getsize(checkpoint_path)
    if size == 0:
        raise EOFError(f"Checkpoint file is empty (size=0): {checkpoint_path}")

    ext = os.path.splitext(checkpoint_path)[1].lower()
    model = create_classification_model(num_classes=num_classes, backbone_name=backbone)

    # Try safetensors first if extension indicates
    if ext == ".safetensors":
        try:
            from safetensors.torch import load_file as safetensors_load
        except Exception as e:
            raise ImportError("safetensors not installed. Install with `pip install safetensors`") from e
        sd = safetensors_load(checkpoint_path)
        # sd is a dict of tensors; convert keys if needed
        try:
            model.load_state_dict(sd)
            return model
        except Exception as e:
            # try to strip "module." prefix keys if present
            new_sd = {k.replace("module.", ""): v for k, v in sd.items()}
            model.load_state_dict(new_sd)
            return model

    # Otherwise try torch.load (handles .pth / .pt)
    try:
        state = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:
        # Torch failed to load — likely corrupted/truncated
        tb = traceback.format_exc()
        raise RuntimeError(f"torch.load failed for {checkpoint_path}.\nError: {e}\nTraceback:\n{tb}")

    # If state is an actual model object (rare), return it
    if not isinstance(state, dict):
        # assume full model was saved
        try:
            model = state
            return model
        except Exception:
            raise RuntimeError("Loaded checkpoint is not a state-dict and could not be used as a model object.")

    # If state is a dict, extract state_dict
    sd = None
    if "state_dict" in state:
        sd = state["state_dict"]
    elif "model_state_dict" in state:
        sd = state["model_state_dict"]
    else:
        sd = state

    # fix keys with "module." prefix if necessary
    try:
        model.load_state_dict(sd)
        return model
    except RuntimeError as e:
        # try removing "module." prefix
        new_sd = {}
        for k, v in sd.items():
            new_key = k.replace("module.", "") if k.startswith("module.") else k
            new_sd[new_key] = v
        try:
            model.load_state_dict(new_sd)
            return model
        except Exception as e2:
            raise RuntimeError(f"Failed to load state_dict into model. Original error: {e}\nAfter stripping 'module.': {e2}")

# ---- load model at startup (with helpful messages) ----
print(f"Starting Flask app. Device={DEVICE}. Loading model from: {CHECKPOINT}")
try:
    MODEL = load_checkpoint_safe(CHECKPOINT, num_classes=NUM_CLASSES, backbone=BACKBONE)
    MODEL.to(DEVICE)
    MODEL.eval()
    print("Model loaded successfully.")
except Exception as e:
    # re-raise with clear message (Flask will show it)
    raise RuntimeError(f"Failed to load model checkpoint: {e}")

# ---- inference helpers ----
def preprocess_imagefile(file_bytes: bytes, transform):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = np.array(img)
    t = transform(image=img)['image'].unsqueeze(0)  # 1,C,H,W
    return t

def predict_topk(model, input_tensor: torch.Tensor, classes: List[str], topk:int=3):
    input_tensor = input_tensor.to(DEVICE)
    with torch.no_grad():
        out = model(input_tensor)
        probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy()[0]
    topk_idx = probs.argsort()[-topk:][::-1]
    return [(classes[int(i)], float(probs[int(i)])) for i in topk_idx]

# ---- Flask app ----
app = Flask(__name__)
CORS(app) 
INDEX_HTML = """
<!doctype html>
<title>Car Model Classifier (Flask)</title>
<h2>Upload image to predict car model</h2>
<form action="/predict" method="post" enctype="multipart/form-data">
  <input type="file" name="file" accept="image/*">
  <input type="submit" value="Upload & Predict">
</form>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error":"no file part in request"}), 400
    file = request.files['file']
    if file.filename == "":
        return jsonify({"error":"empty filename"}), 400
    # read bytes
    try:
        contents = file.read()
        inp = preprocess_imagefile(contents, TRANSFORM)
    except Exception as e:
        return jsonify({"error": f"invalid image or preprocessing failed: {e}"}), 400

    try:
        preds = predict_topk(MODEL, inp, CLASSES, topk=TOPK)
    except Exception as e:
        return jsonify({"error": f"inference failed: {e}"}), 500

    return jsonify({"predictions": [{"label": p[0], "score": p[1]} for p in preds]})

# ---- run server (for development only) ----
if __name__ == "__main__":
    # Running Flask directly is fine for local dev.
    # In production, use Gunicorn / uWSGI + Nginx.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))


