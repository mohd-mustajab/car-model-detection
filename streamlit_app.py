# streamlit_local_infer.py
import streamlit as st
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as tv_models
from PIL import Image
import numpy as np

# optionally import your small model factory if you've got one
from src.model import create_classification_model

CKPT_CANDIDATES = [
    Path("backend/checkpoints/best_resnet50.pth"),
    Path("checkpoints/best_resnet50.pth"),
    Path("checkpoints/best_small.pth"),
]
CLASSES_FILE = Path("classes.txt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224

def find_ckpt():
    for p in CKPT_CANDIDATES:
        if p.exists() and p.is_file() and p.stat().st_size > 100:
            return p
    raise FileNotFoundError("No checkpoint found in candidates.")

def load_classes(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def extract_state_dict(loaded):
    # accept dict wrapping or raw state_dict
    if isinstance(loaded, dict):
        for k in ("state_dict","model_state","model_state_dict","model"):
            if k in loaded and isinstance(loaded[k], dict):
                sd = loaded[k]; break
        else:
            sd = loaded
    else:
        raise RuntimeError("Unsupported checkpoint content")
    return {k.replace("module.", ""): v for k,v in sd.items()}

def looks_like_resnet(sd: dict):
    for k in sd.keys():
        if k.startswith("layer1.") or k.startswith("layer4."):
            return True
    return False

@st.cache_resource(show_spinner=False)
def load_model_and_classes():
    ckpt = find_ckpt()
    raw = torch.load(str(ckpt), map_location="cpu")
    sd = extract_state_dict(raw)

    classes = load_classes(CLASSES_FILE)
    num_classes = len(classes)

    if looks_like_resnet(sd) or "resnet" in ckpt.name.lower():
        model = tv_models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        # fallback to your SmallConvNet factory
        model = create_classification_model(num_classes=num_classes, backbone_name="small")

    # Try to load with best effort: strict then strict=False then partial matching
    try:
        model.load_state_dict(sd, strict=True)
    except Exception:
        try:
            model.load_state_dict(sd, strict=False)
        except Exception:
            # partial match: copy only matched shapes
            md = model.state_dict()
            matched = {k: v for k, v in sd.items() if k in md and tuple(md[k].shape) == tuple(v.shape)}
            md.update(matched)
            model.load_state_dict(md)  # will raise if nothing matched but usually okay

    model.eval()
    model.to(DEVICE)
    # preprocessing transform (same as backend)
    transform = T.Compose([
        T.Resize(IMAGE_SIZE),
        T.CenterCrop(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return model, classes, transform

def predict_topk(image: Image.Image, model, transform, classes, k=3):
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()
    topk_idx = np.argsort(probs)[-k:][::-1]
    return [(classes[i] if i < len(classes) else f"class_{i}", float(probs[i])) for i in topk_idx]

# ----------------- Streamlit UI -----------------
st.set_page_config(
    page_title="Car Model Detection",
    page_icon="🚗",  # Optional: You can also set a page icon (favicon)
    layout="wide",  # Optional: Other layout options like "centered"
    initial_sidebar_state="auto" # Optional: "expanded" or "collapsed"
)
st.title("🚗 Car Model Detection")
st.markdown("---")

model, classes, transform = load_model_and_classes()

uploaded = st.file_uploader("Upload a car image", type=["jpg", "jpeg", "png"])

# Store prediction state
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if uploaded is not None:
    # Preview image
    try:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, width=400)

        # Predict Button
        predict_btn = st.button("🔍 Predict", use_container_width=True)

        if predict_btn:
            with st.spinner("Running model... please wait ⏳"):
                try:
                    top3 = predict_topk(img, model, transform, classes, k=3)
                    st.session_state.prediction_done = True
                    st.session_state.results = top3
                except Exception as e:
                    st.error(f"Inference failed: {e}")
                    st.session_state.prediction_done = False

        # Display predictions
        if st.session_state.prediction_done:
            st.markdown("## 🎯 Top Predictions")
            for i, (label, score) in enumerate(st.session_state.results, start=1):
                st.markdown(
                    f"""
                    <div style="
                        padding:12px;
                        margin-bottom:10px;
                        border-radius:10px;
                        background:rgba(124,58,237,0.12);
                        border-left:4px solid #7c3aed;
                        animation: fadeIn 0.5s ease;
                    ">
                        <b>{i}. {label}</b>  
                        <br>
                        Confidence: <b>{(score * 100):.2f}%</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    except Exception as e:
        st.error(f"Error loading image: {e}")

# fade-in animation CSS
st.markdown("""
<style>
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0px); }
}
</style>
""", unsafe_allow_html=True)

st.footer = st.markdown("Deloped by Mohd Mustajab")
