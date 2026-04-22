import csv
import random
from datetime import datetime
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from src.model import build_model

BASE_DIR = Path(__file__).resolve().parent
MEMES_DIR = BASE_DIR / "memes"
REFERENCE_ROOT = BASE_DIR / "reference_species"
FEEDBACK_FILE = BASE_DIR / "feedback.csv"
LOGO_PATH = BASE_DIR / "logo.png"
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

st.set_page_config(
    page_title="MushroomGuard",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main {
            background: #f7f8fb;
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }

        .card {
            background: white;
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 18px;
            padding: 1rem;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
            margin-bottom: 1rem;
        }

        .title {
            font-size: 2rem;
            font-weight: 800;
            margin: 0;
            color: #0f172a;
        }

        .subtitle {
            font-size: 1rem;
            color: #64748b;
            margin-top: 0.25rem;
        }

        .section-title {
            font-size: 1rem;
            font-weight: 800;
            margin-bottom: 0.6rem;
            color: #0f172a;
        }

        .small-note {
            color: #64748b;
            font-size: 0.92rem;
        }

        .result-good {
            background: #e8f7ef;
            color: #1f8a5b;
            border: 1px solid #c7ead8;
            border-radius: 14px;
            padding: 0.8rem 1rem;
            font-weight: 800;
        }

        .result-bad {
            background: #feeceb;
            color: #b42318;
            border: 1px solid #f5cbc6;
            border-radius: 14px;
            padding: 0.8rem 1rem;
            font-weight: 800;
        }

        .footer {
            color: #64748b;
            font-size: 0.82rem;
            margin-top: 0.8rem;
            padding-top: 0.8rem;
            border-top: 1px solid rgba(15, 23, 42, 0.08);
        }

        .stButton > button {
            width: 100%;
            border-radius: 12px;
            font-weight: 700;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.35rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model(model_path="best_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(BASE_DIR / model_path, map_location=device)
    class_names = checkpoint["class_names"]

    model = build_model(num_classes=len(class_names), fine_tune=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names, device


@st.cache_resource
def get_feature_extractor():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.to(device)
    model.eval()
    return model, device


@st.cache_resource
def load_reference_bank(group_name: str):
    group_dir = REFERENCE_ROOT / group_name
    if not group_dir.exists():
        return None

    feature_model, device = get_feature_extractor()

    species_names = []
    sample_paths = []
    centroids = []

    species_dirs = sorted([p for p in group_dir.iterdir() if p.is_dir()])

    for species_dir in species_dirs:
        image_paths = [
            p for p in species_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
        ]

        if not image_paths:
            continue

        image_paths = image_paths[:5]
        embeddings = []

        for img_path in image_paths:
            try:
                image = Image.open(img_path).convert("RGB")
                emb = extract_embedding(image, feature_model, device)
                embeddings.append(emb.cpu())
            except Exception:
                continue

        if not embeddings:
            continue

        centroid = torch.stack(embeddings).mean(dim=0)
        centroid = F.normalize(centroid, dim=0)

        species_names.append(species_dir.name)
        sample_paths.append(str(image_paths[0]))
        centroids.append(centroid)

    if not centroids:
        return None

    return {
        "species_names": species_names,
        "sample_paths": sample_paths,
        "centroids": torch.stack(centroids),
    }


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def extract_embedding(image, model, device):
    transform = get_transform()
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(image_tensor)

    emb = F.normalize(emb, dim=1)
    return emb.squeeze(0)


def find_most_similar(image, bank, model, device):
    query_emb = extract_embedding(image, model, device)
    centroids = bank["centroids"].to(device)

    sims = torch.matmul(centroids, query_emb)
    best_idx = int(torch.argmax(sims).item())

    species_name = bank["species_names"][best_idx]
    sample_path = bank["sample_paths"][best_idx]
    similarity = float(sims[best_idx].item())

    return species_name, sample_path, similarity


def predict_pil_image(image, model, class_names, device):
    transform = get_transform()
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    predicted_idx = torch.argmax(probs).item()
    predicted_class = class_names[predicted_idx]
    confidence = probs[predicted_idx].item()

    probabilities = {
        class_names[i]: float(probs[i].item())
        for i in range(len(class_names))
    }

    return predicted_class, confidence, probabilities


def get_confidence_label(confidence):
    if confidence < 0.65:
        return "low"
    elif confidence < 0.80:
        return "medium"
    return "high"


def get_random_meme(prediction):
    folder = MEMES_DIR / prediction.lower()
    if not folder.exists():
        return None

    files = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
    ]
    if not files:
        return None

    return random.choice(files)


def save_feedback(prediction, confidence, feedback, uploaded_name):
    file_exists = FEEDBACK_FILE.exists()

    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "filename", "prediction", "confidence", "feedback"])

        writer.writerow([
            datetime.now().isoformat(),
            uploaded_name,
            prediction,
            f"{confidence:.4f}",
            feedback
        ])


with st.sidebar:
    st.markdown("## 🍄 MushroomGuard")
    st.caption("Mushroom toxicity screening + visual similarity")

    st.markdown("**Group 5**")

    st.markdown(
        """
        <div style="font-size: 0.76rem; line-height: 1.15;">
        <b>Members</b><br>
        ANY WAN YING LIU LIU<br>
        ROCÍO CEPERO ORTIZ<br>
        PABLO GÓMEZ PRIETO<br>
        YARA MARÍN FERNÁNDEZ<br>
        MIGUEL ROMERO CLAVERO<br>
        IGNACIO JAVIER SEISDEDOS RICO<br>
        EMMA VIZOSO ROMERO
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="font-size: 0.78rem; line-height: 1.15;">
        <b>Professor</b><br>
        RUBÉN SÁNCHEZ GARCÍA<br>
        <span style="color: #6b7280;">Special thanks for the guidance and feedback ✨</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("### How it works")
    st.write("1. Upload a clear mushroom photo.")
    st.write("2. Read the edible/poisonous prediction.")
    st.write("3. Check the closest reference inside the same group.")
    st.write("4. Give feedback.")

    st.markdown("---")
    st.caption("Supported: raw mushroom photos")
    st.caption("Not ideal: cooked dishes, drawings, blurry images")


header_left, header_right = st.columns([1, 4], vertical_alignment="center")

with header_left:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=140)

with header_right:
    st.markdown(
        """
        <div class="card">
            <div class="title">MushroomGuard</div>
            <div class="subtitle">
                Deep learning for mushroom toxicity risk screening and visual similarity search.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.warning(
    "This prototype is for educational and risk-screening purposes only. Do not rely on it alone for real-world mushroom consumption decisions."
)

uploaded_file = st.file_uploader("Choose a mushroom image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    model, class_names, device = load_model()
    prediction, confidence, probabilities = predict_pil_image(image, model, class_names, device)
    confidence_level = get_confidence_label(confidence)

    meme_path = get_random_meme(prediction)
    bank = load_reference_bank(prediction.lower())

    left_col, right_col = st.columns([1.05, 1], gap="large")

    with left_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Uploaded image</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Prediction result</div>', unsafe_allow_html=True)

        if prediction.lower() == "poisonous":
            st.markdown(
                '<div class="result-bad">Predicted class: poisonous</div>',
                unsafe_allow_html=True
            )
            st.error("Warning: this mushroom may be poisonous.")
        else:
            st.markdown(
                '<div class="result-good">Predicted class: edible</div>',
                unsafe_allow_html=True
            )
            st.success("This mushroom is predicted as likely edible.")

        st.metric("Confidence", f"{confidence:.2%}")
        st.caption(f"Confidence level: {confidence_level}")
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Meme verdict</div>', unsafe_allow_html=True)
        if meme_path is not None:
            st.image(str(meme_path), caption=f"{prediction} meme", use_container_width=True)
        else:
            st.info(f"No meme found for class: {prediction}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Class probabilities</div>', unsafe_allow_html=True)
        for class_name, prob in probabilities.items():
            st.write(f"**{class_name.capitalize()}**")
            st.progress(float(prob))
            st.caption(f"{prob:.2%}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Most similar mushroom type</div>', unsafe_allow_html=True)

        if bank is not None:
            try:
                feature_model, feature_device = get_feature_extractor()
                species_name, ref_path, similarity = find_most_similar(
                    image, bank, feature_model, feature_device
                )
                st.metric("Most similar type", species_name)
                st.metric("Similarity", f"{similarity:.2%}")
                st.caption("Visual resemblance reference, not taxonomic identification.")
                if ref_path and Path(ref_path).exists():
                    st.image(ref_path, caption=f"Reference image: {species_name}", use_container_width=True)
                else:
                    st.info("Reference image not found.")
            except Exception as e:
                st.info(f"Could not compute similarity for this image: {e}")
        else:
            st.info(f"No reference bank found for class: {prediction.lower()}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Feedback</div>', unsafe_allow_html=True)
    st.write("Was I right?")

    fb_left, fb_right = st.columns(2)

    with fb_left:
        if st.button("✅ Yes"):
            save_feedback(prediction, confidence, "yes", uploaded_file.name)
            st.success("Thanks for your feedback 🙌")

    with fb_right:
        if st.button("❌ No"):
            save_feedback(prediction, confidence, "no", uploaded_file.name)
            st.warning("Thanks — sorry about that. Hope you did not get poisoned 🙏")

    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown(
        """
        <div class="card">
            <div class="section-title">Start here</div>
            <p class="small-note">
                Upload a mushroom image to see the prediction, meme verdict, and the closest visual reference.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="footer">
        MushroomGuard · Student prototype for computer vision, interpretability, and risk-screening.<br>
        Group 5 · Special thanks to Rubén Sánchez García.
    </div>
    """,
    unsafe_allow_html=True,
)
