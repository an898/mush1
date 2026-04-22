import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.model import build_model


def load_model(model_path="best_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)

    class_names = checkpoint["class_names"]
    model = build_model(num_classes=len(class_names), fine_tune=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names, device


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def predict_image(image_path, model_path="best_model.pth"):
    model, class_names, device = load_model(model_path)
    transform = get_transform()

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()

    return class_names[pred_idx], confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict mushroom class from an image.")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("--model", type=str, default="best_model.pth", help="Path to the trained model")

    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    label, confidence = predict_image(str(image_path), args.model)
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2%}")