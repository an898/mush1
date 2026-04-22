from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_feature_extractor():
    device = get_device()
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.to(device)
    model.eval()
    return model, device


def get_image_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_species_bank(bank_path):
    return torch.load(bank_path, map_location="cpu")


def embed_pil_image(image, model, device):
    transform = get_image_transform()
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(tensor)

    emb = F.normalize(emb, dim=1)
    return emb.squeeze(0)


def find_most_similar(image, bank, model, device):
    query_emb = embed_pil_image(image, model, device)
    centroids = bank["centroids"].to(device)

    sims = torch.matmul(centroids, query_emb)
    best_idx = int(torch.argmax(sims).item())

    species_name = bank["species_names"][best_idx]
    sample_path = bank["sample_paths"][best_idx]
    similarity = float(sims[best_idx].item())

    return species_name, sample_path, similarity