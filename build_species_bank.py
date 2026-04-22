from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from src.species_similarity import get_feature_extractor, get_image_transform, embed_pil_image, get_device


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "species_bank.pt"

CANDIDATE_ROOTS = [
    BASE_DIR / "species_dataset" / "data" / "data",
    BASE_DIR / "species_dataset" / "data",
    BASE_DIR / "species_dataset",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def find_species_root():
    for candidate in CANDIDATE_ROOTS:
        if candidate.exists() and candidate.is_dir():
            subdirs = [p for p in candidate.iterdir() if p.is_dir()]
            if len(subdirs) > 0:
                return candidate
    raise FileNotFoundError("No se encontró la carpeta raíz del dataset de especies.")


def main():
    species_root = find_species_root()
    print(f"Usando dataset de especies en: {species_root}")

    model, device = get_feature_extractor()
    species_names = []
    sample_paths = []
    centroids = []
    counts = []

    species_dirs = sorted([p for p in species_root.iterdir() if p.is_dir()])

    for species_dir in tqdm(species_dirs, desc="Building species bank"):
        image_paths = sorted([
            p for p in species_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ])

        if not image_paths:
            continue

        embeddings = []
        for img_path in image_paths:
            try:
                image = Image.open(img_path).convert("RGB")
                emb = embed_pil_image(image, model, device)
                embeddings.append(emb.cpu())
            except Exception as e:
                print(f"Saltando {img_path.name}: {e}")

        if not embeddings:
            continue

        centroid = torch.stack(embeddings).mean(dim=0)
        centroid = torch.nn.functional.normalize(centroid, dim=0)

        species_names.append(species_dir.name)
        sample_paths.append(str(image_paths[0]))
        centroids.append(centroid)
        counts.append(len(embeddings))

    bank = {
        "species_names": species_names,
        "sample_paths": sample_paths,
        "centroids": torch.stack(centroids),
        "counts": counts,
    }

    torch.save(bank, OUTPUT_PATH)

    print("\nBanco de especies creado.")
    print(f"Clases guardadas: {len(species_names)}")
    print(f"Archivo: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()