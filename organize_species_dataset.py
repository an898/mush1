from pathlib import Path
import csv
import shutil

BASE_DIR = Path(__file__).resolve().parent

# Busca el dataset donde realmente están las carpetas de especies
CANDIDATE_ROOTS = [
    BASE_DIR / "species_dataset" / "data" / "data",
    BASE_DIR / "species_dataset" / "data",
    BASE_DIR / "species_dataset",
]

OUTPUT_ROOT = BASE_DIR / "reference_species"
REPORT_CSV = BASE_DIR / "species_grouping_report.csv"

# Cambia a True si quieres copiar físicamente las carpetas a reference_species/
COPY_FOLDERS = True

# Clasificación conservadora: solo lo bastante claro va a edible o poisonous.
EDIBLE_EXACT = {
    "almond_mushroom",
    "amethyst_chanterelle",
    "aniseed_funnel",
    "bay_bolete",
    "bearded_milkcap",
    "beefsteak_fungus",
    "black_morel",
    "butter_cap",
    "cauliflower_fungus",
    "chanterelle",
    "chicken_of_the_woods",
    "common_morel",
    "common_puffball",
    "fairy_ring_champignons",
    "field_blewit",
    "field_mushroom",
    "giant_puffball",
    "hedgehog_fungus",
    "hen_of_the_woods",
    "honey_fungus",
    "horn_of_plenty",
    "horse_mushroom",
    "king_alfreds_cakes",
    "lions_mane",
    "morel",
    "oyster_mushroom",
    "pale_oyster",
    "penny_bun",
    "pine_bolete",
    "saffron_milkcap",
    "st_georges_mushroom",
    "truffles",
    "velvet_shank",
    "winter_chanterelle",
    "wood_blewit",
}

POISONOUS_EXACT = {
    "amanita_gemmata",
    "deadly_fibrecap",
    "deadly_webcap",
    "deathcap",
    "destroying_angel",
    "devils_bolete",
    "false_chanterelle",
    "false_deathcap",
    "false_morel",
    "false_saffron_milkcap",
    "fly_agaric",
    "fools_funnel",
    "funeral_bell",
    "grey_spotted_amanita",
    "liberty_cap",
    "panthercap",
    "poison_pie",
    "the_sickener",
    "warted_amanita",
    "weeping_widow",
    "white_false_death_cap",
    "yellow_stainer",
}

# Reglas rápidas por nombre
POISONOUS_KEYWORDS = [
    "deadly",
    "deathcap",
    "destroying",
    "false_",
    "funeral_bell",
    "fools_funnel",
    "poison_pie",
    "yellow_stainer",
    "liberty_cap",
    "weeping_widow",
    "panthercap",
    "sickener",
    "fly_agaric",
    "warted_amanita",
    "poison",
    "devils_bolete",
]

EDIBLE_KEYWORDS = [
    "chanterelle",
    "bolete",
    "morel",
    "oyster",
    "puffball",
    "truffle",
    "honey_fungus",
    "hen_of_the_woods",
    "lions_mane",
    "horn_of_plenty",
    "horse_mushroom",
    "field_mushroom",
    "penny_bun",
    "st_georges_mushroom",
    "wood_blewit",
    "winter_chanterelle",
    "saffron_milkcap",
    "fairy_ring_champignons",
    "king_alfreds_cakes",
    "chicken_of_the_woods",
    "velvet_shank",
]


def find_species_root():
    for candidate in CANDIDATE_ROOTS:
        if candidate.exists() and candidate.is_dir():
            subdirs = [p for p in candidate.iterdir() if p.is_dir()]
            if subdirs:
                return candidate
    raise FileNotFoundError("No encontré la carpeta raíz del dataset de especies.")


def classify_species(name: str):
    n = name.lower()

    if n in POISONOUS_EXACT:
        return "poisonous", "exact_match"
    if n in EDIBLE_EXACT:
        return "edible", "exact_match"

    if any(k in n for k in POISONOUS_KEYWORDS):
        return "poisonous", "keyword_rule"

    if any(k in n for k in EDIBLE_KEYWORDS):
        return "edible", "keyword_rule"

    return "uncertain", "no_clear_rule"


def first_image_in_folder(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            return str(p)
    return ""


def copy_species_folder(src: Path, group: str):
    dst = OUTPUT_ROOT / group / src.name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def main():
    species_root = find_species_root()
    print(f"Usando dataset en: {species_root}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "edible").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "poisonous").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "uncertain").mkdir(parents=True, exist_ok=True)

    rows = []
    counts = {"edible": 0, "poisonous": 0, "uncertain": 0}

    species_dirs = sorted([p for p in species_root.iterdir() if p.is_dir()])

    for species_dir in species_dirs:
        species_name = species_dir.name
        label, reason = classify_species(species_name)
        sample_image = first_image_in_folder(species_dir)

        target_path = ""
        if COPY_FOLDERS:
            dst = copy_species_folder(species_dir, label)
            target_path = str(dst)

        counts[label] += 1
        rows.append(
            {
                "species_name": species_name,
                "label": label,
                "reason": reason,
                "source_path": str(species_dir),
                "target_path": target_path,
                "sample_image": sample_image,
            }
        )

        print(f"{species_name:35s} -> {label}")

    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "species_name",
                "label",
                "reason",
                "source_path",
                "target_path",
                "sample_image",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nDone.")
    print("Edible   :", counts["edible"])
    print("Poisonous:", counts["poisonous"])
    print("Uncertain:", counts["uncertain"])
    print(f"Reporte guardado en: {REPORT_CSV}")
    if COPY_FOLDERS:
        print(f"Carpetas copiadas a: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()