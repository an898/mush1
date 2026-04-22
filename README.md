# 🍄 MushroomGuard

## Overview
MushroomGuard is a deep learning-based prototype designed to perform **mushroom toxicity risk screening from images**.

The system allows users to upload an image of a mushroom and receive:
- A prediction of whether it is **likely edible or poisonous**
- A **confidence score**
- A **visually similar mushroom type** with a reference image
- A simple **feedback mechanism** to improve the system

This project is designed as an **educational and decision-support tool**, not as a certified safety system.

---

## Project Structure

mushroomguard/

├── app.py                     # Streamlit application (inference + UI)

├── train.py                  # Model training script

├── build_species_bank.py     # Builds visual similarity reference bank

├── organize_species_dataset.py # Groups species into edible/poisonous

├── split_dataset.py          # Dataset splitting

├── test_dataloader.py        # Dataset loading test

├── test_model.py             # Model forward pass test

├── predict.py                # Single-image prediction script

├── requirements.txt          # Dependencies

├── README.md

├── best_model.pth            # Trained model weights

├── species_bank.pt           # Precomputed similarity embeddings

├── logo.png

├── feedback.csv              # User feedback (auto-generated)

├── data/                     # Training dataset (train/val/test)

├── src/                      # Core modules (model, utils, similarity)

├── memes/                    # Meme outputs for demo

├── reference_species/        # Reference images grouped by toxicity

└── species_dataset/          # Raw species dataset

---

## Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd mushroomguard

2. Create a virtual environment

python -m venv venv
source venv/bin/activate

3. Install dependencies

pip install -r requirements.txt


⸻

Environment Requirements
	•	Python 3.10+
	•	PyTorch
	•	Streamlit
	•	Standard ML libraries (see requirements.txt)

⸻

Reproducing Results

1. Train the model

python train.py

This trains a CNN (ResNet-based) classifier for:
	•	edible vs poisonous mushrooms

⸻

2. Build the species similarity bank

python build_species_bank.py

This creates:

species_bank.pt

Used to find the most visually similar mushroom.

⸻

3. Run the application

python -m streamlit run app.py


⸻

System Pipeline

User Upload
    ↓
Image Preprocessing
    ↓
CNN Model (ResNet)
    ↓
Prediction (edible / poisonous)
    ↓
Confidence Score
    ↓
Similarity Search (within predicted class)
    ↓
Reference Mushroom Image
    ↓
User Feedback


⸻

Features
	•	Binary classification (edible vs poisonous)
	•	Transfer learning with pretrained CNN
	•	Visual similarity retrieval using embeddings
	•	Clean interactive web interface (Streamlit)
	•	Feedback collection system

⸻

Technical Design

Model
	•	ResNet-based CNN
	•	Fine-tuned on mushroom dataset
	•	Cross-entropy loss
	•	Data augmentation

Similarity Module
	•	Feature extraction using CNN embeddings
	•	Cosine similarity
	•	Search restricted to predicted toxicity class

⸻

Limitations
	•	Model trained on limited dataset → may not generalize to all species
	•	Visual similarity does not guarantee biological accuracy
	•	Some edible mushrooms resemble poisonous ones
	•	Cannot replace expert identification

⸻

Ethical & Safety Considerations

This system:
	•	is NOT a food safety tool
	•	may produce incorrect predictions
	•	includes warnings and confidence scores

Users should never rely solely on this system for real-world mushroom consumption decisions.

⸻

Future Work
	•	Multi-species classification instead of binary
	•	Larger and more diverse datasets
	•	Mobile deployment
	•	Better uncertainty detection
	•	Integration with expert validation systems

⸻

Contributors

Group 5
	•	Any Wan Ying Liu Liu
	•	Rocío Cepero Ortiz
	•	Pablo Gómez Prieto
	•	Yara Marín Fernández
	•	Miguel Romero Clavero
	•	Ignacio Javier Seisdedos Rico
	•	Emma Vizoso Romero

Professor:
Rubén Sánchez García

⸻

Notes

This project was developed as part of a course on AI, Machine Learning, and Analytics.

