import torch
from src.dataset_utils import get_dataloaders
from src.model import build_model

# Load one batch
train_loader, val_loader, test_loader, class_names = get_dataloaders()
images, labels = next(iter(train_loader))

# Build model
model = build_model(num_classes=len(class_names))

# Forward pass
outputs = model(images)

print("Class names:", class_names)
print("Input batch shape:", images.shape)
print("Output shape:", outputs.shape)
print("First output row:", outputs[0])
print("Predicted class indices:", torch.argmax(outputs, dim=1)[:10].tolist())