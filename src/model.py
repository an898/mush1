import torch.nn as nn
from torchvision import models


def build_model(num_classes=2, fine_tune=True):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last residual block if fine-tuning
    if fine_tune:
        for param in model.layer4.parameters():
            param.requires_grad = True

    # Replace classifier
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes)
    )

    return model