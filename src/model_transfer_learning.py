# src/model_transfer_learning.py

import torch
import torch.nn as nn
from torchvision import models

def build_resnet50(num_classes, pretrained=True, freeze_features=True):
    model = models.resnet50(pretrained=pretrained)

    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )

    return model

def build_model(model_name, num_classes):
    if model_name == 'resnet50':
        return build_resnet50(num_classes=num_classes)
    else:
        raise ValueError("Unsupported model name. Try 'resnet50'.")
