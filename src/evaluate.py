# src/evaluate.py

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from src.model_custom_cnn import build_model
from src.preprocessing import prepare_dataloaders

# Parameters
DATA_DIR = r"E:\Brain Tumor MRI Image Classification\Data"
NUM_CLASSES = 4
MODEL_PATH = "outputs/models/custom_cnn.pth"

# Load dataloaders
_, _, test_loader = prepare_dataloaders(DATA_DIR, batch_size=32)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Metrics
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=test_loader.dataset.classes))

print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred, average='macro'):.4f}")
print(f"Recall:    {recall_score(y_true, y_pred, average='macro'):.4f}")
print(f"F1 Score:  {f1_score(y_true, y_pred, average='macro'):.4f}")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_loader.dataset.classes, yticklabels=test_loader.dataset.classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
