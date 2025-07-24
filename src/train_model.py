# src/train_model.py

import sys
import os

# 🔧 Allow relative imports from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path

from src.preprocessing import prepare_dataloaders
from src.model_custom_cnn import build_model as build_custom_cnn
from src.model_transfer_learning import build_model as build_transfer_model

# ------------------ TRAINING LOOP ------------------ #
def train_model(model, dataloaders, criterion, optimizer, num_epochs=10, device='cpu', save_path='outputs/models/best_model.pth'):
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase.capitalize()}"):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), save_path)
                print(f"✅ Best model saved to {save_path}!")

    print(f"\n🎯 Best Validation Accuracy: {best_acc:.4f}")
    return model

# ------------------ MAIN ------------------ #
if __name__ == "__main__":
    # ⚙️ CONFIG
    DATA_DIR = r"E:\Brain Tumor MRI Image Classification\Data"
    model_type = input("Enter model type [resnet50 / custom]: ").strip().lower()
    num_epochs = 10
    batch_size = 32
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔄 Load data
    train_loader, valid_loader, test_loader, class_names = prepare_dataloaders(DATA_DIR, batch_size=batch_size)
    dataloaders = {'train': train_loader, 'valid': valid_loader}
    num_classes = len(class_names)

    # 🧠 Build model
    if model_type == "custom":
        model = build_custom_cnn(num_classes=num_classes)
        save_path = "outputs/models/custom_model.pth"
    else:
        model = build_transfer_model("resnet50", num_classes=num_classes)
        save_path = "outputs/models/best_model.pth"

    # ⚙️ Optimizer & Loss
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 📁 Ensure output dir
    Path("outputs/models").mkdir(parents=True, exist_ok=True)

    # 🚀 Train
    train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        save_path=save_path
    )
