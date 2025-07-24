import matplotlib.pyplot as plt
import torch
import os

def plot_training_curves(history_custom, history_resnet, save_path="outputs/comparison_plot.png"):
    epochs = range(1, len(history_custom['train_acc']) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_custom['train_acc'], label='Custom CNN - Train')
    plt.plot(epochs, history_custom['val_acc'], label='Custom CNN - Val')
    plt.plot(epochs, history_resnet['train_acc'], label='ResNet50 - Train')
    plt.plot(epochs, history_resnet['val_acc'], label='ResNet50 - Val')
    plt.title("Model Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_custom['train_loss'], label='Custom CNN - Train')
    plt.plot(epochs, history_custom['val_loss'], label='Custom CNN - Val')
    plt.plot(epochs, history_resnet['train_loss'], label='ResNet50 - Train')
    plt.plot(epochs, history_resnet['val_loss'], label='ResNet50 - Val')
    plt.title("Model Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# Example: Loading from history if saved as .pt
def load_history(path):
    return torch.load(path)

if __name__ == "__main__":
    custom_history = load_history("outputs/history/custom_history.pt")
    resnet_history = load_history("outputs/history/resnet_history.pt")
    plot_training_curves(custom_history, resnet_history)
