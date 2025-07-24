# src/preprocessing.py

import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_data_transforms(img_size=224):
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    valid_test_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    return train_transforms, valid_test_transforms

def prepare_dataloaders(data_dir, batch_size=32):
    train_dir = os.path.join(data_dir, 'train', 'train')
    valid_dir = os.path.join(data_dir, 'valid', 'valid')
    test_dir = os.path.join(data_dir, 'test', 'test')

    train_tf, test_tf = get_data_transforms()

    train_dataset = datasets.ImageFolder(train_dir, transform=train_tf)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=test_tf)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_tf)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    class_names = train_dataset.classes
    print("Classes:", class_names)

    return train_loader, valid_loader, test_loader, class_names

def get_class_labels(train_dir):
    return sorted([folder for folder in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, folder))])
