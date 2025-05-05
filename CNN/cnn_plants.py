#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN pour classification de plantes à partir de fichiers JPG.
Label déduit automatiquement des noms de fichiers (Healthy vs Diseased)
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# === Dataset pour fichiers étiquetés par nom ===
class PlantDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform

        for img_file in os.listdir(root_dir):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):  # sécurité
                continue
            img_path = os.path.join(root_dir, img_file)
            label = self.extract_label(img_file)
            self.images.append(img_path)
            self.labels.append(label)

    def extract_label(self, filename):
        """Déduit le label : 0 = Healthy, 1 = Diseased"""
        lower = filename.lower()
        return 0 if 'healthy' in lower else 1

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# === CNN simple ===
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# === Affichage ===
def imshow(img, title=None):
    img = img * 0.5 + 0.5  # Dénormalisation
    npimg = img.numpy()
    plt.figure(figsize=(10, 5))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

# === Code principal ===
def main():
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Dataset
    dataset = PlantDataset(root_dir='plants/test/test', transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Nombre d'images entraînement : {len(train_dataset)}")
    print(f"Nombre d'images validation : {len(val_dataset)}")

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Afficher un batch
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images[:8]))
    print("Labels (0 = Healthy, 1 = Diseased) :", labels[:8].tolist())

    # Entraînement
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 1  # Augmente ensuite à 10+
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Époque {epoch+1}/{num_epochs}")):
            images = images.to(device)
            labels = labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            running_corrects += (preds == labels).sum().item()
            total_samples += labels.size(0)
            running_loss += loss.item()

            if i % 10 == 0:
                print(f"  Batch {i}: loss={loss.item():.4f}")

        train_loss = running_loss / len(train_loader)
        train_acc = running_corrects / total_samples

        # Validation
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).float()
                outputs = model(images).squeeze()
                loss = loss_fn(outputs, labels)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_corrects += (preds == labels).sum().item()
                val_total += labels.size(0)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_acc = val_corrects / val_total

        print(f"\n[Époque {epoch+1}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

# === Lancement sécurisé ===
if __name__ == "__main__":
    main()
