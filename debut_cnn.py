#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 16:02:10 2025

@author: mcheron001
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

# === 1. Dataset personnalisé ===
class MelanomaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform

        for label, subfolder in enumerate(['Benign', 'Malignant']):
            folder_path = os.path.join(root_dir, subfolder)
            for img_file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_file)
                self.images.append(img_path)
                self.labels.append(label)  # 0: Benign, 1: Malignant

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label

# === 2. Définir les transformations (resize, normalisation) ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# === 3. Créer le dataset ===
dataset = MelanomaDataset(root_dir='archive/test', transform=transform)

# === 4. Split en train/val (80/20) ===
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# === 5. DataLoaders ===
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

print(f"Nombre d'images entraînement : {len(train_dataset)}")
print(f"Nombre d'images validation : {len(val_dataset)}")


import matplotlib.pyplot as plt
import torchvision
import numpy as np

# Fonction pour afficher une grille d'images
def imshow(img, title=None):
    img = img * 0.5 + 0.5  # dénormalisation
    npimg = img.numpy()
    plt.figure(figsize=(10, 5))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

# Charger un batch d'images du DataLoader d'entraînement
dataiter = iter(train_loader)
images, labels = next(dataiter)

# Afficher une grille
imshow(torchvision.utils.make_grid(images[:8]))
print("Labels (0 = Benign, 1 = Malignant) :", labels[:8].tolist())



#%%CNN


import torch.nn as nn
import torch.nn.functional as F

# === Définition du modèle CNN ===
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolution 1: entrée 3 canaux (RGB), sortie 16 canaux, filtre 3x3
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling 2x2
        # Convolution 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # Fully connected layers (à ajuster si tu changes la taille des images)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)  # 224 / 2 / 2 = 56 après 2 poolings
        self.fc2 = nn.Linear(128, 1)  # Sortie unique pour binaire

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 + ReLU + Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 + ReLU + Pool
        x = x.view(-1, 32 * 56 * 56)          # Flatten
        x = F.relu(self.fc1(x))               # Fully connected
        x = self.fc2(x)                       # Sortie brute (pas de Sigmoid ici)
        return x


tqdm = lambda x, **kwargs: x  # fallback simple

# === 1. Créer une instance du modèle ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

# === 2. Définir la loss et l’optimiseur ===
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# === 3. Boucle d'entraînement ===
num_epochs = 10  # tu peux adapter ce nombre

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device).float()  # BCEWithLogitsLoss attend des floats

        optimizer.zero_grad()
        outputs = model(images).squeeze()

        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        # Accuracy
        preds = (torch.sigmoid(outputs) > 0.5).float()
        running_corrects += (preds == labels).sum().item()
        total_samples += labels.size(0)
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_acc = running_corrects / total_samples

    # === 4. Évaluation sur validation ===
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