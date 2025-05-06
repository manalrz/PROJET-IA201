# cnn_plants_binary.py

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

# === Dataset binaire avec sous-dossiers ===
class PlantBinaryDatasetFromFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform

        for subfolder in os.listdir(root_dir):
            label = 0 if "healthy" in subfolder.lower() else 1
            folder_path = os.path.join(root_dir, subfolder)
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(folder_path, filename))
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# === Modèle CNN simple ===
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# === Fonction d'affichage ===
def imshow(img, title=None):
    img = img * 0.5 + 0.5
    npimg = img.numpy()
    plt.figure(figsize=(10, 5))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

# === Entraînement principal ===
def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = PlantBinaryDatasetFromFolder(
        'new_plants/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train', transform=transform)
    val_dataset = PlantBinaryDatasetFromFolder(
        'new_plants/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/test', transform=transform)

    print(f"Nombre d'images entraînement : {len(train_dataset)}")
    print(f"Nombre d'images validation : {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images[:8]))
    print("Labels (0 = Healthy, 1 = Diseased) :", labels[:8].tolist())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for images, labels in tqdm(train_loader, desc=f"Époque {epoch+1}/{num_epochs}"):
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

        train_loss = running_loss / len(train_loader)
        train_acc = running_corrects / total_samples

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

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"[Époque {epoch+1}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # === Courbes
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss au fil des époques')
    plt.xlabel('Époque')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(val_accuracies, label='Val Acc')
    plt.title('Accuracy au fil des époques')
    plt.xlabel('Époque')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # === Bilan final
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).float()
            outputs = model(images).squeeze()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print("\n=== Bilan Final ===")
    print(f"Précision : {precision:.4f}")
    print(f"Rappel    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

if __name__ == "__main__":
    main()
