import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torchvision

# === Dataset ===
class PlantDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform

        for img_file in os.listdir(root_dir):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path = os.path.join(root_dir, img_file)
            label = 0 if 'healthy' in img_file.lower() else 1
            self.images.append(img_path)
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

# === Modèle avec ResNet18 gelé ===
class TransferNet(nn.Module):
    def __init__(self):
        super(TransferNet, self).__init__()
        base = models.resnet18(pretrained=True)
        for param in base.parameters():
            param.requires_grad = False
        self.features = nn.Sequential(*list(base.children())[:-1])  # [B, 512, 1, 1]

        self.classifier = nn.Sequential(
            nn.Flatten(),              # [B, 512]
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)          # binaire
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)
        x = self.classifier(x)
        return x

# === Affichage rapide ===
def imshow(img):
    img = img * 0.5 + 0.5
    npimg = img.numpy()
    plt.figure(figsize=(10, 5))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()

# === Entraînement ===
def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # normes ImageNet
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = PlantDataset(root_dir='plants/test/test', transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    print(f"Images entraînement : {len(train_dataset)}, validation : {len(val_dataset)}")
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images[:8]))
    print("Labels :", labels[:8].tolist())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransferNet().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)  # Seule la partie FC est entraînée

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_corrects = 0

        for images, labels in tqdm(train_loader, desc=f"Époque {epoch+1}/{num_epochs}"):
            images = images.to(device)
            labels = labels.to(device).float()

            outputs = model(images).squeeze()
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_corrects += (preds == labels).sum().item()
            train_loss += loss.item()

        acc = train_corrects / len(train_dataset)
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).float()
                outputs = model(images).squeeze()
                loss = loss_fn(outputs, labels)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_corrects += (preds == labels).sum().item()
                val_loss += loss.item()

        val_acc = val_corrects / len(val_dataset)
        val_loss /= len(val_loader)

        print(f"\n[Époque {epoch+1}] Train Loss: {train_loss:.4f} | Train Acc: {acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

# === Lancement ===
if __name__ == "__main__":
    main()
