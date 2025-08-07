
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image
import numpy as np


def create_image_paths(data_dir):
    image_paths = []
    labels = []
    class_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    for label, class_dir in enumerate(class_dirs):
        images = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if img.endswith('.tif') or img.endswith('.jpg')]
        image_paths.extend(images)
        labels.extend([label] * len(images))

    if len(image_paths) == 0:
        raise ValueError("no")
    return image_paths, labels


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, num_classes)

    def forward(self, x):
        return self.cnn(x)


class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)


def evaluate_model(model, data_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for img, label in data_loader:
            outputs = model(img)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)


def train_cnn_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=30):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for img, label in train_loader:
            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 在验证集上评估
        val_labels, val_preds = evaluate_model(model, val_loader)
        accuracy = accuracy_score(val_labels, val_preds)
        precision = precision_score(val_labels, val_preds, average='macro')
        recall = recall_score(val_labels, val_preds, average='macro')
        f1 = f1_score(val_labels, val_preds, average='macro')

        print(f"Epoch [{epoch + 1}/{num_epochs}] | "
              f"Loss: {total_loss / len(train_loader):.4f} | "
              f"Val Acc: {accuracy * 100:.2f}%, Precision: {precision:.4f}, "
              f"Recall: {recall:.4f}, F1: {f1:.4f}")


if __name__ == '__main__':
    data_dir = r"D:\UCM\UCM100\Triplet"
    image_paths, labels = create_image_paths(data_dir)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    combined = list(zip(image_paths, labels))
    random.shuffle(combined)
    image_paths[:], labels[:] = zip(*combined)

    train_size = int(0.8 * len(image_paths))
    train_image_paths, test_image_paths = image_paths[:train_size], image_paths[train_size:]
    train_labels, test_labels = labels[:train_size], labels[train_size:]

    train_dataset = ImageDataset(train_image_paths, train_labels, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    test_dataset = ImageDataset(test_image_paths, test_labels, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    num_classes = len(set(labels))
    cnn_model = SimpleCNN(num_classes)
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_cnn_model(cnn_model, train_loader, test_loader, optimizer, criterion)

    test_labels, predicted_labels = evaluate_model(cnn_model, test_loader)
    accuracy = accuracy_score(test_labels, predicted_labels)
    precision = precision_score(test_labels, predicted_labels, average='macro')
    recall = recall_score(test_labels, predicted_labels, average='macro')
    f1 = f1_score(test_labels, predicted_labels, average='macro')
    print(f"(Accuracy): {accuracy * 100:.4f}%")
    print(f" (Precision): {precision:.4f}")
    print(f"(Recall): {recall:.4f}")
    print(f"(F1 Score): {f1:.4f}")
