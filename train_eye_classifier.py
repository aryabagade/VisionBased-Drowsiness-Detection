"""
DriveGuard AI - Model Training Pipeline
Trains DrowsyEyeNet on Open vs Closed Eye Datasets.
Generates metrics (Accuracy, Precision, Recall, F1-Score) and saves 'pretrainedModel.pth'.
"""

import os
import json
import random
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models.eye_classifier import DrowsyEyeNet, get_eye_transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DEMO_DIR = os.path.join(BASE_DIR, "driveguard_streamlit_demo")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DEMO_DIR, exist_ok=True)


def generate_synthetic_eye_patch(is_open=True, size=(64, 64)):
    """
    Generates a realistic eye patch with diverse skin tones, pupil/iris,
    eyelids, eyelashes, and lighting variations for robust training/benchmarking.
    """
    w, h = size
    # Random skin tone
    skin_r = random.randint(140, 240)
    skin_g = int(skin_r * random.uniform(0.65, 0.85))
    skin_b = int(skin_r * random.uniform(0.50, 0.75))
    img = np.ones((h, w, 3), dtype=np.uint8) * np.array([[[skin_b, skin_g, skin_r]]], dtype=np.uint8)

    # Eye center
    cx = w // 2 + random.randint(-3, 3)
    cy = h // 2 + random.randint(-2, 2)
    eye_w = random.randint(w // 3, w // 2)

    if is_open:
        eye_h = random.randint(h // 4, h // 3)
        # Sclera (white of eye)
        sclera_color = (random.randint(225, 250), random.randint(225, 250), random.randint(225, 250))
        cv2.ellipse(img, (cx, cy), (eye_w, eye_h), 0, 0, 360, sclera_color, -1)

        # Iris
        iris_r = random.randint(eye_h // 2, eye_h)
        iris_color = (random.randint(20, 80), random.randint(30, 90), random.randint(30, 110))
        cv2.circle(img, (cx, cy), iris_r, iris_color, -1)

        # Pupil
        pupil_r = max(2, iris_r // 2)
        cv2.circle(img, (cx, cy), pupil_r, (10, 10, 10), -1)

        # Specular reflection (glint)
        cv2.circle(img, (cx - 2, cy - 2), max(1, pupil_r // 3), (255, 255, 255), -1)

        # Upper eyelid line
        cv2.ellipse(img, (cx, cy), (eye_w + 2, eye_h + 2), 0, 180, 360, (skin_b//2, skin_g//2, skin_r//2), 2)
    else:
        # Closed eye: skin folds, closed crease curve, eyelashes
        crease_h = random.randint(1, 4)
        crease_color = (max(0, skin_b - 50), max(0, skin_g - 50), max(0, skin_r - 50))
        # Draw closed slit
        cv2.ellipse(img, (cx, cy), (eye_w, crease_h + 2), 0, 0, 180, crease_color, 2)
        # Eyelash fringes
        for ex in range(cx - eye_w + 4, cx + eye_w - 4, 3):
            ey = cy + random.randint(0, 2)
            cv2.line(img, (ex, ey), (ex + random.randint(-1, 1), ey + random.randint(3, 7)), (30, 25, 20), 1)

    # Add Gaussian blur / subtle sensor noise
    img = cv2.GaussianBlur(img, (3, 3), 0.5)
    noise = np.random.normal(0, random.uniform(2, 6), (h, w, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


class EyeDataset(Dataset):
    """Dataset loader supporting both real folders (dataset/train, val) and synthetic generation."""
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item, label = self.data_list[idx]
        if isinstance(item, str):
            image = Image.open(item).convert('RGB')
        else:
            # item is numpy array
            image = Image.fromarray(cv2.cvtColor(item, cv2.COLOR_BGR2RGB))
            
        if self.transform:
            image = self.transform(image)
        return image, label


def prepare_data(num_samples_per_class=1200):
    """
    Checks for dataset/ directory. If not present with real images,
    generates a comprehensive balanced dataset.
    """
    train_data = []
    val_data = []

    # Check for real user dataset
    real_train_dir = os.path.join(DATASET_DIR, "train")
    if os.path.exists(real_train_dir):
        print("[DriveGuard AI] Found existing user dataset in dataset/train.")
        for label_idx, cls_name in enumerate(["closed", "open"]):
            cls_dir = os.path.join(real_train_dir, cls_name)
            if os.path.exists(cls_dir):
                for fname in os.listdir(cls_dir):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        train_data.append((os.path.join(cls_dir, fname), label_idx))
                        
        real_val_dir = os.path.join(DATASET_DIR, "val")
        if os.path.exists(real_val_dir):
            for label_idx, cls_name in enumerate(["closed", "open"]):
                cls_dir = os.path.join(real_val_dir, cls_name)
                if os.path.exists(cls_dir):
                    for fname in os.listdir(cls_dir):
                        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                            val_data.append((os.path.join(cls_dir, fname), label_idx))

    # If no real dataset found or small, generate high-quality balanced dataset
    if len(train_data) < 100:
        print(f"[DriveGuard AI] Generating {num_samples_per_class * 2} balanced eye patches for training & validation...")
        all_samples = []
        # Class 0: Closed
        for _ in range(num_samples_per_class):
            img = generate_synthetic_eye_patch(is_open=False)
            all_samples.append((img, 0))
        # Class 1: Open
        for _ in range(num_samples_per_class):
            img = generate_synthetic_eye_patch(is_open=True)
            all_samples.append((img, 1))

        random.seed(42)
        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * 0.8)
        train_data = all_samples[:split_idx]
        val_data = all_samples[split_idx:]

    print(f"[DriveGuard AI] Dataset prepared: {len(train_data)} training samples, {len(val_data)} validation samples.")
    return train_data, val_data


def train_model(epochs=15, batch_size=32, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DriveGuard AI] Training using device: {device}")

    train_samples, val_samples = prepare_data()
    
    train_dataset = EyeDataset(train_samples, transform=get_eye_transforms(image_size=64, is_training=True))
    val_dataset = EyeDataset(val_samples, transform=get_eye_transforms(image_size=64, is_training=False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = DrowsyEyeNet(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0

    print("=" * 60)
    print("STARTING DRIVEGUARD AI MODEL TRAINING")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_loss = running_loss / total
        train_acc = (correct / total) * 100.0

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        val_loss = val_loss / val_total
        val_acc = (val_correct / val_total) * 100.0

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save model checkpoint to both models/ and driveguard_streamlit_demo/
            p1 = os.path.join(MODELS_DIR, "pretrainedModel.pth")
            p2 = os.path.join(DEMO_DIR, "pretrainedModel.pth")
            torch.save(model.state_dict(), p1)
            torch.save(model.state_dict(), p2)

    print("=" * 60)
    print(f"Training Complete! Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {os.path.join(MODELS_DIR, 'pretrainedModel.pth')}")
    print(f"Model saved to: {os.path.join(DEMO_DIR, 'pretrainedModel.pth')}")

    # Compute Confusion Matrix and Classification Metrics
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    
    tp = int(np.sum((all_targets == 0) & (all_preds == 0))) # Closed classified as Closed
    fn = int(np.sum((all_targets == 0) & (all_preds == 1))) # Closed classified as Open
    fp = int(np.sum((all_targets == 1) & (all_preds == 0))) # Open classified as Closed
    tn = int(np.sum((all_targets == 1) & (all_preds == 1))) # Open classified as Open

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * (precision * recall) / max(1e-6, precision + recall)
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)

    metrics = {
        "Accuracy": round(accuracy * 100, 2),
        "Precision_Closed": round(precision * 100, 2),
        "Recall_Closed": round(recall * 100, 2),
        "F1_Score_Closed": round(f1 * 100, 2),
        "True_Positives (Closed)": tp,
        "False_Negatives": fn,
        "False_Positives": fp,
        "True_Negatives (Open)": tn,
        "Total_Validation_Samples": len(all_targets)
    }

    with open(os.path.join(REPORTS_DIR, "metrics_summary.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Plot Training Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=200)
    ax1.plot(range(1, epochs + 1), history['train_loss'], label='Train Loss', color='#2B6CB0', lw=2)
    ax1.plot(range(1, epochs + 1), history['val_loss'], label='Val Loss', color='#C53030', lw=2, linestyle='--')
    ax1.set_title('Cross-Entropy Loss vs Epoch', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.plot(range(1, epochs + 1), history['train_acc'], label='Train Acc', color='#2B6CB0', lw=2)
    ax2.plot(range(1, epochs + 1), history['val_acc'], label='Val Acc', color='#2F855A', lw=2, linestyle='--')
    ax2.set_title('Classification Accuracy (%) vs Epoch', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "training_curves.png"))
    plt.close()

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=200)
    cm = np.array([[tp, fn], [fp, tn]])
    cax = ax.matshow(cm, cmap='Blues')
    fig.colorbar(cax)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=12, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Closed (0)', 'Open (1)'], fontsize=10)
    ax.set_yticklabels(['Closed (0)', 'Open (1)'], fontsize=10)
    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_ylabel('Ground Truth Label', fontweight='bold')
    ax.set_title('Validation Confusion Matrix', fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"))
    plt.close()

    print("[DriveGuard AI] Training curves and confusion matrix saved to 'reports/' directory.")
    return metrics


if __name__ == "__main__":
    train_model(epochs=12, batch_size=32)
