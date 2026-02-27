"""Train the Speech Emotion Recognition model.

Usage:
    cd backend
    python train.py
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from app import (
    Config,
    AudioFeatureExtractor,
    EmotionRecognitionModel,
    EmotionDataset,
    prepare_dataset,
)


def train(model, train_loader, val_loader, config):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5,
    )
    best_val_acc = 0.0
    weights_path = os.path.join(config.WEIGHTS_DIR, "best_model.pth")

    for epoch in range(config.NUM_EPOCHS):
        # --- train ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for feats, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}"):
            feats, labels = feats.to(config.DEVICE), labels.to(config.DEVICE)

            optimizer.zero_grad()
            out = model(feats)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, pred = torch.max(out, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

        train_acc = 100.0 * correct / total

        # --- validate ---
        model.eval()
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(config.DEVICE), labels.to(config.DEVICE)
                out = model(feats)
                _, pred = torch.max(out, 1)
                v_total += labels.size(0)
                v_correct += (pred == labels).sum().item()

        val_acc = 100.0 * v_correct / v_total
        scheduler.step(val_acc)

        print(
            f"  Loss: {running_loss / len(train_loader):.4f}  "
            f"Train: {train_acc:.1f}%  Val: {val_acc:.1f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(config.WEIGHTS_DIR, exist_ok=True)
            torch.save(model.state_dict(), weights_path)
            print(f"  -> Saved best model ({val_acc:.1f}%)")

    print(f"\nDone. Best val accuracy: {best_val_acc:.1f}%")


def main():
    config = Config()
    prepare_dataset()

    extractor = AudioFeatureExtractor(config)

    paths, labels = [], []
    for emo, label in config.EMOTION_TO_LABEL.items():
        edir = os.path.join(config.DATASET_DIR, emo)
        if not os.path.isdir(edir):
            continue
        for f in os.listdir(edir):
            if f.endswith(".wav"):
                paths.append(os.path.join(edir, f))
                labels.append(label)

    tr_p, va_p, tr_l, va_l = train_test_split(paths, labels, test_size=0.2, random_state=42)

    train_ds = EmotionDataset(tr_p, tr_l, extractor, augment=True)
    val_ds = EmotionDataset(va_p, va_l, extractor, augment=False)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE)

    model = EmotionRecognitionModel(config).to(config.DEVICE)
    print(f"Device: {config.DEVICE}  |  Train: {len(tr_p)}  Val: {len(va_p)}")

    train(model, train_loader, val_loader, config)


if __name__ == "__main__":
    main()
