#!/usr/bin/env python3
"""
ResNet-50 disease detection training — transfer learning.

Architecture per TR-13, training config per TR-14.
Class names loaded from config/column_map.yaml — never hardcoded.
Checkpoint per TR-15: includes class_names.

CLI:
  python train.py
  python train.py --data-dir data/ --epochs 10 --config config/column_map.yaml
"""

import argparse
import logging
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights
from sklearn.metrics import f1_score, classification_report
import numpy as np

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("disease-train")

CONFIG_PATH = Path(__file__).parent / "config" / "column_map.yaml"

# TR-14 transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# TR-14
BATCH_SIZE = 32
LR = 1e-4
PATIENCE = 3


def load_column_map(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_class_names(col_map: dict) -> list[str]:
    """Return class names in stable order from column_map.yaml."""
    return list(col_map["class_mapping"].keys())


def build_model(num_classes: int) -> nn.Module:
    """ResNet-50 per TR-13: freeze all, unfreeze layer4, replace fc."""
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model


def load_datasets(data_dir: Path, class_names: list[str]) -> tuple:
    """Load ImageFolder datasets; reorder to match class_names from config."""
    train_ds = datasets.ImageFolder(str(data_dir / "train"), transform=train_transform)
    val_ds   = datasets.ImageFolder(str(data_dir / "val"),   transform=val_transform)
    test_ds  = datasets.ImageFolder(str(data_dir / "test"),  transform=val_transform)

    # Validate class alignment
    folder_classes = sorted(train_ds.class_to_idx.keys())
    for cls in class_names:
        if cls not in train_ds.class_to_idx:
            raise ValueError(f"Class '{cls}' from column_map.yaml not found in {data_dir}/train/")

    log.info("Dataset classes (folder): %s", folder_classes)
    log.info("Class names (config):     %s", class_names)
    log.info("Train: %d  Val: %d  Test: %d", len(train_ds), len(val_ds), len(test_ds))
    return train_ds, val_ds, test_ds


def evaluate_f1(model, loader, device) -> tuple[float, list, list]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return f1, all_preds, all_labels


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(imgs)
    return total_loss / len(loader.dataset)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.getenv("DATA_DIR", "data"))
    parser.add_argument("--model-dir", default=os.getenv("MODEL_DIR", "model_weights"))
    parser.add_argument("--epochs", type=int, default=int(os.getenv("EPOCHS", "10")))
    parser.add_argument("--config", default=str(CONFIG_PATH))
    args = parser.parse_args()

    col_map     = load_column_map(Path(args.config))
    class_names = get_class_names(col_map)          # from config — not hardcoded
    num_classes = len(class_names)
    log.info("Class names from config: %s", class_names)

    data_dir = Path(args.data_dir)
    train_ds, val_ds, test_ds = load_datasets(data_dir, class_names)

    # Build class_to_idx mapping aligned with config order
    folder_idx = train_ds.class_to_idx           # {"early_blight": 0, ...} folder-alphabetical
    # Remap targets to config order
    config_idx = {cls: i for i, cls in enumerate(class_names)}

    def remap(ds, config_idx, folder_idx):
        """Remap dataset labels from folder order → config order.

        DatasetFolder.__getitem__ reads labels from ds.samples — remapping
        only ds.targets would be a silent no-op, so rewrite samples too.
        """
        folder_to_config = {folder_idx[cls]: config_idx[cls] for cls in class_names}
        ds.samples = [(p, folder_to_config[t]) for p, t in ds.samples]
        ds.imgs = ds.samples
        ds.targets = [t for _, t in ds.samples]
        ds.class_to_idx = config_idx
        return ds

    train_ds = remap(train_ds, config_idx, folder_idx)
    val_ds   = remap(val_ds,   config_idx, folder_idx)
    test_ds  = remap(test_ds,  config_idx, folder_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    model     = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )

    best_val_f1   = 0.0
    best_epoch    = 0
    patience_cnt  = 0
    best_state    = None

    log.info("=== Training ResNet-50 — %d epochs ===", args.epochs)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_f1, _, _ = evaluate_f1(model, val_loader, device)
        elapsed = time.time() - t0

        marker = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            patience_cnt = 0
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}
            marker = " ← best"
        else:
            patience_cnt += 1

        log.info("Epoch %2d/%d | loss=%.4f | val_f1=%.4f | %.0fs%s",
                 epoch, args.epochs, train_loss, val_f1, elapsed, marker)

        if patience_cnt >= PATIENCE:
            log.info("Early stopping (patience=%d)", PATIENCE)
            break

    # Load best weights
    if best_state:
        model.load_state_dict(best_state)

    # Evaluate on test set
    test_f1, test_preds, test_labels = evaluate_f1(model, test_loader, device)
    report = classification_report(
        test_labels, test_preds,
        target_names=class_names, zero_division=0
    )
    log.info("Test classification report:\n%s", report)

    log.info("=== Training Complete ===")
    log.info("Best epoch:       %d/%d", best_epoch, args.epochs)
    log.info("Val F1:           %.4f", best_val_f1)
    log.info("Test weighted F1: %.4f  (target ≥ 0.85)", test_f1)

    if test_f1 < 0.85:
        log.warning(
            "NOTE: F1 = %.4f, below 0.85 target.\n"
            "Likely cause: synthetic dataset (no real PlantVillage images).\n"
            "Recommendation: download real dataset via scripts/download_dataset.py",
            test_f1
        )

    # Save checkpoint per TR-15
    out = Path(args.model_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_names":      class_names,   # always saved in checkpoint
        "val_f1":           best_val_f1,
        "test_f1":          test_f1,
        "epoch":            best_epoch,
    }
    torch.save(checkpoint, out / "disease_model.pth")
    log.info("Checkpoint saved → %s", out / "disease_model.pth")


if __name__ == "__main__":
    main()
