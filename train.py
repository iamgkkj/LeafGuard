#!/usr/bin/env python3
"""
Training script for LeafGuard AI
- Supports cbam and baseline models
- CrossEntropy + AdamW + cosine LR + early stopping on val Macro-F1
- Saves checkpoints to checkpoints/{model_type}.pth and logs to SQLite/JSON
"""
import argparse
import json
import os
import random
import sqlite3
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# Local imports
from models.baseline import get_baseline_model
from models.mobilenet_cbam import get_cbam_model

# 38 classes
try:
    import json as _json

    with open(Path(__file__).parent / "data" / "classes.json") as f:
        CLASSES = _json.load(f)["classes"]
except Exception:
    from data.preprocess import DEFAULT_CLASSES

    CLASSES = DEFAULT_CLASSES

NUM_CLASSES = len(CLASSES)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_transforms(model_type: str, train: bool = True):
    if model_type == "cbam" and train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]
        )
    else:
        # baseline or val: minimal augmentation
        if train:
            return transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=MEAN, std=STD),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=MEAN, std=STD),
                ]
            )


def find_dataset_root(data_dir: Path) -> Path:
    candidates = [
        Path(data_dir),
        Path(data_dir) / "New Plant Diseases Dataset(Augmented)" / "New Plant Diseases Dataset(Augmented)" / "train",
        Path(data_dir) / "New Plant Diseases Dataset(Augmented)" / "train",
        Path(data_dir) / "train",
        Path(data_dir) / "valid",
    ]
    for c in candidates:
        if c.exists() and any(p.is_dir() for p in c.iterdir()):
            # check if contains class folders (look for at least 5)
            dirs = [d for d in c.iterdir() if d.is_dir()]
            if len(dirs) >= 5:
                return c
    return Path(data_dir)


def build_dataloaders(data_dir: Path, batch_size: int, model_type: str, num_workers: int = 2, val_split: float = 0.2):
    root = find_dataset_root(data_dir)
    print(f"[train] Using dataset root: {root}")
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}. Run data/download.py first.")

    # Check if train/valid split already exists (Kaggle dataset has train/valid)
    train_root = root
    val_root = None
    # If parent contains both train and valid, use them
    parent = Path(data_dir)
    maybe_train = parent / "New Plant Diseases Dataset(Augmented)" / "train"
    maybe_valid = parent / "New Plant Diseases Dataset(Augmented)" / "valid"
    if maybe_train.exists() and maybe_valid.exists():
        print(f"[train] Found Kaggle train/valid split")
        train_ds = datasets.ImageFolder(str(maybe_train), transform=get_transforms(model_type, train=True))
        val_ds = datasets.ImageFolder(str(maybe_valid), transform=get_transforms(model_type, train=False))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, val_loader, train_ds.classes

    # Otherwise, ImageFolder on root and random split
    full_ds = datasets.ImageFolder(str(root), transform=get_transforms(model_type, train=True))
    # For val, need separate transform; create two datasets with different transforms
    val_ds_raw = datasets.ImageFolder(str(root), transform=get_transforms(model_type, train=False))
    n = len(full_ds)
    n_val = int(n * val_split)
    n_train = n - n_val
    print(f"[train] Random split: {n_train} train / {n_val} val (total {n})")
    # Use indices split
    indices = list(range(n))
    random.seed(42)
    random.shuffle(indices)
    train_idx = indices[n_val:]
    val_idx = indices[:n_val]

    # Create subsets
    from torch.utils.data import Subset

    train_subset = Subset(full_ds, train_idx)
    val_subset = Subset(val_ds_raw, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, full_ds.classes


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    for imgs, labels in tqdm(loader, desc="train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    epoch_loss = running_loss / len(loader.dataset) if hasattr(loader.dataset, "__len__") else running_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds) if all_labels else 0
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) if all_labels else 0
    return epoch_loss, acc, f1


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    # handle Subset length
    try:
        n = len(loader.dataset)
    except Exception:
        n = len(all_labels)
    epoch_loss = running_loss / max(1, n)
    acc = accuracy_score(all_labels, all_preds) if all_labels else 0
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) if all_labels else 0
    return epoch_loss, acc, f1


def log_to_db(db_path: Path, run_data: dict):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_type TEXT,
            epochs INTEGER,
            batch_size INTEGER,
            lr REAL,
            accuracy REAL,
            macro_f1 REAL,
            params INTEGER,
            flops INTEGER,
            size_mb REAL,
            fps REAL,
            duration_s REAL,
            timestamp TEXT
        )
        """
    )
    cur.execute(
        "INSERT INTO runs (model_type, epochs, batch_size, lr, accuracy, macro_f1, params, flops, size_mb, fps, duration_s, timestamp) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            run_data.get("model_type"),
            run_data.get("epochs"),
            run_data.get("batch_size"),
            run_data.get("lr"),
            run_data.get("accuracy"),
            run_data.get("macro_f1"),
            run_data.get("params"),
            run_data.get("flops"),
            run_data.get("size_mb"),
            run_data.get("fps"),
            run_data.get("duration_s"),
            run_data.get("timestamp"),
        ),
    )
    conn.commit()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Train LeafGuard AI")
    parser.add_argument("--data-dir", type=str, default="data", help="dataset root")
    parser.add_argument("--model-type", type=str, choices=["cbam", "baseline"], default="cbam")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5, help="early stopping patience on val F1")
    parser.add_argument("--output", type=str, default=None, help="checkpoint path")
    parser.add_argument("--dry-run", action="store_true", help="run 1 epoch on synthetic data (no dataset needed)")
    parser.add_argument("--no-pretrained", action="store_true", help="disable ImageNet pretraining")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] Device: {device}, model={args.model_type}, epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}")

    # Model
    pretrained = not args.no_pretrained
    if args.model_type == "cbam":
        model = get_cbam_model(num_classes=NUM_CLASSES, pretrained=pretrained)
    else:
        model = get_baseline_model(num_classes=NUM_CLASSES, pretrained=pretrained)
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"[train] Params: {params:,}")

    # Dry run: synthetic data, 2 epoch mini-loop to verify pipeline
    if args.dry_run:
        print("[train] Dry run mode — synthetic data")
        from torch.utils.data import TensorDataset

        synth_train = TensorDataset(torch.randn(64, 3, 224, 224), torch.randint(0, NUM_CLASSES, (64,)))
        synth_val = TensorDataset(torch.randn(32, 3, 224, 224), torch.randint(0, NUM_CLASSES, (32,)))
        train_loader = DataLoader(synth_train, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(synth_val, batch_size=args.batch_size)
        # Quick train 2 epochs
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
        best_f1 = 0
        for epoch in range(min(2, args.epochs)):
            tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
            scheduler.step()
            print(f" dry-epoch {epoch+1}: train loss {tr_loss:.3f} acc {tr_acc:.3f} f1 {tr_f1:.3f} | val loss {val_loss:.3f} acc {val_acc:.3f} f1 {val_f1:.3f}")
            best_f1 = max(best_f1, val_f1)
        print(f"[train] Dry run done, best val F1 {best_f1:.3f}")
        # save dummy checkpoint
        ckpt_path = Path(args.output) if args.output else Path("checkpoints") / f"{args.model_type}_dry.pth"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict(), "model_type": args.model_type, "classes": CLASSES, "epochs": 2}, str(ckpt_path))
        print(f"[train] Saved dry checkpoint to {ckpt_path} ({ckpt_path.stat().st_size/1024/1024:.2f} MB)")
        return

    # Real data
    data_dir = Path(args.data_dir)
    try:
        train_loader, val_loader, class_names = build_dataloaders(data_dir, args.batch_size, args.model_type, args.num_workers)
        print(f"[train] Classes: {len(class_names)} -> {class_names[:3]} ...")
    except Exception as e:
        print(f"[train] Failed to build dataloaders: {e}")
        print("[train] Hint: run python data/download.py or pass --dry-run")
        return

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = 0
    best_state = None
    patience_counter = 0
    start = time.time()

    for epoch in range(args.epochs):
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        print(f" epoch {epoch+1}/{args.epochs} lr={lr_now:.2e} train loss {tr_loss:.4f} acc {tr_acc:.4f} f1 {tr_f1:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f} f1 {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[train] Early stopping at epoch {epoch+1} (best F1 {best_f1:.4f})")
                break

    duration = time.time() - start
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save checkpoint
    ckpt_path = Path(args.output) if args.output else Path("checkpoints") / f"{args.model_type}.pth"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_type": args.model_type,
            "classes": CLASSES,
            "best_f1": best_f1,
            "epochs": args.epochs,
            "lr": args.lr,
        },
        str(ckpt_path),
    )
    size_mb = ckpt_path.stat().st_size / 1024 / 1024
    print(f"[train] Saved checkpoint to {ckpt_path} ({size_mb:.2f} MB, params {params:,}, best F1 {best_f1:.4f})")
    if size_mb > 10:
        print(f"[train] WARNING: model >10MB ({size_mb:.1f}MB), smartphone claim may not hold")

    # Log to SQLite / JSON
    run_data = {
        "model_type": args.model_type,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "accuracy": float(val_acc),
        "macro_f1": float(best_f1),
        "params": params,
        "flops": 0,  # filled by evaluate.py
        "size_mb": float(size_mb),
        "fps": 0,
        "duration_s": float(duration),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    # JSON log
    json_path = Path("reports") / f"train_{args.model_type}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(run_data, f, indent=2)
    print(f"[train] JSON log -> {json_path}")

    # SQLite
    db_path = Path("leafguard.db")
    try:
        log_to_db(db_path, run_data)
        print(f"[train] Logged to {db_path}")
    except Exception as e:
        print(f"[train] DB log failed: {e}")


if __name__ == "__main__":
    main()
