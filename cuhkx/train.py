"""
CUHK-X Small Model Track Training Pipeline.

Features:
- GroupKFold cross-subject validation
- Label smoothing cross-entropy
- Cosine annealing with warmup
- Stochastic modality dropout
- Knowledge distillation from teacher (optional)
- Automatic model size monitoring

Usage:
    python -m cuhkx.train --data_root /path/to/HAR/data \
        --epochs 100 --batch_size 8 --lr 1e-3
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from .dataset import CUHKXDataset, TRAIN_USERS
from .models.fusion import CUHKXModel


def parse_args():
    p = argparse.ArgumentParser(description="CUHK-X Small Model Track Training")
    p.add_argument("--data_root", required=True, help="Path to HAR/data or extracted training root")
    p.add_argument("--output_dir", default="./output", help="Output directory for checkpoints")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--n_folds", type=int, default=6)
    p.add_argument("--fold", type=int, default=-1, help="Train single fold, -1 for all")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--modality_dropout", type=float, default=0.2)
    p.add_argument("--single_modality_prob", type=float, default=0.05)
    p.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    p.add_argument("--teacher_checkpoint", type=str, default=None, help="Teacher model for KD")
    p.add_argument("--kd_alpha", type=float, default=0.7, help="KD loss weight (1-alpha = CE weight)")
    p.add_argument("--size_limit_mb", type=float, default=100.0)
    return p.parse_args()


def build_splits(data_root, n_folds=6):
    """Build GroupKFold cross-subject splits."""
    samples = []
    data_dir = Path(data_root)
    for mod_dir in data_dir.iterdir():
        if not mod_dir.is_dir():
            continue
        for action_dir in mod_dir.iterdir():
            if not action_dir.is_dir():
                continue
            action_name = action_dir.name
            try:
                action_id = int(action_name.split("_")[0])
            except (ValueError, IndexError):
                continue
            for user_dir in action_dir.iterdir():
                if not user_dir.is_dir():
                    continue
                try:
                    user_id = int(user_dir.name.replace("user", ""))
                except ValueError:
                    continue
                if user_id not in TRAIN_USERS:
                    continue
                for trial_dir in user_dir.iterdir():
                    if not trial_dir.is_dir():
                        continue
                    rel_path = str(trial_dir.relative_to(data_root))
                    samples.append({
                        "path": rel_path + "/",
                        "action_id": action_id,
                        "user_id": user_id,
                    })

    # Deduplicate (same trial appears under each modality)
    seen = set()
    unique = []
    for s in samples:
        key = s["path"]
        if key not in seen:
            seen.add(key)
            unique.append(s)

    paths = np.array([s["path"] for s in unique])
    labels = np.array([s["action_id"] for s in unique])
    groups = np.array([s["user_id"] for s in unique])

    gkf = GroupKFold(n_splits=n_folds)
    splits = []
    for train_idx, val_idx in gkf.split(paths, labels, groups):
        splits.append((train_idx, val_idx))

    return unique, splits


def train_epoch(model, dataloader, criterion, optimizer, device, kd_criterion=None, teacher=None, kd_alpha=0.7):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)

        labels = batch["label"]
        modality_mask = batch.get("mask", None)
        optimizer.zero_grad()

        outputs = model(
            skeleton=batch["skeleton"],
            depth=batch["depth"],
            thermal=batch["thermal"],
            ir=batch["ir"],
            imu=batch["imu"],
            radar=batch["radar"],
            modality_mask=modality_mask,
            training=True,
        )

        loss = criterion(outputs, labels)

        if teacher is not None and kd_criterion is not None:
            with torch.no_grad():
                teacher_out = teacher(
                    skeleton=batch["skeleton"],
                    depth=batch["depth"],
                    thermal=batch["thermal"],
                    ir=batch["ir"],
                    imu=batch["imu"],
                    radar=batch["radar"],
                    modality_mask=modality_mask,
                    training=False,
                )
            kd_loss = kd_criterion(
                nn.functional.log_softmax(outputs / 3.0, dim=1),
                nn.functional.softmax(teacher_out / 3.0, dim=1),
            )
            loss = (1 - kd_alpha) * loss + kd_alpha * kd_loss * 9.0

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / len(dataloader), correct / total if total > 0 else 0.0


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)

        labels = batch["label"].to(device)
        modality_mask = batch.get("mask", None)
        outputs = model(
            skeleton=batch["skeleton"],
            depth=batch["depth"],
            thermal=batch["thermal"],
            ir=batch["ir"],
            imu=batch["imu"],
            radar=batch["radar"],
            modality_mask=modality_mask,
            training=False,
        )

        loss = criterion(outputs, labels)
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / len(dataloader), correct / total if total > 0 else 0.0


def collate_fn(batch):
    """Custom collate function - all modalities always present (zero-filled if missing)."""
    keys = ["skeleton", "depth", "thermal", "ir", "imu", "radar"]
    result = {k: [] for k in keys}
    result["label"] = []
    result["path"] = []
    result["mask"] = []

    for item in batch:
        data = item["data"]
        mask = item.get("modality_mask", [1.0] * 6)
        for k in keys:
            result[k].append(data[k])
        result["label"].append(item["label"])
        result["path"].append(item["path"])
        result["mask"].append(mask)

    for k in keys:
        result[k] = torch.stack(result[k])

    result["label"] = torch.tensor(result["label"], dtype=torch.long)
    result["mask"] = torch.tensor(result["mask"], dtype=torch.float32)
    return result


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Build splits
    print("Building cross-subject splits...")
    samples, splits = build_splits(args.data_root, args.n_folds)
    print(f"Total training samples: {len(samples)}")

    # Train each fold
    folds_to_run = range(args.n_folds) if args.fold < 0 else [args.fold]

    for fold_idx in folds_to_run:
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/{args.n_folds}")
        print(f"{'='*60}")

        train_idx, val_idx = splits[fold_idx]
        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]

        train_users = set(s["user_id"] for s in train_samples)
        val_users = set(s["user_id"] for s in val_samples)
        print(f"Train users: {sorted(train_users)}")
        print(f"Val users: {sorted(val_users)}")
        print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")

        # Create datasets
        train_ds = CUHKXDataset(
            root=args.data_root,
            split_file=None,
            is_train=True,
            modality_dropout_prob=args.modality_dropout,
            single_modality_prob=args.single_modality_prob,
        )
        train_ds.df = train_samples
        train_ds.labels = [s["action_id"] for s in train_samples]

        val_ds = CUHKXDataset(
            root=args.data_root,
            split_file=None,
            is_train=False,
        )
        val_ds.df = val_samples
        val_ds.labels = [s["action_id"] for s in val_samples]

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
        )

        # Create model
        model = CUHKXModel(
            num_classes=40,
            embed_dim=args.embed_dim,
            dropout=args.dropout,
        ).to(device)

        total_params, _ = model.count_parameters()
        print(f"Model parameters: {total_params:,} ({total_params * 4 / (1024**2):.1f} MB FP32)")

        # Load teacher if KD enabled
        teacher = None
        kd_criterion = None
        if args.teacher_checkpoint:
            teacher = CUHKXModel(num_classes=40, embed_dim=args.embed_dim, dropout=0.0).to(device)
            teacher.load_state_dict(torch.load(args.teacher_checkpoint, map_location=device))
            teacher.eval()
            kd_criterion = nn.KLDivLoss(reduction="batchmean")
            print("Knowledge distillation enabled")

        # Resume from checkpoint
        start_epoch = 0
        if args.checkpoint:
            ckpt = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(ckpt["model"])
            start_epoch = ckpt.get("epoch", 0)
            print(f"Resumed from epoch {start_epoch}")

        # Optimizer & scheduler
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.epochs, T_mult=1, eta_min=1e-6
        )

        best_val_acc = 0.0

        for epoch in range(start_epoch, args.epochs):
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device,
                kd_criterion=kd_criterion, teacher=teacher, kd_alpha=args.kd_alpha,
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            scheduler.step()

            # Check model size
            tmp_path = os.path.join(args.output_dir, f"temp_fold{fold_idx}.pth")
            torch.save(model.state_dict(), tmp_path)
            model_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)

            print(f"Epoch {epoch+1:3d} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Size: {model_size_mb:.1f}MB | LR: {scheduler.get_last_lr()[0]:.2e}")

            if model_size_mb > args.size_limit_mb:
                print(f"WARNING: Model exceeds {args.size_limit_mb}MB limit!")

            # Save best checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_path = os.path.join(args.output_dir, f"best_fold{fold_idx}.pth")
                torch.save({
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "model_size_mb": model_size_mb,
                }, best_path)
                print(f"  -> Saved best checkpoint (val_acc={val_acc:.4f})")

            os.remove(tmp_path)

    print(f"\nTraining complete! Best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
