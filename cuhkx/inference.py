"""
CUHK-X Small Model Track Inference Pipeline.

Generates submission.csv for the 405 test clips.

Usage:
    python -m cuhkx.inference --checkpoint model.pth \
        --test_root /path/to/small_model_track_test \
        --test_csv /path/to/test.csv \
        --output submission.csv
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import CUHKXDataset, MODALITIES
from .models.fusion import CUHKXModel


def parse_args():
    p = argparse.ArgumentParser(description="CUHK-X Inference")
    p.add_argument("--checkpoint", required=True, help="Trained model checkpoint")
    p.add_argument("--test_root", required=True, help="Path to small_model_track_test directory")
    p.add_argument("--test_csv", required=True, help="Path to test.csv")
    p.add_argument("--output", default="submission.csv")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--quantize", action="store_true", help="Apply INT8 dynamic quantization")
    return p.parse_args()


def collate_inference(batch):
    """Collate function for inference - all modalities always present (zero-filled if missing)."""
    keys = ["skeleton", "depth", "thermal", "ir", "imu", "radar"]
    result = {k: [] for k in keys}
    result["path"] = []
    result["mask"] = []

    for item in batch:
        data = item["data"]
        for k in keys:
            result[k].append(data[k])
        result["path"].append(item["path"])
        result["mask"].append(item.get("modality_mask", [1.0] * 6))

    for k in keys:
        result[k] = torch.stack(result[k])
    result["mask"] = torch.tensor(result["mask"], dtype=torch.float32)

    return result


def sliding_window_inference(model, sample_data, device, window_size=60, stride=30):
    """Apply sliding window for clips longer than window_size.

    Args:
        sample_data: Dict of modality tensors, each shape (B, C, T, ...) or (B, C, T)
    Returns:
        Averaged logits: (B, 40)
    """
    model.eval()
    all_logits = []

    # Get temporal dimension
    temporal_dims = {}
    for k, v in sample_data.items():
        if v is not None:
            if v.dim() == 5:  # (B, C, T, H, W)
                temporal_dims[k] = v.size(2)
            elif v.dim() == 3:  # (B, C, T)
                temporal_dims[k] = v.size(2)

    # Use fixed window (no sliding needed for T=60)
    with torch.no_grad():
        outputs = model(
            skeleton=sample_data.get("skeleton"),
            depth=sample_data.get("depth"),
            thermal=sample_data.get("thermal"),
            ir=sample_data.get("ir"),
            imu=sample_data.get("imu"),
            radar=sample_data.get("radar"),
            training=False,
        )
    return outputs


def quantize_model(model):
    """Apply dynamic INT8 quantization to linear and recurrent layers."""
    try:
        quantized = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.GRU},
            dtype=torch.qint8,
        )
        return quantized
    except Exception as e:
        print(f"Quantization failed: {e}, using FP32 model")
        return model


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = CUHKXModel(num_classes=40, embed_dim=args.embed_dim).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    # Quantize if requested
    if args.quantize:
        model = quantize_model(model)
        print("Applied INT8 dynamic quantization")

    model.eval()

    # Check model size
    tmp_path = "temp_check.pth"
    torch.save(model.state_dict(), tmp_path)
    size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    print(f"Model size: {size_mb:.1f} MB")
    os.remove(tmp_path)

    # Create dataset
    test_ds = CUHKXDataset(
        root=args.test_root,
        split_file=args.test_csv,
        is_train=False,
    )

    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_inference,
    )

    # Run inference
    predictions = []
    paths = []

    print(f"Running inference on {len(test_ds)} clips...")
    with torch.no_grad():
        for batch in test_loader:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)

            outputs = model(
                skeleton=batch["skeleton"],
                depth=batch["depth"],
                thermal=batch["thermal"],
                ir=batch["ir"],
                imu=batch["imu"],
                radar=batch["radar"],
                modality_mask=batch.get("mask"),
                training=False,
            )

            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().tolist())
            paths.extend(batch["path"])

    # Write submission
    output_path = args.output
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "prediction"])
        for path, pred in zip(paths, predictions):
            writer.writerow([path, pred])

    print(f"Generated {output_path} with {len(predictions)} predictions")


if __name__ == "__main__":
    main()
