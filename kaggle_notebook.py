"""
CUHK-X Small Model Track — Kaggle Training & Inference Notebook

Upload this as a Kaggle Notebook. The dataset is auto-mounted at:
  /kaggle/input/competitions/cuhk-x-competition-small-model-track/

Run all cells in order: Install → Train → Infer → Submit
"""

# ──────────────────────────────────────────────
# CELL 1: Install dependencies (run once)
# ──────────────────────────────────────────────
# !pip install -q torch torchvision numpy pandas scikit-learn Pillow tqdm

# ──────────────────────────────────────────────
# CELL 2: Imports & Constants
# ──────────────────────────────────────────────
import csv
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset

# ── Paths (Kaggle auto-mounts the competition data) ──
# On Kaggle, competitions mount directly under /kaggle/input/<competition-name>/
# The directory name may vary — auto-detect it below.
import glob as _glob

def _find_kaggle_input():
    """Auto-detect the Kaggle competition input directory."""
    for pattern in [
        "/kaggle/input/cuhk-x-competition-small-model-track",
        "/kaggle/input/competitions/cuhk-x-competition-small-model-track",
    ]:
        if os.path.isdir(pattern):
            return pattern
    # Fallback: search
    for d in _glob.glob("/kaggle/input/*"):
        if "cuhk" in d.lower() or "small" in d.lower():
            return d
    return None

KAGGLE_INPUT = _find_kaggle_input()
if KAGGLE_INPUT is None:
    print("WARNING: Could not auto-detect Kaggle input. Listing /kaggle/input/:")
    for d in sorted(os.listdir("/kaggle/input/") if os.path.exists("/kaggle/input/") else []):
        print(f"  {d}")
        sub = os.path.join("/kaggle/input", d)
        if os.path.isdir(sub):
            for f in sorted(os.listdir(sub))[:10]:
                print(f"    {f}")
    raise SystemExit("Set KAGGLE_INPUT manually above.")

print(f"KAGGLE_INPUT = {KAGGLE_INPUT}")

# Training data paths — Kaggle provides split zip volumes that need merging
TRAIN_DATA_DIR = f"{KAGGLE_INPUT}/Training/data"
TRAIN_ROOT     = f"{TRAIN_DATA_DIR}/HAR/data"  # after extraction

# Test data — Kaggle usually pre-extracts the zip
TEST_ROOT = f"{KAGGLE_INPUT}/Testing/data/small_model_track_test"
TEST_CSV  = f"{KAGGLE_INPUT}/Testing/test_file/test.csv"
OUTPUT_DIR = "/kaggle/working"

# Quick check — list what's available
print(f"\nContents of {TRAIN_DATA_DIR}:")
if os.path.isdir(TRAIN_DATA_DIR):
    for f in sorted(os.listdir(TRAIN_DATA_DIR))[:15]:
        sz = os.path.getsize(f"{TRAIN_DATA_DIR}/{f}") / (1024**3) if os.path.isfile(f"{TRAIN_DATA_DIR}/{f}") else 0
        print(f"  {f}{'  ' if os.path.isdir(f'{TRAIN_DATA_DIR}/{f}') else f'  ({sz:.1f} GB)'}")
else:
    print("  NOT FOUND — check KAGGLE_INPUT above")

# ── Hyperparameters ──
NUM_CLASSES   = 40
TARGET_FRAMES = 60
IMG_SIZE      = 112
HEATMAP_SIZE  = 56
RADAR_GRID    = 32
EMBED_DIM     = 512
BATCH_SIZE    = 8
EPOCHS        = 100
LR            = 1e-3
N_FOLDS       = 6
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_USERS    = list(range(1, 10)) + list(range(16, 25))
MODALITIES     = ["Depth_Color", "IR", "Thermal", "IMU", "Radar", "Skeleton"]
IMU_FEATURES   = ["AccX","AccY","AccZ","AsX","AsY","AsZ","AngleX","AngleY","AngleZ"]
SENSOR_ORDER   = ["WTC", "WTRA", "WTLA", "WTLL", "WTRL"]

print(f"Device: {DEVICE}")
print(f"Train root: {TRAIN_ROOT}  exists={os.path.isdir(TRAIN_ROOT)}")
print(f"Test root:  {TEST_ROOT}   exists={os.path.isdir(TEST_ROOT)}")


# ──────────────────────────────────────────────
# CELL 2.5: Extract training data from zip volumes (run once, takes ~5 min)
# ──────────────────────────────────────────────

import zipfile

def merge_and_extract_zip_volumes(data_dir, output_name="HAR_full.zip", extract_to=None):
    """Merge HAR.z01..HAR.z08 + HAR.zip and extract. Handles Kaggle's split zip format."""
    z01_path = f"{data_dir}/HAR.z01"
    zip_path = f"{data_dir}/HAR.zip"
    merged_path = f"/kaggle/working/{output_name}"

    if not os.path.isfile(z01_path) and not os.path.isfile(zip_path):
        print(f"No zip volumes found at {data_dir}/HAR.z01 or HAR.zip")
        print("Training data may already be extracted or in a different location.")
        return False

    if os.path.isdir(f"{data_dir}/HAR/data"):
        print("Training data already extracted: HAR/data/ exists ✓")
        return True

    print("Merging zip volumes (HAR.z01..HAR.zip)...")
    with open(merged_path, "wb") as out:
        for i in range(1, 9):
            part = f"{data_dir}/HAR.z0{i}"
            if os.path.isfile(part):
                print(f"  Reading {os.path.basename(part)} ({os.path.getsize(part)/1024**3:.1f} GB)")
                with open(part, "rb") as f:
                    out.write(f.read())
        # Finally HAR.zip (central directory)
        if os.path.isfile(zip_path):
            print(f"  Reading HAR.zip ({os.path.getsize(zip_path)/1024**3:.1f} GB)")
            with open(zip_path, "rb") as f:
                out.write(f.read())

    print(f"Merged: {merged_path} ({os.path.getsize(merged_path)/1024**3:.1f} GB)")

    if extract_to is None:
        extract_to = data_dir  # extract next to zip volumes
    print(f"Extracting to {extract_to} ...")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(merged_path) as zf:
        zf.extractall(extract_to)

    # Clean up merged file to save space
    os.remove(merged_path)
    print("Extraction complete! Removed merged zip to save space.")
    return True


# Extract test data zip if needed
test_zip = f"{KAGGLE_INPUT}/Testing/data/small_model_track_test.zip"
if os.path.isfile(test_zip) and not os.path.isdir(TEST_ROOT):
    print("Extracting test data...")
    with zipfile.ZipFile(test_zip) as zf:
        zf.extractall(f"{KAGGLE_INPUT}/Testing/data/")
    print("Test extraction complete.")

# Extract training data
merge_and_extract_zip_volumes(TRAIN_DATA_DIR)

print(f"\nTrain root: {TRAIN_ROOT}  exists={os.path.isdir(TRAIN_ROOT)}")
print(f"Test root:  {TEST_ROOT}   exists={os.path.isdir(TEST_ROOT)}")

if not os.path.isdir(TRAIN_ROOT):
    print("\n⚠️  TRAINING DATA NOT FOUND! Listing what's available:")
    for root, dirs, files in os.walk(TRAIN_DATA_DIR):
        level = root.replace(TRAIN_DATA_DIR, "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        if level > 2:
            break
        for d in dirs[:5]:
            print(f"{indent}  {d}/")
        for f in files[:5]:
            sz = os.path.getsize(os.path.join(root, f)) / 1024**3
            print(f"{indent}  {f} ({sz:.1f} GB)")



# ──────────────────────────────────────────────
# CELL 3: Dataset & Preprocessing
# ──────────────────────────────────────────────

def _resize_frame(img, size=IMG_SIZE):
    w, h = img.size
    short = min(w, h)
    left, top = (w - short)//2, (h - short)//2
    return img.crop((left, top, left+short, top+short)).resize((size, size), Image.BILINEAR)

def _temporal_interpolate(data, target_len):
    T = len(data)
    if T >= target_len:
        idx = np.linspace(0, T-1, target_len).astype(int)
        return data[idx]
    pad = np.repeat(data[-1:], target_len - T, axis=0)
    return np.concatenate([data, pad], axis=0)

def load_visual(mod_dir, n_frames=TARGET_FRAMES):
    if not os.path.isdir(mod_dir): return None
    files = sorted([f for f in os.listdir(mod_dir) if not f.startswith(".")])
    if not files: return None
    frames = []
    for f in files:
        try:
            img = Image.open(os.path.join(mod_dir, f)).convert("L")
            img = _resize_frame(img)
            frames.append(np.array(img, dtype=np.float32))
        except: continue
    if not frames: return None
    arr = np.stack(frames)
    arr = _temporal_interpolate(arr, n_frames)
    return torch.from_numpy(arr).unsqueeze(0)  # (1, T, H, W)

def load_imu(imu_dir, n_frames=TARGET_FRAMES):
    if not os.path.isdir(imu_dir): return None
    sensor_idx = {s:i for i,s in enumerate(SENSOR_ORDER)}
    readings = defaultdict(dict)
    for fname in sorted(os.listdir(imu_dir)):
        if not fname.endswith(".csv") or fname.startswith("."): continue
        try:
            with open(os.path.join(imu_dir, fname), encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    dev = row.get("DeviceName", row.get("设备名称", ""))
                    sensor = next((s for s in SENSOR_ORDER if s in dev), None)
                    if sensor is None: continue
                    feats = []
                    for feat in IMU_FEATURES:
                        try: feats.append(float(row[feat]))
                        except: feats.append(0.0)
                    readings[len(readings)][sensor] = feats
        except: continue
    if not readings: return None
    frames = []
    for ts in sorted(readings.keys()):
        frame = np.zeros(45, dtype=np.float32)
        for s, feats in readings[ts].items():
            idx = sensor_idx.get(s, -1)
            if idx >= 0: frame[idx*9:(idx+1)*9] = feats
        frames.append(frame)
    data = np.array(frames, dtype=np.float32)
    data = _temporal_interpolate(data, n_frames)
    return torch.from_numpy(data.T)  # (45, T)

def load_radar(radar_dir, n_frames=TARGET_FRAMES, gs=RADAR_GRID):
    if not os.path.isdir(radar_dir): return None
    all_pts = []
    for fname in sorted(os.listdir(radar_dir)):
        if not fname.endswith(".csv") or fname.startswith("."): continue
        try:
            with open(os.path.join(radar_dir, fname)) as f:
                for row in csv.DictReader(f):
                    try:
                        x, y, z = float(row.get("x",0)), float(row.get("y",0)), float(row.get("z",0))
                        all_pts.append((x, y, z))
                    except: continue
        except: continue
    if not all_pts:
        return torch.zeros(3, n_frames, gs, gs)
    pts = np.array(all_pts)
    for i in range(3):
        col = pts[:, i]
        valid = np.isfinite(col)
        if valid.any() and col[valid].std() > 0:
            pts[valid, i] = (col[valid] - col[valid].mean()) / col[valid].std()
    frames = np.zeros((3, n_frames, gs, gs), dtype=np.float32)
    indices = np.linspace(0, n_frames-1, len(pts)).astype(int)
    for j, idx in enumerate(indices):
        x, y, z = pts[j]
        xi = int(np.clip((x+3)/6*gs, 0, gs-1))
        yi = int(np.clip((y+3)/6*gs, 0, gs-1))
        zi = int(np.clip((z+3)/6*gs, 0, gs-1))
        frames[0, idx, yi, xi] += 1.0  # top
        frames[1, idx, zi, xi] += 1.0  # front
        frames[2, idx, zi, yi] += 1.0  # side
    frames = np.clip(frames, 0, 10) / 10.0
    return torch.from_numpy(frames)

def load_skeleton(skel_dir, n_frames=TARGET_FRAMES, n_joints=17, h=HEATMAP_SIZE):
    pred_dir = os.path.join(skel_dir, "predictions")
    if not os.path.isdir(pred_dir): pred_dir = skel_dir
    if not os.path.isdir(pred_dir): return None
    jf_list = sorted([f for f in os.listdir(pred_dir) if f.endswith(".json") and not f.startswith(".")])
    if not jf_list: return None
    all_kp = []
    for jf in jf_list:
        try:
            with open(os.path.join(pred_dir, jf)) as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                kpts = np.array(data[0].get("keypoints", []), dtype=np.float32)
                if len(kpts) >= n_joints: all_kp.append(kpts[:n_joints])
        except: continue
    if not all_kp: return None
    kp_arr = np.stack(all_kp)  # (T, J, 3)
    kp_arr = _temporal_interpolate(kp_arr, n_frames)
    hm = np.zeros((n_joints, n_frames, h, h), dtype=np.float32)
    sigma = 1.5
    gx, gy = np.meshgrid(np.arange(h), np.arange(h))
    for t in range(n_frames):
        for j in range(n_joints):
            xi = int(np.clip((kp_arr[t,j,0]+1)/2*h, 0, h-1))
            yi = int(np.clip((kp_arr[t,j,1]+1)/2*h, 0, h-1))
            hm[j, t] = np.exp(-((gx-xi)**2 + (gy-yi)**2) / (2*sigma**2))
    return torch.from_numpy(hm)  # (J, T, H, H)


class CUHKXDataset(Dataset):
    def __init__(self, root, split_file=None, is_train=True, samples_list=None):
        self.root = Path(root)
        self.is_train = is_train
        self.shapes = {
            "Skeleton": (17, TARGET_FRAMES, 56, 56),
            "Depth_Color": (1, TARGET_FRAMES, 112, 112),
            "Thermal": (1, TARGET_FRAMES, 112, 112),
            "IR": (1, TARGET_FRAMES, 112, 112),
            "IMU": (45, TARGET_FRAMES),
            "Radar": (3, TARGET_FRAMES, 32, 32),
        }
        self.mod_order = ["skeleton","depth","thermal","ir","imu","radar"]
        self.dir_names = ["Skeleton","Depth_Color","Thermal","IR","IMU","Radar"]

        if samples_list is not None:
            self.df = samples_list
            self.labels = [s["action_id"] for s in samples_list]
        elif split_file and os.path.exists(split_file):
            with open(split_file, encoding="utf-8-sig") as f:
                self.df = list(csv.DictReader(f))
            self.labels = [int(r.get("action_id", 0)) for r in self.df]
        else:
            self.df = []
            self.labels = []

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df[idx]
        sp = self.root / row["path"].rstrip("/")

        data, mask = {}, []
        for key, dn in zip(self.mod_order, self.dir_names):
            md = sp / dn
            if not md.exists():
                data[key] = torch.zeros(self.shapes[dn], dtype=torch.float32)
                mask.append(0.0)
                continue
            try:
                if dn == "IMU":       t = load_imu(str(md))
                elif dn == "Radar":   t = load_radar(str(md))
                elif dn == "Skeleton": t = load_skeleton(str(md))
                else:                  t = load_visual(str(md))
                if t is None:
                    data[key] = torch.zeros(self.shapes[dn], dtype=torch.float32)
                    mask.append(0.0)
                else:
                    data[key] = t
                    mask.append(1.0)
            except:
                data[key] = torch.zeros(self.shapes[dn], dtype=torch.float32)
                mask.append(0.0)

        return {
            "data": data,
            "label": self.labels[idx] if self.labels else 0,
            "modality_mask": mask,
            "path": row.get("path", ""),
        }


def discover_training_samples(data_root):
    """Scan HAR/data/<modality>/<action>/<user>/<trial>/ to build sample list."""
    samples = []
    dd = Path(data_root)
    for mod_dir in dd.iterdir():
        if not mod_dir.is_dir(): continue
        for action_dir in mod_dir.iterdir():
            if not action_dir.is_dir(): continue
            try: action_id = int(action_dir.name.split("_")[0])
            except: continue
            for user_dir in action_dir.iterdir():
                if not user_dir.is_dir(): continue
                try: user_id = int(user_dir.name.replace("user", ""))
                except: continue
                if user_id not in TRAIN_USERS: continue
                for trial_dir in user_dir.iterdir():
                    if not trial_dir.is_dir(): continue
                    samples.append({
                        "path": str(trial_dir.relative_to(data_root)) + "/",
                        "action_id": action_id,
                        "user_id": user_id,
                    })
    # Deduplicate (same trial appears under each modality)
    seen = set()
    unique = []
    for s in samples:
        if s["path"] not in seen:
            seen.add(s["path"])
            unique.append(s)
    return unique


def build_splits(data_root, n_splits=N_FOLDS):
    samples = discover_training_samples(data_root)
    paths  = np.array([s["path"] for s in samples])
    labels = np.array([s["action_id"] for s in samples])
    groups = np.array([s["user_id"] for s in samples])
    gkf = GroupKFold(n_splits=n_splits)
    splits = [(train_idx, val_idx) for train_idx, val_idx in gkf.split(paths, labels, groups)]
    return samples, splits


def collate_fn(batch):
    keys = ["skeleton","depth","thermal","ir","imu","radar"]
    result = {k: [] for k in keys}
    result["label"] = []
    result["mask"] = []
    for item in batch:
        for k in keys: result[k].append(item["data"][k])
        result["label"].append(item["label"])
        result["mask"].append(item.get("modality_mask", [1.0]*6))
    for k in keys: result[k] = torch.stack(result[k])
    result["label"] = torch.tensor(result["label"], dtype=torch.long)
    result["mask"] = torch.tensor(result["mask"], dtype=torch.float32)
    return result


# ──────────────────────────────────────────────
# CELL 4: Model Architecture
# ──────────────────────────────────────────────

class SkeletonEncoder(nn.Module):
    def __init__(self, in_c=17, out_dim=EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_c, 24, 3, padding=1, bias=False), nn.BatchNorm3d(24), nn.ReLU(inplace=True), nn.MaxPool3d((1,2,2)),
            nn.Conv3d(24, 48, 3, padding=1, bias=False), nn.BatchNorm3d(48), nn.ReLU(inplace=True), nn.MaxPool3d((2,2,2)),
            nn.Conv3d(48, 96, 3, padding=1, bias=False), nn.BatchNorm3d(96), nn.ReLU(inplace=True), nn.MaxPool3d((2,2,2)),
            nn.Conv3d(96, 192, 3, padding=1, bias=False), nn.BatchNorm3d(192), nn.ReLU(inplace=True), nn.MaxPool3d((2,2,2)),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(192, out_dim)
    def forward(self, x): return self.fc(self.pool(self.net(x)).flatten(1))

class VisualEncoder(nn.Module):
    def __init__(self, out_dim=EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 24, (1,3,3), (1,2,2), (0,1,1), bias=False), nn.BatchNorm3d(24), nn.ReLU(inplace=True),
            nn.Conv3d(24, 48, (1,3,3), (1,2,2), (0,1,1), bias=False), nn.BatchNorm3d(48), nn.ReLU(inplace=True),
            nn.Conv3d(48, 96, (3,3,3), (2,2,2), (1,1,1), bias=False), nn.BatchNorm3d(96), nn.ReLU(inplace=True),
            nn.Conv3d(96, 192, (3,3,3), (2,2,2), (1,1,1), bias=False), nn.BatchNorm3d(192), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(192, out_dim)
    def forward(self, x): return self.fc(self.pool(self.net(x)).flatten(1))

class IMUEncoder(nn.Module):
    def __init__(self, in_c=45, out_dim=EMBED_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_c, 48, 5, padding=2, bias=False), nn.BatchNorm1d(48), nn.ReLU(inplace=True),
            nn.Conv1d(48, 96, 3, 2, 1, bias=False), nn.BatchNorm1d(96), nn.ReLU(inplace=True),
            nn.Conv1d(96, 192, 3, 2, 1, bias=False), nn.BatchNorm1d(192), nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(192, 192, 2, batch_first=True, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(384, out_dim)
    def forward(self, x):
        x = self.conv(x).permute(0, 2, 1)
        _, h = self.gru(x)
        return self.fc(torch.cat([h[-2], h[-1]], dim=-1))

class RadarEncoder(nn.Module):
    def __init__(self, in_c=3, out_dim=EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_c, 24, (1,3,3), (1,2,2), (0,1,1), bias=False), nn.BatchNorm3d(24), nn.ReLU(inplace=True),
            nn.Conv3d(24, 48, (1,3,3), (1,2,2), (0,1,1), bias=False), nn.BatchNorm3d(48), nn.ReLU(inplace=True),
            nn.Conv3d(48, 96, (3,1,1), (2,1,1), (1,0,0), bias=False), nn.BatchNorm3d(96), nn.ReLU(inplace=True),
            nn.Conv3d(96, 192, (3,1,1), (2,1,1), (1,0,0), bias=False), nn.BatchNorm3d(192), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(192, out_dim)
    def forward(self, x): return self.fc(self.pool(self.net(x)).flatten(1))

class GatedFusion(nn.Module):
    def __init__(self, M=6, D=EMBED_DIM):
        super().__init__()
        self.M, self.D = M, D
        self.gate = nn.Sequential(nn.Linear(M*D, D), nn.ReLU(inplace=True), nn.Linear(D, M), nn.Sigmoid())
    def forward(self, embs, mask):
        B = embs[0].size(0)
        stacked = torch.stack(embs, dim=1)  # (B, M, D)
        gates = self.gate(stacked.reshape(B, -1)) * mask
        fused = (gates.unsqueeze(-1) * stacked).sum(dim=1)
        return fused / mask.sum(dim=1, keepdim=True).clamp(min=1) * self.M

class CUHKXModel(nn.Module):
    def __init__(self, nc=NUM_CLASSES, D=EMBED_DIM, dropout=0.4):
        super().__init__()
        self.skel_enc = SkeletonEncoder(out_dim=D)
        self.depth_enc = VisualEncoder(out_dim=D)
        self.ir_enc = VisualEncoder(out_dim=D)
        self.thermal_enc = VisualEncoder(out_dim=D)
        self.imu_enc = IMUEncoder(out_dim=D)
        self.radar_enc = RadarEncoder(out_dim=D)
        self.fusion = GatedFusion(M=6, D=D)
        self.cls = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(D, D//2), nn.ReLU(inplace=True),
            nn.Dropout(dropout*0.5), nn.Linear(D//2, nc),
        )
        self._mod_drop = 0.2
        self._single_drop = 0.05

    def forward(self, skeleton, depth, thermal, ir, imu, radar, modality_mask=None, training=True):
        mods = [skeleton, depth, thermal, ir, imu, radar]
        B = mods[0].size(0)
        device = mods[0].device
        if modality_mask is None:
            modality_mask = torch.ones(B, 6, device=device)

        if training:
            mods, modality_mask = self._apply_dropout(mods, modality_mask)

        embs = [
            self.skel_enc(mods[0]),
            self.depth_enc(mods[1]),
            self.thermal_enc(mods[2]),
            self.ir_enc(mods[3]),
            self.imu_enc(mods[4]),
            self.radar_enc(mods[5]),
        ]
        fused = self.fusion(embs, modality_mask)
        return self.cls(fused)

    def _apply_dropout(self, mods, mask):
        B, device = mods[0].size(0), mods[0].device
        new_mask = mask.clone()
        new_mods = list(mods)
        if random.random() < self._single_drop:
            active = torch.where(mask.sum(dim=0) > 0)[0]
            if len(active) > 0:
                ki = random.choice(active.tolist())
                new_mask.zero_()
                new_mask[:, ki] = mask[:, ki]
                for i in range(6):
                    if i != ki: new_mods[i] = torch.zeros_like(mods[i])
            return new_mods, new_mask
        for i in range(6):
            if random.random() < self._mod_drop:
                new_mask[:, i] = 0.0
                new_mods[i] = torch.zeros_like(mods[i])
        if new_mask.sum() == 0:
            active = torch.where(mask.sum(dim=0) > 0)[0]
            if len(active) > 0:
                ki = random.choice(active.tolist())
                new_mask[:, ki] = 1.0
                new_mods[ki] = mods[ki]
        return new_mods, new_mask

    def count_params(self):
        return sum(p.numel() for p in self.parameters())
    def size_mb(self):
        return self.count_params() * 4 / (1024**2)


# ──────────────────────────────────────────────
# CELL 5: Training
# ──────────────────────────────────────────────

def train_epoch(model, loader, crit, opt, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for batch in loader:
        for k in batch:
            if isinstance(batch[k], torch.Tensor): batch[k] = batch[k].to(device)
        labels = batch["label"]
        opt.zero_grad()
        out = model(batch["skeleton"], batch["depth"], batch["thermal"], batch["ir"],
                     batch["imu"], batch["radar"], batch["mask"], training=True)
        loss = crit(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_sum += loss.item()
        correct += (out.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return loss_sum/len(loader), correct/total

@torch.no_grad()
def validate(model, loader, crit, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for batch in loader:
        for k in batch:
            if isinstance(batch[k], torch.Tensor): batch[k] = batch[k].to(device)
        labels = batch["label"]
        out = model(batch["skeleton"], batch["depth"], batch["thermal"], batch["ir"],
                     batch["imu"], batch["radar"], batch["mask"], training=False)
        loss_sum += crit(out, labels).item()
        correct += (out.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return loss_sum/len(loader), correct/total


def train_model(data_root, output_dir=OUTPUT_DIR, epochs=EPOCHS):
    os.makedirs(output_dir, exist_ok=True)
    samples, splits = build_splits(data_root)
    print(f"Training samples: {len(samples)}")

    best_accs = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"\n{'='*50}\nFold {fold_idx+1}/{N_FOLDS}\n{'='*50}")

        train_s = [samples[i] for i in train_idx]
        val_s   = [samples[i] for i in val_idx]
        print(f"Train: {len(train_s)}  Val: {len(val_s)}")

        train_ds = CUHKXDataset(data_root, samples_list=train_s)
        val_ds   = CUHKXDataset(data_root, samples_list=val_s)

        train_ld = DataLoader(train_ds, BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)
        val_ld   = DataLoader(val_ds, BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)

        model = CUHKXModel().to(DEVICE)
        print(f"Params: {model.count_params():,}  Size: {model.size_mb():.1f} MB")

        crit = nn.CrossEntropyLoss(label_smoothing=0.1)
        opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=epochs, eta_min=1e-6)

        best_acc = 0.0
        for ep in range(epochs):
            tl, ta = train_epoch(model, train_ld, crit, opt, DEVICE)
            vl, va = validate(model, val_ld, crit, DEVICE)
            sched.step()

            if va > best_acc:
                best_acc = va
                torch.save({"model": model.state_dict(), "epoch": ep+1, "val_acc": va},
                           f"{output_dir}/best_fold{fold_idx}.pth")

            sz = os.path.getsize(f"{output_dir}/best_fold{fold_idx}.pth")/(1024**2) if os.path.exists(f"{output_dir}/best_fold{fold_idx}.pth") else 0
            print(f"Ep {ep+1:3d} | TrL:{tl:.4f} TrA:{ta:.4f} | VlL:{vl:.4f} VlA:{va:.4f} | {sz:.1f}MB")

        best_accs.append(best_acc)
        print(f"Fold {fold_idx+1} best val acc: {best_acc:.4f}")

    print(f"\nAverage CV accuracy: {np.mean(best_accs):.4f} ± {np.std(best_accs):.4f}")
    return best_accs


# ──────────────────────────────────────────────
# CELL 6: Run Training
# ──────────────────────────────────────────────

if __name__ == "__main__" or True:
    train_model(TRAIN_ROOT)


# ──────────────────────────────────────────────
# CELL 7: Inference & Submission
# ──────────────────────────────────────────────

@torch.no_grad()
def generate_submission(checkpoint_path, test_root, test_csv, output_csv="submission.csv"):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model = CUHKXModel().to(DEVICE)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    ds = CUHKXDataset(test_root, split_file=test_csv, is_train=False)
    loader = DataLoader(ds, BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)

    preds, paths = [], []
    for batch in loader:
        for k in batch:
            if isinstance(batch[k], torch.Tensor): batch[k] = batch[k].to(DEVICE)
        out = model(batch["skeleton"], batch["depth"], batch["thermal"], batch["ir"],
                     batch["imu"], batch["radar"], batch["mask"], training=False)
        preds.extend(out.argmax(1).cpu().tolist())
        paths.extend(batch["path"])

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "prediction"])
        for p, pr in zip(paths, preds):
            writer.writerow([p, pr])
    print(f"Generated {output_csv} ({len(preds)} predictions)")
    return output_csv


# ──────────────────────────────────────────────
# CELL 8: Generate submission (pick the best fold)
# ──────────────────────────────────────────────

import glob

# Find all fold checkpoints and pick the one with best val_acc
checkpoints = sorted(glob.glob(f"{OUTPUT_DIR}/best_fold*.pth"))
if not checkpoints:
    print("ERROR: No checkpoints found! Did training complete?")
    print(f"Looking in: {OUTPUT_DIR}")
    print(f"Contents: {os.listdir(OUTPUT_DIR) if os.path.exists(OUTPUT_DIR) else 'dir not found'}")
else:
    best_ckpt = None
    best_acc = -1
    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        acc = ckpt.get("val_acc", 0)
        print(f"  {os.path.basename(ckpt_path)}: val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_ckpt = ckpt_path

    print(f"\nBest checkpoint: {os.path.basename(best_ckpt)} (val_acc={best_acc:.4f})")
    print("Running inference on 405 test clips...")
    output = generate_submission(best_ckpt, TEST_ROOT, TEST_CSV, "/kaggle/working/submission.csv")
    print(f"\n✅ Done! Download '{output}' and submit at:")
    print("   https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track/submit")
