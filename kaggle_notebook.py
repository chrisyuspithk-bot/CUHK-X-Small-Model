"""
CUHK-X Small Model Track — Standalone Kaggle Notebook

Fully self-contained: downloads data from HuggingFace, trains, and submits.
No data mounting needed — just run the 9 cells in order.

Cells:
  1: pip install all dependencies
  2: Imports & hyperparameters
  3: Download data from HuggingFace (~41.5 GB, ~20 min)
  4: Dataset & preprocessing
  5: Model architecture
  6: Training functions
  7: Run training (6-fold GroupKFold CV)
  8: Inference function
  9: Generate submission.csv
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 1 of 9: Install dependencies                          ║
# ╚══════════════════════════════════════════════════════════════╝
!pip install -q torch torchvision numpy pandas scikit-learn Pillow tqdm huggingface_hub

# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 2 of 9: Imports & Hyperparameters                     ║
# ╚══════════════════════════════════════════════════════════════╝
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

# Placeholders — Cell 3 will set the real paths after downloading
TRAIN_ROOT = "/kaggle/working/data/HAR/data"
TEST_ROOT  = "/kaggle/working/data/small_model_track_test"
TEST_CSV   = "/kaggle/working/data/test.csv"
OUTPUT_DIR = "/kaggle/working"

print(f"Device: {DEVICE}")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 3 of 9: Get the data                                   ║
# ║  ⚠️  Use Kaggle Add Data (free, no disk cost) NOT download   ║
# ╚══════════════════════════════════════════════════════════════╝

import zipfile, shutil
from huggingface_hub import hf_hub_download

HF_REPO = "Kevin-Pal/CUHK-X_Small_Model_Track"
DATA_DIR = "/kaggle/working/data"
TRAIN_VOLUMES = [f"HAR.z0{i}" for i in range(1, 9)] + ["HAR.zip"]
TRAIN_SUBDIR = "Small-Model-Track/Training/data"

os.makedirs(DATA_DIR, exist_ok=True)

# Clean up stale files from previous failed runs
for stale in os.listdir(DATA_DIR):
    if stale.startswith("HAR.z") or stale == "HAR_full.zip":
        os.remove(os.path.join(DATA_DIR, stale))
        print(f"Cleaned up stale file: {stale}")

# ──────────────────────────────────────────────────────────
# STEP 1: Try Kaggle Add Data (no disk cost — USE THIS!)
#   Notebook sidebar → Data → Add Input → Competitions
#   → CUHK-X Small Model Track
# ──────────────────────────────────────────────────────────

KAGGLE_MOUNTED_TRAIN = None
KAGGLE_MOUNTED_TEST = None
KAGGLE_MOUNTED_CSV = None

for kaggle_dir in [
    "/kaggle/input/cuhk-x-competition-small-model-track",
    "/kaggle/input/competitions/cuhk-x-competition-small-model-track",
]:
    if not os.path.isdir(kaggle_dir):
        continue
    print(f"Found Kaggle input: {kaggle_dir}")
    for root, dirs, files in os.walk(kaggle_dir):
        # Pre-extracted training data
        if os.path.basename(root) == "data" and any(d in dirs for d in ["Depth_Color", "Skeleton"]):
            KAGGLE_MOUNTED_TRAIN = root
        # Pre-extracted test data
        if "small_model_track_test" in dirs:
            KAGGLE_MOUNTED_TEST = os.path.join(root, "small_model_track_test")
        # Test zip
        if "small_model_track_test.zip" in files:
            test_zip_src = os.path.join(root, "small_model_track_test.zip")
            if not KAGGLE_MOUNTED_TEST and not os.path.isdir(f"{DATA_DIR}/small_model_track_test"):
                import shutil
                shutil.copy2(test_zip_src, f"{DATA_DIR}/small_model_track_test.zip")
                with zipfile.ZipFile(f"{DATA_DIR}/small_model_track_test.zip") as zf:
                    zf.extractall(DATA_DIR)
                os.remove(f"{DATA_DIR}/small_model_track_test.zip")
                KAGGLE_MOUNTED_TEST = f"{DATA_DIR}/small_model_track_test"
        # test.csv
        if "test.csv" in files:
            KAGGLE_MOUNTED_CSV = os.path.join(root, "test.csv")
    # Find zip volumes — merge directly from Kaggle input (don't copy!)
    for root, dirs, files in os.walk(kaggle_dir):
        if any(f.startswith("HAR.z0") for f in files) or "HAR.zip" in files:
            KAGGLE_ZIP_DIR = root
            print(f"Found zip volumes in: {root}")
            if not os.path.isdir(f"{DATA_DIR}/HAR/data"):
                print("Merging directly from Kaggle input (no disk waste)...")
                merged = f"{DATA_DIR}/HAR_full.zip"
                with open(merged, "wb") as out:
                    for vol in TRAIN_VOLUMES:
                        src = os.path.join(root, vol)
                        if os.path.isfile(src):
                            out.write(open(src, "rb").read())
                print(f"Extracting ({os.path.getsize(merged)/1024**3:.1f} GB)...")
                with zipfile.ZipFile(merged) as zf:
                    zf.extractall(DATA_DIR)
                os.remove(merged)
                print("Extraction complete ✓")
            break
    break

if KAGGLE_MOUNTED_TRAIN:
    TRAIN_ROOT = KAGGLE_MOUNTED_TRAIN
    print(f"✅ Using Kaggle-mounted training data: {TRAIN_ROOT}")
else:
    if os.path.isdir(f"{DATA_DIR}/HAR/data"):
        print("Training data already extracted ✓")
    else:
        print("\n❌ No data found!")
        print("You MUST add the competition data to this notebook:")
        print("  Notebook sidebar → Data → Add Input → Competitions")
        print("  → CUHK-X Small Model Track")
        raise SystemExit("Add the competition data via the sidebar.")
    TRAIN_ROOT = f"{DATA_DIR}/HAR/data"

if KAGGLE_MOUNTED_TEST:
    TEST_ROOT = KAGGLE_MOUNTED_TEST
    print(f"✅ Using Kaggle-mounted test data: {TEST_ROOT}")
else:
    TEST_ROOT = f"{DATA_DIR}/small_model_track_test"

if KAGGLE_MOUNTED_CSV:
    TEST_CSV = KAGGLE_MOUNTED_CSV
    print(f"✅ Using Kaggle-mounted test.csv: {TEST_CSV}")
else:
    # Kaggle test data zip usually includes test.csv alongside the clips
    for candidate in [
        f"{DATA_DIR}/test.csv",
        f"{DATA_DIR}/small_model_track_test/test.csv",
    ]:
        if os.path.isfile(candidate):
            TEST_CSV = candidate
            break
    else:
        # Download just test.csv from HF (small file, won't fill disk)
        csv_local = f"{DATA_DIR}/test.csv"
        if not os.path.isfile(csv_local):
            print("Downloading test.csv (small file)...")
            downloaded = hf_hub_download(HF_REPO,
                filename="Small-Model-Track/Testing/test_file/test.csv",
                repo_type="dataset", local_dir=DATA_DIR)
            if downloaded != csv_local and os.path.isfile(downloaded):
                shutil.move(downloaded, csv_local)
        TEST_CSV = csv_local

OUTPUT_DIR = "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Verify ──
import shutil as _shutil
disk = _shutil.disk_usage("/kaggle/working")
print(f"\n── Data status ──")
print(f"Train root: {TRAIN_ROOT}  exists={os.path.isdir(TRAIN_ROOT)}")
print(f"Test root:  {TEST_ROOT}   exists={os.path.isdir(TEST_ROOT)}")
print(f"Test CSV:   {TEST_CSV}    exists={os.path.isfile(TEST_CSV)}")
print(f"Free disk:  {disk.free / 1024**3:.1f} GB / {disk.total / 1024**3:.1f} GB total")

if os.path.isdir(TRAIN_ROOT):
    mods = [d for d in os.listdir(TRAIN_ROOT) if os.path.isdir(os.path.join(TRAIN_ROOT, d))]
    print(f"Training modalities: {mods}")
if os.path.isdir(TEST_ROOT):
    try:
        samples = len([d for d in os.listdir(TEST_ROOT) if d.startswith("SM_test")])
        print(f"Test samples: {samples}")
    except: pass

if not os.path.isdir(TRAIN_ROOT):
    raise SystemExit(
        "\n❌ Training data NOT found!\n"
        "You must add the competition data to this notebook:\n"
        "  Kaggle notebook sidebar → Data → Add Input → Competitions\n"
        "  → CUHK-X Small Model Track → Add\n"
    )



# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 4 of 9: Dataset & Preprocessing                      ║
# ╚══════════════════════════════════════════════════════════════╝

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


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 5 of 9: Model Architecture                            ║
# ╚══════════════════════════════════════════════════════════════╝

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


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 6 of 9: Training functions                            ║
# ╚══════════════════════════════════════════════════════════════╝

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


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 7 of 9: Run Training (6-fold CV, ~30 min/fold GPU)   ║
# ╚══════════════════════════════════════════════════════════════╝

# Run training on all 6 folds
if not os.path.isdir(TRAIN_ROOT):
    print(f"\n❌ Training data NOT found at: {TRAIN_ROOT}")
    print("Did you run Cell 3 (extraction) yet?")
    print("If extraction completed but path is wrong, set TRAIN_ROOT manually above.")
else:
    train_model(TRAIN_ROOT)


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 8 of 9: Inference function                            ║
# ╚══════════════════════════════════════════════════════════════╝

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


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 9 of 9: Generate submission.csv                       ║
# ╚══════════════════════════════════════════════════════════════╝

import glob

# Find all fold checkpoints and pick the one with best val_acc
checkpoints = sorted(glob.glob(f"{OUTPUT_DIR}/best_fold*.pth"))
if not checkpoints:
    print("ERROR: No checkpoints found! Did training complete?")
    print(f"Looking in: {OUTPUT_DIR}")
    print(f"Contents: {os.listdir(OUTPUT_DIR) if os.path.exists(OUTPUT_DIR) else 'dir not found'}")
    print("\nMake sure Cell 7 (training) ran successfully first.")
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
