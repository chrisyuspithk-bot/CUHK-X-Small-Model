"""
CUHK-X Small Model Track Dataset and Preprocessing Pipeline.

Handles 6 modalities: Depth_Color, IR, Thermal, IMU, Radar, Skeleton.
Supports cross-subject splits, stochastic modality dropout, and
temporal windowing (T=60 frames).

Directory structure:
  Training: HAR/data/<modality>/<action>/<user>/<trial>/<files>
  Testing:  small_model_track_test/<id>/<modality>/<files>
"""

import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
NUM_CLASSES = 40
TARGET_FRAMES = 60
IMG_SIZE = 112
RADAR_GRID_SIZE = 32
NUM_JOINTS = 17
HEATMAP_SIZE = 56
IMU_FEATURES = 45  # 5 sensors × 9 features
IMU_SENSOR_FEATURES = ["AccX", "AccY", "AccZ", "AsX", "AsY", "AsZ", "AngleX", "AngleY", "AngleZ"]

TRAIN_USERS = list(range(1, 10)) + list(range(16, 25))  # 1-9, 16-24
TEST_USERS = [10, 11, 25, 26]

ACTION_NAMES = [
    "Wash_face", "Brush_teeth", "Comb_hair", "Take_off_clothes", "Wipe_hands",
    "Put_on_clothes", "Drink_water", "Eat_food", "Take_and_use_tableware", "Pour_drinks",
    "Stir_drinks", "Peel_fruits", "Sweep_the_floor", "Mop_the_floor", "Wipe_bowls",
    "Wipe_windows_and_tables", "Fold_clothes", "Tap_the_keyboard", "Write", "Make_a_phone_call",
    "Check_the_time", "Read_documents", "Turn_pages", "Listen_to_music_with_headphones",
    "Use_a_mobile_phone", "Watch_TV", "Play_games", "Take_a_selfie", "Jog_in_place",
    "Do_squats", "Do_jumping_jacks", "Do_stretching_exercises", "Stand_up", "Lie_down",
    "Sit_down", "Do_lunges", "Walk", "Take_medicine", "Massage_oneself", "Take_body_temperature",
]

MODALITIES = ["Depth_Color", "IR", "Thermal", "IMU", "Radar", "Skeleton"]


# ──────────────────────────────────────────────
# Preprocessing helpers
# ──────────────────────────────────────────────

def _resize_frame(img, size=IMG_SIZE):
    """Center-crop and resize an image to (size, size)."""
    w, h = img.size
    short = min(w, h)
    left = (w - short) // 2
    top = (h - short) // 2
    img = img.crop((left, top, left + short, top + short))
    return img.resize((size, size), Image.BILINEAR)


def _load_visual_frames(frame_dir, num_frames=TARGET_FRAMES, img_size=IMG_SIZE):
    """Load and preprocess visual frames (Depth_Color, IR, Thermal).
    Returns tensor of shape (1, T, H, W) for single-channel,
    or (3, T, H, W) for RGB.
    """
    if not os.path.isdir(frame_dir):
        return None
    files = sorted([f for f in os.listdir(frame_dir) if not f.startswith(".")])
    if not files:
        return None

    frames = []
    for f in files:
        try:
            img = Image.open(os.path.join(frame_dir, f)).convert("L")  # grayscale
            img = _resize_frame(img, img_size)
            frames.append(np.array(img, dtype=np.float32))
        except Exception:
            continue

    if not frames:
        return None

    frames = _temporal_interpolate(np.stack(frames), num_frames)
    return torch.from_numpy(frames).unsqueeze(0)  # (1, T, H, W)


def _temporal_interpolate(data, target_len):
    """Interpolate or trim temporal data to exactly `target_len` frames."""
    T = len(data)
    if data.ndim == 3:  # (T, J, C) - skeleton keypoints
        if T >= target_len:
            idx = np.linspace(0, T - 1, target_len).astype(int)
            return data[idx]
        pad = np.repeat(data[-1:], target_len - T, axis=0)
        return np.concatenate([data, pad], axis=0)
    elif data.ndim == 2:  # (T, features) or (T, D)
        if T >= target_len:
            idx = np.linspace(0, T - 1, target_len).astype(int)
            return data[idx]
        pad = np.repeat(data[-1:], target_len - T, axis=0)
        return np.concatenate([data, pad], axis=0)
    # For 1D case or fallback
    return data


def _load_imu(imu_dir, num_frames=TARGET_FRAMES):
    """Load IMU CSV files and produce (45, T) tensor.
    Two files: up(LA+RA+C).csv (3 sensors: chest, right arm, left arm)
               down(LL+RL).csv (2 sensors: left leg, right leg)
    5 sensors × 9 features = 45 channels.
    """
    if not os.path.isdir(imu_dir):
        return None

    SENSOR_ORDER = ["WTC", "WTRA", "WTLA", "WTLL", "WTRL"]
    SENSOR_TO_IDX = {s: i for i, s in enumerate(SENSOR_ORDER)}

    # Parse all rows, keyed by (rounded_timestamp, sensor_name)
    readings = defaultdict(dict)  # ts -> {sensor_name: [9 features]}
    for fname in sorted(os.listdir(imu_dir)):
        if not fname.endswith(".csv") or fname.startswith("."):
            continue
        fpath = os.path.join(imu_dir, fname)
        try:
            with open(fpath, encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    device_name = row.get("DeviceName", row.get("设备名称", ""))
                    # Extract sensor key (WTC, WTRA, WTLA, WTLL, WTRL)
                    sensor = None
                    for s in SENSOR_ORDER:
                        if s in device_name:
                            sensor = s
                            break
                    if sensor is None:
                        continue

                    ts_ms = int(_parse_timestamp(row) / 10)  # round to 10ms buckets
                    features = []
                    for feat in IMU_SENSOR_FEATURES:
                        try:
                            features.append(float(row[feat]))
                        except (KeyError, ValueError):
                            features.append(0.0)
                    readings[ts_ms][sensor] = features
        except Exception:
            continue

    if not readings:
        return None

    # Build (T, 45) array by grouping sensors at each timestamp
    timestamps = sorted(readings.keys())
    all_frames = []
    for ts in timestamps:
        frame = np.zeros(45, dtype=np.float32)
        for sensor_name, features in readings[ts].items():
            idx = SENSOR_TO_IDX.get(sensor_name, -1)
            if idx >= 0:
                frame[idx * 9 : (idx + 1) * 9] = features
        all_frames.append(frame)

    data = np.array(all_frames, dtype=np.float32)  # (N, 45)
    data = _temporal_interpolate(data, num_frames)  # (T, 45)
    return torch.from_numpy(data.T)  # (45, T)


def _parse_timestamp(row):
    """Parse timestamp from IMU row for sorting."""
    for key in ["time", "时间"]:
        if key in row:
            try:
                return float(row[key].replace("-", "").replace(" ", "").replace(":", "")[:14])
            except ValueError:
                pass
    return 0.0


def _load_radar(radar_dir, num_frames=TARGET_FRAMES, grid_size=RADAR_GRID_SIZE):
    """Load mmWave radar CSV, voxelize, and project to 2D pseudo-images.
    Returns (3, T, H, W) tensor (top/front/side projections).
    """
    if not os.path.isdir(radar_dir):
        return None

    all_points = []
    for fname in sorted(os.listdir(radar_dir)):
        if not fname.endswith(".csv") or fname.startswith("."):
            continue
        fpath = os.path.join(radar_dir, fname)
        try:
            with open(fpath) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        x = float(row.get("x", 0))
                        y = float(row.get("y", 0))
                        z = float(row.get("z", 0))
                        v = float(row.get("v", 0))
                        snr = float(row.get("snr", 0))
                        ts = float(row.get("timestamp", 0))
                        all_points.append((ts, x, y, z, v, snr))
                    except (ValueError, KeyError):
                        continue
        except Exception:
            continue

    if not all_points:
        # Return zero tensor for empty radar
        return torch.zeros(3, num_frames, grid_size, grid_size, dtype=torch.float32)

    all_points.sort(key=lambda x: x[0])
    pts = np.array(all_points)  # (N, 6): ts, x, y, z, v, snr

    # Normalize spatial coords
    for i in range(1, 4):
        col = pts[:, i]
        valid = np.isfinite(col)
        if valid.any():
            mean, std = col[valid].mean(), col[valid].std()
            if std > 0:
                pts[valid, i] = (col[valid] - mean) / std

    # Split into T temporal bins
    frames = np.zeros((3, num_frames, grid_size, grid_size), dtype=np.float32)
    if len(pts) > 0:
        # Assign to frames based on sorted order
        indices = np.linspace(0, num_frames - 1, len(pts)).astype(int)
        for j, idx in enumerate(indices):
            x, y, z = pts[j, 1], pts[j, 2], pts[j, 3]
            # Top view (X-Y)
            xi = int(np.clip((x + 3) / 6 * grid_size, 0, grid_size - 1))
            yi = int(np.clip((y + 3) / 6 * grid_size, 0, grid_size - 1))
            frames[0, idx, yi, xi] += 1.0
            # Front view (X-Z)
            zi = int(np.clip((z + 3) / 6 * grid_size, 0, grid_size - 1))
            frames[1, idx, zi, xi] += 1.0
            # Side view (Y-Z)
            frames[2, idx, zi, yi] += 1.0

    # Clip max values
    frames = np.clip(frames, 0, 10) / 10.0
    return torch.from_numpy(frames)


def _load_skeleton(skel_dir, num_frames=TARGET_FRAMES, num_joints=NUM_JOINTS, h=HEATMAP_SIZE):
    """Load skeleton JSON files and generate 3D heatmaps.
    Returns (J, T, H, H) tensor (J=17 joints).
    """
    pred_dir = os.path.join(skel_dir, "predictions") if os.path.isdir(os.path.join(skel_dir, "predictions")) else skel_dir
    if not os.path.isdir(pred_dir):
        return None

    json_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".json") and not f.startswith(".")])
    if not json_files:
        return None

    all_keypoints = []
    for jf in json_files:
        try:
            with open(os.path.join(pred_dir, jf)) as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                kpts = np.array(data[0].get("keypoints", []), dtype=np.float32)
                if len(kpts) >= num_joints:
                    all_keypoints.append(kpts[:num_joints])
        except Exception:
            continue

    if not all_keypoints:
        return None

    kp_array = np.stack(all_keypoints)  # (T, J, 3)
    kp_array = _temporal_interpolate(kp_array, num_frames)  # (T, J, 3)

    # Generate Gaussian heatmaps
    heatmaps = np.zeros((num_joints, num_frames, h, h), dtype=np.float32)
    sigma = 1.5
    xs = np.arange(h)
    ys = np.arange(h)
    gx, gy = np.meshgrid(xs, ys)

    for t in range(num_frames):
        for j in range(num_joints):
            x, y = kp_array[t, j, 0], kp_array[t, j, 1]
            # Normalize coordinates to heatmap grid
            xi = int(np.clip((x + 1) / 2 * h, 0, h - 1))
            yi = int(np.clip((y + 1) / 2 * h, 0, h - 1))
            heatmaps[j, t] = np.exp(-((gx - xi) ** 2 + (gy - yi) ** 2) / (2 * sigma ** 2))

    return torch.from_numpy(heatmaps)  # (J, T, H, H)


# ──────────────────────────────────────────────
# Dataset class
# ──────────────────────────────────────────────

class CUHKXDataset(Dataset):
    """PyTorch Dataset for CUHK-X Small Model Track.

    Args:
        root: Path to training data root (e.g., HAR/data/) or test data root.
        split_file: Path to CSV with columns [path, action_id] (train) or [path] (test).
        is_train: Training or inference mode.
        modalities: List of modalities to load.
        num_frames: Temporal window size.
        modality_dropout_prob: Probability of dropping a modality during training.
        single_modality_prob: Probability of keeping only one modality.
    """

    def __init__(
        self,
        root,
        split_file,
        is_train=True,
        modalities=None,
        num_frames=TARGET_FRAMES,
        modality_dropout_prob=0.2,
        single_modality_prob=0.05,
    ):
        self.root = Path(root)
        self.is_train = is_train
        self.num_frames = num_frames
        self.modalities = modalities or MODALITIES
        self.modality_dropout_prob = modality_dropout_prob if is_train else 0.0
        self.single_modality_prob = single_modality_prob if is_train else 0.0

        # Load CSV
        self.df = []
        with open(split_file, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                self.df.append(row)

        # Build action mapping for training
        if is_train and "action_id" in self.df[0]:
            self.labels = [int(row.get("action_id", 0)) for row in self.df]
        else:
            self.labels = None

        # Map action names to IDs
        self.name_to_id = {name: i for i, name in enumerate(ACTION_NAMES)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Load all modalities for a single sample and return dict with label."""
        row = self.df[idx]
        sample_path = self.root / row["path"].rstrip("/")

        shapes = {
            "Skeleton": (17, self.num_frames, 56, 56),
            "Depth_Color": (1, self.num_frames, 112, 112),
            "Thermal": (1, self.num_frames, 112, 112),
            "IR": (1, self.num_frames, 112, 112),
            "IMU": (45, self.num_frames),
            "Radar": (3, self.num_frames, 32, 32),
        }

        mod_order = ["skeleton", "depth", "thermal", "ir", "imu", "radar"]
        dir_names = ["Skeleton", "Depth_Color", "Thermal", "IR", "IMU", "Radar"]

        data = {}
        modality_mask = []
        for key, dir_name in zip(mod_order, dir_names):
            mod_dir = sample_path / dir_name
            if not mod_dir.exists():
                data[key] = torch.zeros(shapes[dir_name], dtype=torch.float32)
                modality_mask.append(0.0)
                continue

            try:
                if dir_name == "IMU":
                    tensor = _load_imu(str(mod_dir), self.num_frames)
                elif dir_name == "Radar":
                    tensor = _load_radar(str(mod_dir), self.num_frames)
                elif dir_name == "Skeleton":
                    tensor = _load_skeleton(str(mod_dir), self.num_frames)
                else:
                    tensor = _load_visual_frames(str(mod_dir), self.num_frames)

                if tensor is None:
                    data[key] = torch.zeros(shapes[dir_name], dtype=torch.float32)
                    modality_mask.append(0.0)
                else:
                    data[key] = tensor
                    modality_mask.append(1.0)
            except Exception:
                data[key] = torch.zeros(shapes[dir_name], dtype=torch.float32)
                modality_mask.append(0.0)

        label = self.labels[idx] if self.labels is not None else 0

        return {
            "data": data,
            "label": label,
            "path": row.get("path", ""),
            "modality_mask": modality_mask,
        }

    def _discover_training_samples(self):
        """Scan training directory to build list of (path, action_id, user_id)."""
        samples = []
        data_dir = self.root / "data" if (self.root / "data").exists() else self.root
        for mod_dir_name in os.listdir(str(data_dir)):
            mod_dir = data_dir / mod_dir_name
            if not mod_dir.is_dir():
                continue
            for action_dir in mod_dir.iterdir():
                if not action_dir.is_dir():
                    continue
                action_name = action_dir.name
                # Extract action_id from name like "0_Wash_face"
                action_id = None
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
                    for trial_dir in user_dir.iterdir():
                        if not trial_dir.is_dir():
                            continue
                        # Construct relative path
                        rel_path = str(trial_dir.relative_to(self.root))
                        samples.append({
                            "path": rel_path + "/",
                            "action_id": action_id,
                            "user_id": user_id,
                        })
        return samples


def build_train_splits(data_root, n_splits=6, train_users=None):
    """Build GroupKFold splits ensuring no subject leakage.

    Returns list of (train_indices, val_indices) pairs.
    """
    from sklearn.model_selection import GroupKFold

    if train_users is None:
        train_users = TRAIN_USERS

    dataset = CUHKXDataset(data_root, None, is_train=True)
    samples = dataset._discover_training_samples()

    # Filter to training users only
    samples = [s for s in samples if s["user_id"] in train_users]

    paths = np.array([s["path"] for s in samples])
    labels = np.array([s["action_id"] for s in samples])
    groups = np.array([s["user_id"] for s in samples])

    gkf = GroupKFold(n_splits=n_splits)
    splits = []
    for train_idx, val_idx in gkf.split(paths, labels, groups):
        train_users_set = set(groups[train_idx])
        val_users_set = set(groups[val_idx])
        assert len(train_users_set & val_users_set) == 0, "Data leakage!"
        splits.append((train_idx, val_idx))

    return samples, splits
