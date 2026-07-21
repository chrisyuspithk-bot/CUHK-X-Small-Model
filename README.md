# CUHK-X Small Model Track: Multimodal HAR Pipeline

End-to-end pipeline for the [CUHK-X Competition Small Model Track](https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track) — a multimodal human activity recognition system for edge deployment.

**Key constraints satisfied:**
- Model size **< 100 MB** (6.2M params, 23.8 MB FP32, ~15 MB INT8)
- CNN/RNN architectures only — no large pretrained backbones
- Cross-subject validation with GroupKFold (no subject leakage)
- Handles missing modalities via zero-fill + gated fusion masks

---

## 🚀 Quick Start — Run on Kaggle (Recommended)

Kaggle gives you **free GPU (T4/P100)** and the dataset is **pre-mounted** — no downloads needed.

### 1. Open a Kaggle Notebook

Go to https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track/code → **New Notebook**

### 2. Upload `kaggle_notebook.py`

Copy the entire contents of [`kaggle_notebook.py`](kaggle_notebook.py) into the notebook cells, or upload the file directly.

**Enable GPU**: Right sidebar → Accelerator → **GPU T4 x2**

### 3. Run all cells in order

| Cell | What it does |
|------|-------------|
| 1-2 | Imports, auto-detects Kaggle data paths |
| 3 | Dataset class, preprocessing, temporal interpolation |
| 4 | Model architecture (6 encoders + Gated Fusion) |
| 5-6 | **Training** — GroupKFold cross-subject validation, 6 folds, 100 epochs |
| 7-8 | **Inference** — generates `submission.csv` for all 405 test clips |

### 4. Submit

After the notebook finishes, `submission.csv` appears in the **Output** section of the notebook sidebar. Click **Submit to Competition**.

---

## 📁 Project Structure

```
cuhkx/                    # Modular Python package (for local training)
├── dataset.py            # CUHKXDataset, preprocessing, temporal interpolation
├── models/
│   ├── encoders.py       # Skeleton, IMU, Radar, SharedVisual encoders
│   └── fusion.py         # GatedMultimodalFusion + CUHKXModel
├── train.py              # Training loop with GroupKFold CV
└── inference.py          # Inference + submission generation

kaggle_notebook.py        # 🚀 Self-contained Kaggle notebook (just copy-paste!)
```

---

## 🏗️ Architecture

```
Skeleton (17×T×56×56)  →  3D-CNN (PoseConv3D-style)  ─┐
Depth   (1×T×112×112)  →  MobileNetV3-style 3D-CNN    ─┤
IR      (1×T×112×112)  →  MobileNetV3-style 3D-CNN    ─┤
Thermal (1×T×112×112)  →  MobileNetV3-style 3D-CNN    ─┼─ Gated Fusion → Classifier → 40 classes
IMU     (45×T)         →  1D-CNN + BiGRU              ─┤
Radar   (3×T×32×32)    →  Pseudo-3D CNN               ─┘
```

### Key Technical Features

- **Gated Multimodal Fusion (GMF)**: Dynamic per-modality, per-sample weighting
- **Stochastic Modality Dropout**: 20% per modality + 5% single-modality (prevents over-dependence on any one sensor)
- **Zero-fill for missing modalities**: Model adaptively suppresses missing sensors via fusion masks
- **GroupKFold cross-subject validation**: 6-fold, zero subject leakage between train/val
- **Label smoothing** (α=0.1) + **Cosine annealing** with warm restarts
- **Knowledge distillation** support for teacher-student training
- **INT8 dynamic quantization** support

---

## 🖥️ Local Training (alternative)

```bash
pip install -r requirements.txt

# Download data from Hugging Face (~41.5 GB, needs ~80 GB disk)
# See kaggle_notebook.py for the download script, or use Kaggle directly.

# Train
python -m cuhkx.train \
    --data_root /path/to/HAR/data \
    --output_dir ./output \
    --epochs 100 --batch_size 8 \
    --lr 1e-3 --embed_dim 512 --n_folds 6

# Inference
python -m cuhkx.inference \
    --checkpoint ./output/best_fold0.pth \
    --test_root /path/to/parent/of/small_model_track_test \
    --test_csv ./test.csv \
    --output submission.csv
```

---

## 📊 Model Size

| Precision | Parameters | Size |
|-----------|-----------|------|
| FP32 | 6,237,670 | ~23.8 MB |
| INT8 (quantized) | 6,237,670 | ~15 MB |
| **Competition limit** | — | **< 100 MB** ✅ |

---

## 📝 Submission Format

The competition expects a CSV with 405 rows:
```
path,prediction
small_model_track_test/SM_test_0001/,14
small_model_track_test/SM_test_0002/,7
...
small_model_track_test/SM_test_0405/,22
```

Upload at: https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track/submit

---

## 🏆 Tips for Leaderboard

1. **Knowledge distillation** — train a larger teacher first, then distill into the 6.2M student
2. **Fold ensemble** — average logits from all 6 fold checkpoints
3. **More epochs** — try 150-200 epochs with warm restarts
4. **Augmentation** — add temporal jitter, spatial flip for visual modalities

---

## ✅ Competition Rules Compliance

- [x] Model ≤ 100 MB
- [x] CNN/RNN/Transformer architectures only
- [x] No large pretrained backbones
- [x] No closed-source APIs / LLMs
- [x] Cross-subject validation (GroupKFold)
- [x] Handles missing modalities at test time
