# CUHK-X Small Model Track: Multimodal HAR Pipeline

End-to-end pipeline for the CUHK-X Competition Small Model Track — a multimodal human activity recognition system for edge deployment.

**Key constraints satisfied:**
- Model size **< 100 MB** (6.2M params, 23.8 MB FP32, ~15 MB INT8)
- CNN/RNN architectures only — no large pretrained backbones
- Cross-subject validation with GroupKFold (no subject leakage)
- Handles missing modalities via zero-fill + gated fusion masks

## Architecture

```
Skeleton (17×T×56×56)  →  3D-CNN (PoseConv3D-style)  ─┐
Depth   (1×T×112×112)  →  MobileNetV3-style 3D-CNN    ─┤
IR      (1×T×112×112)  →  MobileNetV3-style 3D-CNN    ─┤
Thermal (1×T×112×112)  →  MobileNetV3-style 3D-CNN    ─┼─ Gated Fusion → Classifier → 40 classes
IMU     (45×T)         →  1D-CNN + BiGRU              ─┤
Radar   (3×T×32×32)    →  Pseudo-3D CNN               ─┘
```

### Key Features

- **Gated Multimodal Fusion (GMF)**: Dynamic weighting per modality per sample
- **Stochastic Modality Dropout**: Randomly drops modalities during training (20% each, 5% single-modality)
- **Zero-fill for missing modalities**: Catastrophic failure prevention
- **GroupKFold cross-subject validation**: 6-fold split ensuring no user appears in both train/val
- **Label smoothing** (α=0.1) + **Cosine annealing** with warm restarts
- **Knowledge distillation** support for teacher-student training
- **INT8 dynamic quantization** support for further compression

## Project Structure

```
cuhkx/
├── __init__.py
├── dataset.py          # Dataset, preprocessing, temporal interpolation
├── models/
│   ├── __init__.py
│   ├── encoders.py     # Skeleton, IMU, Radar, SharedVisual encoders
│   └── fusion.py       # GatedMultimodalFusion + CUHKXModel
├── train.py            # Training loop with GroupKFold CV
└── inference.py        # Inference + submission generation
```

## Setup

```bash
pip install -r requirements.txt
```

## Data Preparation

### Option A: From Hugging Face

```bash
python -c "
from huggingface_hub import hf_hub_download
# Download all training volumes
for vol in ['HAR.z01','HAR.z02','HAR.z03','HAR.z04','HAR.z05','HAR.z06','HAR.z07','HAR.z08','HAR.zip']:
    hf_hub_download('Kevin-Pal/CUHK-X_Small_Model_Track',
        filename=f'Small-Model-Track/Training/data/{vol}',
        repo_type='dataset', token='YOUR_HF_TOKEN', local_dir='./data')
# Download test data
hf_hub_download('Kevin-Pal/CUHK-X_Small_Model_Track',
    filename='Small-Model-Track/Testing/data/small_model_track_test.zip',
    repo_type='dataset', token='YOUR_HF_TOKEN', local_dir='./data')
"
# Merge and extract training data
cd data/Small-Model-Track/Training/data
zip -s 0 HAR.zip --out HAR_full.zip
unzip HAR_full.zip -d ../../..
# Extract test data
cd ../../Testing/data
unzip small_model_track_test.zip -d ../../..
```

### Option B: From Kaggle

The dataset is available directly at `/kaggle/input/competitions/cuhk-x-competition-small-model-track/` when running on Kaggle.

## Training

```bash
python -m cuhkx.train \
    --data_root /path/to/HAR/data \
    --output_dir ./output \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-3 \
    --embed_dim 512 \
    --n_folds 6
```

### Training with Knowledge Distillation

First train a large teacher model (unrestricted size), then:

```bash
python -m cuhkx.train \
    --data_root /path/to/HAR/data \
    --teacher_checkpoint ./teacher_model.pth \
    --kd_alpha 0.7
```

## Inference

```bash
python -m cuhkx.inference \
    --checkpoint ./output/best_fold0.pth \
    --test_root /path/to/parent/of/small_model_track_test \
    --test_csv ./data/Small-Model-Track/Testing/test_file/test.csv \
    --output submission.csv \
    --quantize  # Optional: apply INT8 quantization
```

## Model Size

| Precision | Parameters | Size |
|-----------|-----------|------|
| FP32 | 6,237,670 | ~23.8 MB |
| INT8 (quantized) | 6,237,670 | ~15 MB |
| Limit | — | <100 MB |

## Competition Rules Compliance

- [x] Model ≤ 100 MB
- [x] CNN/RNN/Transformer architectures only
- [x] No large pretrained backbones
- [x] No closed-source APIs
- [x] No LLM-based labeling
- [x] Cross-subject validation (GroupKFold)
- [x] Handles missing modalities at test time
