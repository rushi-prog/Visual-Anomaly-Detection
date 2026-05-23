# 🔬 Visual Anomaly Detection

> Detect defects and anomalies in images using one-class learning — where you have abundant "normal" data but few or zero examples of defects.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Anomalib](https://img.shields.io/badge/Anomalib-Intel-0071C5.svg)](https://github.com/openvinotoolkit/anomalib)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12+-EE4C2C.svg)](https://pytorch.org/)

## 🎯 Project Overview

This project explores **visual anomaly detection** — a critical capability for manufacturing quality control, semiconductor inspection, and industrial automation. We implement and compare multiple approaches:

| Model | Architecture | Best For |
|-------|-------------|----------|
| **PatchCore** | Memory Bank (Coreset) | Highest accuracy, interpretable |
| **EfficientAD** | Student-Teacher + Autoencoder | Real-time edge deployment |
| **STFPM** | Student-Teacher Feature Pyramid | Balanced speed/accuracy |
| **Custom Autoencoder** | Convolutional Autoencoder | Understanding fundamentals |

All models are benchmarked on the **MVTec AD** dataset (15 industrial categories).

## 🏗️ Architecture

```
src/
├── data/                   # Dataset loading & exploration
│   ├── mvtec_explorer.py   # MVTec AD visualization
│   └── custom_dataset.py   # Bring-your-own-data support
├── models/                 # Model definitions
│   ├── model_factory.py    # Anomalib model instantiation
│   └── autoencoder.py      # Custom convolutional autoencoder
├── training/               # Training pipelines
│   ├── train_anomalib.py   # Anomalib Engine wrapper
│   └── train_autoencoder.py # Custom PyTorch training loop
├── evaluation/             # Metrics & visualization
│   ├── metrics.py          # AUROC, F1, PRO scoring
│   ├── benchmark.py        # Multi-model comparison
│   └── visualize.py        # Heatmap overlays & plots
└── utils/
    └── config.py           # Configuration management
```

## 🚀 Quick Start

### Setup
```bash
# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

### Train a Model
```bash
# Train PatchCore on bottle category (fastest — single pass)
python scripts/train.py --model patchcore --category bottle

# Train custom autoencoder
python scripts/train.py --model autoencoder --category bottle

# Train STFPM
python scripts/train.py --model stfpm --category carpet --epochs 100
```

### Run Benchmark
```bash
# Compare PatchCore + STFPM across 5 categories
python scripts/benchmark_all.py

# Include custom autoencoder
python scripts/benchmark_all.py --include-autoencoder
```

### Launch Interactive Demo
```bash
python scripts/demo.py
# Opens at http://localhost:7860
```

### Evaluate a Trained Model
```bash
python scripts/evaluate.py --model patchcore --category bottle
python scripts/evaluate.py --model autoencoder --category carpet
```

## 🧠 Key Concepts

### One-Class Learning
We train models using **only normal (defect-free) images**. At inference, anything that deviates from the learned "normal" distribution is flagged as anomalous.

### Anomaly Scoring
Each pixel gets an **anomaly score** forming a heatmap. The image-level score is derived from the heatmap (typically the maximum value). A threshold separates normal from anomalous.

### Model Approaches

1. **PatchCore**: Stores patch-level features in a memory bank → detects anomalies via nearest-neighbor distance
2. **EfficientAD**: Student network mimics teacher on normal data → discrepancies indicate anomalies
3. **STFPM**: Multi-scale feature pyramid matching between teacher and student
4. **Custom Autoencoder**: Reconstructs normal images → high reconstruction error = anomaly

## 📊 Evaluation Metrics

| Metric | What It Measures |
|--------|-----------------|
| **Image AUROC** | Separability of normal vs anomalous images |
| **Pixel AUROC** | Pixel-level anomaly localization quality |
| **F1 Score** | Threshold-dependent classification balance |
| **PRO** | Per-region overlap quality |

## 📂 MVTec AD Dataset

15 industrial categories — 10 objects + 5 textures:

**Objects**: bottle, cable, capsule, hazelnut, metal_nut, pill, screw, toothbrush, transistor, zipper

**Textures**: carpet, grid, leather, tile, wood

The dataset auto-downloads on first training run (~350MB per category).

## 🛠️ Tech Stack

- **Anomalib** — Intel's anomaly detection library (20+ algorithms)
- **PyTorch** — Deep learning framework
- **OpenCV** — Image processing
- **Gradio** — Interactive web demo
- **scikit-learn** — Metrics computation

## 📋 Interview Prep

Key topics this project demonstrates:
- One-class classification / unsupervised anomaly detection
- Autoencoder architecture design & reconstruction-based detection
- Student-teacher knowledge distillation
- Memory-bank / coreset methods
- Anomaly scoring, thresholding, and ROC analysis
- Industrial quality control applications
