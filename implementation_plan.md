# 🔬 Visual Anomaly Detection — Implementation Plan

Detect defects/anomalies in images with zero/few anomaly examples. Build a production-grade anomaly detection system using **Anomalib** (Intel's library) + **PyTorch** + **OpenCV**, benchmarked on **MVTec AD**.

---

## Proposed Changes

### Project Structure

```
tier2/
├── pyproject.toml                  # Project config (uv managed)
├── .python-version                 # Python 3.12
├── README.md                       # Project documentation
├── configs/
│   ├── patchcore.yaml              # PatchCore hyperparams
│   ├── efficientad.yaml            # EfficientAD hyperparams
│   ├── stfpm.yaml                  # STFPM hyperparams
│   └── autoencoder.yaml            # Custom autoencoder config
├── notebooks/
│   ├── 01_explore_mvtec.ipynb      # Dataset exploration & visualization
│   ├── 02_anomalib_quickstart.ipynb # Quick experiments with Anomalib
│   └── 03_autoencoder_scratch.ipynb # Build autoencoder from scratch
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── mvtec_explorer.py       # MVTec dataset loading & visualization
│   │   └── custom_dataset.py       # Folder-based custom dataset wrapper
│   ├── models/
│   │   ├── __init__.py
│   │   ├── autoencoder.py          # Custom convolutional autoencoder
│   │   └── model_factory.py        # Anomalib model instantiation factory
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_anomalib.py       # Train any Anomalib model
│   │   ├── train_autoencoder.py    # Custom autoencoder training loop
│   │   └── callbacks.py            # Custom callbacks (visualization, logging)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py              # AUROC, F1, per-pixel metrics
│   │   ├── benchmark.py            # Multi-model comparison pipeline
│   │   └── visualize.py            # Heatmap overlays, anomaly maps
│   └── utils/
│       ├── __init__.py
│       └── config.py               # Config loading utilities
├── scripts/
│   ├── train.py                    # CLI entry point for training
│   ├── evaluate.py                 # CLI entry point for evaluation
│   ├── benchmark_all.py            # Run all models & compare
│   └── demo.py                     # Gradio interactive demo
├── tests/
│   ├── test_phase1.py              # Dataset + Anomalib basics
│   ├── test_phase2.py              # Custom autoencoder
│   └── test_phase3.py              # Benchmarking pipeline
└── results/                        # Training outputs, checkpoints, plots
    └── .gitkeep
```

---

### Phase 1 — MVTec AD + Anomalib Exploration (Core Foundation)
> Learn anomaly detection fundamentals by training PatchCore, EfficientAD, and STFPM on MVTec AD.

#### [NEW] `pyproject.toml`
- Project metadata, dependencies: `anomalib`, `torch`, `torchvision`, `opencv-python`, `matplotlib`, `seaborn`, `gradio`, `pyyaml`, `tqdm`, `scikit-learn`

#### [NEW] `src/data/mvtec_explorer.py`
- Load MVTec AD dataset via Anomalib's built-in `MVTec` datamodule
- Visualize normal vs. anomalous samples side-by-side
- Display ground truth masks overlaid on anomaly images
- Dataset statistics: image counts per category, resolution info, anomaly type distribution

#### [NEW] `src/models/model_factory.py`
- Factory function to instantiate any Anomalib model by name (`patchcore`, `efficientad`, `stfpm`, `padim`)
- Default config loading from `configs/` directory
- Support for overriding hyperparameters

#### [NEW] `src/training/train_anomalib.py`
- Wrapper around `anomalib.engine.Engine` for training
- Auto-downloads MVTec AD on first run
- Configurable: model name, category, image size, batch size
- Saves checkpoints + training logs

#### [NEW] `configs/patchcore.yaml`, `configs/efficientad.yaml`, `configs/stfpm.yaml`
- Model-specific configurations (backbone, layers to extract, image size, batch size)
- PatchCore: backbone=`wide_resnet50_2`, coreset sampling ratio
- EfficientAD: teacher model, student-teacher + autoencoder balance
- STFPM: feature pyramid layers, backbone selection

#### [NEW] `notebooks/01_explore_mvtec.ipynb`
- Interactive exploration of MVTec AD
- Visualize all 15 categories with sample images
- Anomaly mask overlays
- Per-category anomaly type breakdown

#### [NEW] `notebooks/02_anomalib_quickstart.ipynb`
- Train PatchCore on "bottle" category
- Visualize anomaly heatmaps and predictions
- Try different thresholds and see their effect on precision/recall

---

### Phase 2 — Custom Autoencoder from Scratch
> Deep understanding of one-class learning by building an autoencoder-based anomaly detector without Anomalib.

#### [NEW] `src/models/autoencoder.py`
- Convolutional Autoencoder architecture:
  - **Encoder**: Conv2d → BN → ReLU → MaxPool (4 blocks), progressively reducing spatial dims
  - **Bottleneck**: Compressed latent representation
  - **Decoder**: ConvTranspose2d → BN → ReLU (4 blocks), progressively upsampling
- Anomaly scoring: pixel-wise reconstruction error (MSE or SSIM)
- Anomaly map generation: difference between input and reconstruction

#### [NEW] `src/training/train_autoencoder.py`
- Standard PyTorch training loop (no Anomalib dependency)
- Train on **normal images only** (one-class learning)
- Loss: MSE + optional SSIM loss for perceptual quality
- Learning rate scheduling (CosineAnnealing)
- Validation: compute reconstruction error on held-out normal + anomaly images
- Threshold determination: fit Gaussian to normal reconstruction errors, set threshold at μ + kσ

#### [NEW] `configs/autoencoder.yaml`
- Architecture params: latent_dim, num_channels, image_size
- Training params: lr, epochs, batch_size, scheduler

#### [NEW] `notebooks/03_autoencoder_scratch.ipynb`
- Step-by-step build of the autoencoder
- Visualize encoder feature maps at each layer
- Reconstruction quality: normal vs. anomalous inputs
- Error heatmap generation and comparison with Anomalib models

---

### Phase 3 — Comparative Benchmarking Pipeline
> Systematically compare all models (PatchCore, EfficientAD, STFPM, Custom AE) across multiple MVTec categories.

#### [NEW] `src/evaluation/metrics.py`
- Compute: Image-level AUROC, Pixel-level AUROC, F1-Score, PRO (Per-Region Overlap)
- Per-category and aggregate results
- Threshold sweep for optimal F1

#### [NEW] `src/evaluation/benchmark.py`
- Run all models on selected MVTec categories (e.g., bottle, carpet, hazelnut, screw, tile)
- Collect metrics in a structured DataFrame
- Timing: training time, inference latency (ms/image)
- Memory usage tracking

#### [NEW] `src/evaluation/visualize.py`
- Anomaly heatmap overlays (jet colormap on original image)
- Side-by-side comparison: input → anomaly map → predicted mask → ground truth
- Per-model comparison grid for same test image
- Bar charts: AUROC by category by model
- ROC curves overlay for all models

#### [NEW] `scripts/benchmark_all.py`
- CLI script to run the full benchmark
- Outputs: results table (CSV), comparison plots, per-category breakdowns

---

### Phase 4 — Interactive Demo & Deployment
> Build a polished Gradio app for interactive anomaly detection with model selection.

#### [NEW] `scripts/demo.py`
- Gradio interface with:
  - **Model selector**: dropdown to pick PatchCore / EfficientAD / STFPM / Custom AE
  - **Image upload**: drag-and-drop or webcam capture
  - **Outputs**: original image, anomaly heatmap overlay, anomaly score gauge, pass/fail classification
  - **Threshold slider**: adjust sensitivity in real-time
  - **Gallery**: pre-loaded example images (normal + anomalous)
- OpenVINO export option for optimized inference

---

## User Review Required

> [!IMPORTANT]
> **GPU Availability**: This project benefits significantly from a GPU. PatchCore and EfficientAD training on MVTec AD takes ~5-15 minutes on GPU vs. potentially hours on CPU. Do you have a CUDA GPU available?

> [!IMPORTANT]
> **MVTec AD Categories**: The full dataset is ~4.9 GB (15 categories). To save time/space we can start with 3-5 categories (e.g., bottle, carpet, hazelnut, screw, tile) which gives good coverage of both object and texture anomalies. Want the full dataset or a subset?

## Open Questions

1. **Custom Dataset**: After MVTec, do you want to also try your own custom dataset (e.g., photos from your phone of good/defective items)? This would make the project more portfolio-worthy.

2. **OpenVINO Export**: Do you want to include OpenVINO optimization for Intel CPU deployment, or keep it simpler with just PyTorch inference? (OpenVINO is great for interviews at Intel/semiconductor companies but adds complexity.)

3. **Scope Preference**: This plan is comprehensive (4 phases). Would you prefer to:
   - **Full build**: All 4 phases as described
   - **Lean build**: Phases 1-3 only (skip Gradio demo, add it later)
   - **Express build**: Phase 1 + 2 only (Anomalib + custom autoencoder, skip benchmarking)

---

## Verification Plan

### Automated Tests
- **Phase 1**: Verify MVTec data loads correctly, model factory creates valid models, training runs without errors on 1 category
- **Phase 2**: Verify autoencoder forward pass shape, training loop convergence on small subset, anomaly map generation
- **Phase 3**: Verify metrics computation matches expected ranges, benchmark script produces valid CSV output

### Manual Verification
- Visual inspection of anomaly heatmaps — do they highlight actual defects?
- Reconstruction quality of custom autoencoder — sharp outputs vs. blurry?
- Benchmark results table — do rankings match published literature?
- Gradio demo — responsive, correct predictions on uploaded images

### Expected Results (Literature Reference)
| Model | Image AUROC (bottle) | Notes |
|-------|---------------------|-------|
| PatchCore | ~99.5% | Near-perfect on simple objects |
| EfficientAD | ~99.0% | Faster inference, competitive |
| STFPM | ~97-98% | Lightweight, good baseline |
| Custom AE | ~85-92% | Lower but demonstrates concepts |
