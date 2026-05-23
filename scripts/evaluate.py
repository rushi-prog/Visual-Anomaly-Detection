"""CLI entry point for evaluating trained models.

Usage:
    python scripts/evaluate.py --model patchcore --category bottle
    python scripts/evaluate.py --model autoencoder --category bottle --checkpoint results/custom_autoencoder/bottle/best_model.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained anomaly detection models",
    )

    parser.add_argument("--model", "-m", type=str, required=True, help="Model name")
    parser.add_argument("--category", "-c", type=str, default="bottle", help="MVTec category")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--save-viz", action="store_true", help="Save visualization plots")

    args = parser.parse_args()

    if args.model == "autoencoder":
        _evaluate_autoencoder(args)
    else:
        _evaluate_anomalib(args)


def _evaluate_anomalib(args: argparse.Namespace) -> None:
    """Evaluate an Anomalib model."""
    from anomalib.engine import Engine
    from src.models.model_factory import create_model
    from src.data.mvtec_explorer import get_mvtec_datamodule

    model = create_model(args.model)
    datamodule = get_mvtec_datamodule(args.category)

    engine = Engine()

    ckpt = args.checkpoint
    if ckpt is None:
        from src.utils.config import RESULTS_DIR
        ckpt_dir = RESULTS_DIR / args.model / args.category
        ckpt_files = list(ckpt_dir.rglob("*.ckpt")) if ckpt_dir.exists() else []
        if ckpt_files:
            ckpt = str(max(ckpt_files, key=lambda p: p.stat().st_mtime))
        else:
            print(f"No checkpoint found. Train the model first:")
            print(f"  python scripts/train.py --model {args.model} --category {args.category}")
            return

    results = engine.test(model=model, datamodule=datamodule, ckpt_path=ckpt)
    print(f"\nTest Results for {args.model} on {args.category}:")
    for metric_dict in results:
        for k, v in metric_dict.items():
            print(f"  {k}: {v}")


def _evaluate_autoencoder(args: argparse.Namespace) -> None:
    """Evaluate the custom autoencoder."""
    import torch
    import numpy as np
    from src.models.autoencoder import AnomalyAutoencoder
    from src.data.mvtec_explorer import get_mvtec_datamodule
    from src.evaluation.metrics import compute_image_metrics
    from src.utils.config import RESULTS_DIR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = RESULTS_DIR / "custom_autoencoder" / args.category / "best_model.pt"

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        print(f"No checkpoint found at {ckpt_path}")
        print(f"Train the model first:")
        print(f"  python scripts/train.py --model autoencoder --category {args.category}")
        return

    # Load model
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location=device)
    config = checkpoint.get("config", {})
    model_config = config.get("model", {})

    model = AnomalyAutoencoder(
        in_channels=model_config.get("in_channels", 3),
        latent_dim=model_config.get("latent_dim", 128),
        encoder_channels=model_config.get("encoder_channels"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load data
    datamodule = get_mvtec_datamodule(args.category, num_workers=0)
    datamodule.setup()
    test_loader = datamodule.test_dataloader()

    # Collect predictions
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"]
            scores = model.compute_anomaly_score(images)
            all_scores.extend(scores.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist() if hasattr(labels, "numpy") else labels)

    metrics = compute_image_metrics(
        np.array(all_labels),
        np.array(all_scores),
    )

    print(f"\nTest Results for custom_autoencoder on {args.category}:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        elif not isinstance(v, list):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
