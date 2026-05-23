"""CLI entry point for training anomaly detection models.

Usage:
    # Train PatchCore on bottle
    python scripts/train.py --model patchcore --category bottle

    # Train custom autoencoder
    python scripts/train.py --model autoencoder --category carpet

    # Train EfficientAD with custom epochs
    python scripts/train.py --model efficientad --category hazelnut --epochs 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train anomaly detection models on MVTec AD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/train.py --model patchcore --category bottle
    python scripts/train.py --model autoencoder --category carpet --epochs 50
    python scripts/train.py --model stfpm --category hazelnut --image-size 224
        """,
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default="patchcore",
        choices=["patchcore", "efficientad", "stfpm", "padim", "autoencoder"],
        help="Model to train (default: patchcore)",
    )
    parser.add_argument(
        "--category", "-c",
        type=str,
        default="bottle",
        help="MVTec AD category (default: bottle)",
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=None,
        help="Max training epochs (overrides config)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Input image size (default: 256)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'gpu', 'cpu' (default: auto)",
    )

    args = parser.parse_args()

    image_size = (args.image_size, args.image_size)

    if args.model == "autoencoder":
        # Use custom training loop
        from src.training.train_autoencoder import train_autoencoder

        device = None if args.device == "auto" else args.device
        result = train_autoencoder(
            category=args.category,
            device=device,
        )
        print(f"\n✓ Custom Autoencoder trained on '{args.category}'")
        print(f"  Image AUROC: {result['metrics']['image_auroc']:.4f}")
        print(f"  Image F1:    {result['metrics']['image_f1']:.4f}")
        print(f"  Checkpoint:  {result['checkpoint_path']}")

    else:
        # Use Anomalib training
        from src.training.train_anomalib import train_anomalib_model

        result = train_anomalib_model(
            model_name=args.model,
            category=args.category,
            max_epochs=args.epochs,
            image_size=image_size,
            accelerator=args.device,
        )
        print(f"\n✓ {args.model.title()} trained on '{args.category}'")
        print(f"  Results: {result['results_dir']}")


if __name__ == "__main__":
    main()
