"""CLI script to run the full benchmark comparison.

Usage:
    # Default benchmark (PatchCore + STFPM on 5 categories)
    python scripts/benchmark_all.py

    # Include custom autoencoder
    python scripts/benchmark_all.py --include-autoencoder

    # Specific models and categories
    python scripts/benchmark_all.py --models patchcore stfpm --categories bottle carpet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark anomaly detection models on MVTec AD",
    )

    parser.add_argument(
        "--models", "-m",
        nargs="+",
        default=None,
        help="Models to benchmark (default: patchcore, stfpm)",
    )
    parser.add_argument(
        "--categories", "-c",
        nargs="+",
        default=None,
        help="MVTec categories (default: bottle, carpet, hazelnut, screw, tile)",
    )
    parser.add_argument(
        "--include-autoencoder",
        action="store_true",
        help="Include custom autoencoder in benchmark",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for results",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.include_autoencoder:
        from src.evaluation.benchmark import run_benchmark_with_autoencoder
        df = run_benchmark_with_autoencoder(
            categories=args.categories,
            output_dir=output_dir,
        )
    else:
        from src.evaluation.benchmark import run_benchmark
        df = run_benchmark(
            models=args.models,
            categories=args.categories,
            output_dir=output_dir,
        )

    print(f"\n{'='*60}")
    print(f"  Benchmark Summary")
    print(f"{'='*60}")

    # Print aggregate stats per model
    if "image_auroc" in df.columns:
        summary = df.groupby("model").agg({
            "image_auroc": ["mean", "std", "min", "max"],
            "training_time_s": "sum",
        }).round(4)
        print(summary)


if __name__ == "__main__":
    main()
