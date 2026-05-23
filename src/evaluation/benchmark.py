"""Multi-model benchmarking pipeline for anomaly detection.

Systematically compares PatchCore, EfficientAD, STFPM, and Custom Autoencoder
across MVTec AD categories. Collects metrics, timing, and generates comparison
reports.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import ensure_dir, RESULTS_DIR


def run_benchmark(
    models: list[str] | None = None,
    categories: list[str] | None = None,
    image_size: tuple[int, int] = (256, 256),
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Run a full benchmark comparing multiple models across categories.

    For each (model, category) pair:
    1. Trains the model
    2. Evaluates on test set
    3. Records metrics + timing

    Args:
        models: List of model names. Defaults to ["patchcore", "efficientad", "stfpm"].
        categories: List of MVTec AD categories. Defaults to representative subset.
        image_size: Input image resolution.
        output_dir: Directory to save results. Defaults to results/benchmark/.

    Returns:
        DataFrame with all results indexed by (model, category).
    """
    from src.training.train_anomalib import train_anomalib_model

    if models is None:
        models = ["patchcore", "stfpm"]  # EfficientAD can be slow, add if needed

    if categories is None:
        categories = ["bottle", "carpet", "hazelnut", "screw", "tile"]

    if output_dir is None:
        output_dir = RESULTS_DIR / "benchmark"
    ensure_dir(output_dir)

    all_results = []

    total_combos = len(models) * len(categories)
    current = 0

    for model_name in models:
        for category in categories:
            current += 1
            print(f"\n{'━'*60}")
            print(f"  Benchmark [{current}/{total_combos}]: {model_name} × {category}")
            print(f"{'━'*60}")

            start_time = time.time()

            try:
                result = train_anomalib_model(
                    model_name=model_name,
                    category=category,
                    image_size=image_size,
                )
                elapsed = time.time() - start_time

                # Extract metrics from test results
                test_metrics = {}
                if result.get("test_results"):
                    # Anomalib returns a list of metric dicts
                    for metric_dict in result["test_results"]:
                        if isinstance(metric_dict, dict):
                            test_metrics.update(metric_dict)

                entry = {
                    "model": model_name,
                    "category": category,
                    "training_time_s": round(elapsed, 1),
                    "image_auroc": test_metrics.get("image_AUROC", None),
                    "pixel_auroc": test_metrics.get("pixel_AUROC", None),
                    "image_f1": test_metrics.get("image_F1Score", None),
                    "status": "success",
                }

            except Exception as e:
                elapsed = time.time() - start_time
                print(f"  ✗ Failed: {e}")
                entry = {
                    "model": model_name,
                    "category": category,
                    "training_time_s": round(elapsed, 1),
                    "image_auroc": None,
                    "pixel_auroc": None,
                    "image_f1": None,
                    "status": f"failed: {str(e)[:100]}",
                }

            all_results.append(entry)

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Save results
    csv_path = output_dir / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n{'='*60}")
    print(f"  Benchmark complete! Results saved to {csv_path}")
    print(f"{'='*60}\n")

    # Print summary table
    print(df.to_string(index=False))

    return df


def run_benchmark_with_autoencoder(
    categories: list[str] | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Run benchmark including the custom autoencoder.

    Trains both Anomalib models and the custom autoencoder, then
    combines results into a single comparison table.

    Args:
        categories: MVTec AD categories to benchmark.
        output_dir: Output directory.

    Returns:
        Combined DataFrame with all model results.
    """
    from src.training.train_autoencoder import train_autoencoder

    if categories is None:
        categories = ["bottle", "carpet", "hazelnut"]

    if output_dir is None:
        output_dir = RESULTS_DIR / "benchmark"
    ensure_dir(output_dir)

    # Run Anomalib models
    anomalib_results = run_benchmark(
        models=["patchcore", "stfpm"],
        categories=categories,
        output_dir=output_dir,
    )

    # Run custom autoencoder
    ae_results = []
    for category in categories:
        print(f"\n{'━'*60}")
        print(f"  Benchmark: custom_autoencoder × {category}")
        print(f"{'━'*60}")

        start_time = time.time()
        try:
            result = train_autoencoder(category=category)
            elapsed = time.time() - start_time
            metrics = result.get("metrics", {})

            ae_results.append({
                "model": "custom_autoencoder",
                "category": category,
                "training_time_s": round(elapsed, 1),
                "image_auroc": metrics.get("image_auroc"),
                "pixel_auroc": None,  # Custom AE doesn't compute pixel AUROC by default
                "image_f1": metrics.get("image_f1"),
                "status": "success",
            })
        except Exception as e:
            elapsed = time.time() - start_time
            ae_results.append({
                "model": "custom_autoencoder",
                "category": category,
                "training_time_s": round(elapsed, 1),
                "image_auroc": None,
                "pixel_auroc": None,
                "image_f1": None,
                "status": f"failed: {str(e)[:100]}",
            })

    ae_df = pd.DataFrame(ae_results)
    combined = pd.concat([anomalib_results, ae_df], ignore_index=True)

    # Save combined results
    csv_path = output_dir / "benchmark_results_full.csv"
    combined.to_csv(csv_path, index=False)

    print(f"\n{'='*60}")
    print(f"  Full Benchmark Results")
    print(f"{'='*60}")
    print(combined.to_string(index=False))

    return combined


def load_benchmark_results(path: Path | None = None) -> pd.DataFrame:
    """Load previously saved benchmark results.

    Args:
        path: Path to CSV file. Defaults to results/benchmark/benchmark_results.csv.

    Returns:
        DataFrame with benchmark results.
    """
    if path is None:
        path = RESULTS_DIR / "benchmark" / "benchmark_results.csv"

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No benchmark results found at {path}. "
            "Run run_benchmark() first."
        )

    return pd.read_csv(path)
