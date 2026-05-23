"""Custom folder-based dataset wrapper for anomaly detection.

Provides a convenient wrapper around Anomalib's Folder datamodule
for using your own images (e.g., photos of good/defective items).

Directory structure expected:
    root/
    ├── good/          # Normal training images
    │   ├── img001.png
    │   └── ...
    ├── abnormal/      # Anomalous test images (optional)
    │   ├── img002.png
    │   └── ...
    └── mask/          # Ground truth masks (optional, same names as abnormal)
        ├── img002.png
        └── ...
"""

from __future__ import annotations

from pathlib import Path

from anomalib.data import Folder
from torchvision.transforms.v2 import Resize


def get_custom_datamodule(
    root: str | Path,
    name: str = "custom_dataset",
    normal_dir: str = "good",
    abnormal_dir: str | None = "abnormal",
    mask_dir: str | None = "mask",
    image_size: tuple[int, int] = (256, 256),
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    num_workers: int = 4,
) -> Folder:
    """Create an Anomalib Folder datamodule for custom datasets.

    Args:
        root: Root directory containing the dataset folders.
        name: Dataset name (used in results directory naming).
        normal_dir: Subdirectory with normal (good) images.
        abnormal_dir: Subdirectory with anomalous images. None if no anomalies.
        mask_dir: Subdirectory with ground truth masks. None if no masks.
        image_size: Target image size as (height, width).
        train_batch_size: Batch size for training.
        eval_batch_size: Batch size for evaluation.
        num_workers: Number of data loading workers.

    Returns:
        Configured Folder datamodule.

    Raises:
        FileNotFoundError: If root or normal_dir doesn't exist.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    normal_path = root / normal_dir
    if not normal_path.exists():
        raise FileNotFoundError(
            f"Normal images directory not found: {normal_path}\n"
            f"Expected structure:\n"
            f"  {root}/\n"
            f"  ├── {normal_dir}/   ← Place normal images here\n"
            f"  ├── {abnormal_dir}/  ← Place anomalous images here\n"
            f"  └── {mask_dir}/      ← Place masks here (optional)"
        )

    # Check optional dirs
    abnormal_dir_arg = abnormal_dir if abnormal_dir and (root / abnormal_dir).exists() else None
    mask_dir_arg = mask_dir if mask_dir and (root / mask_dir).exists() else None

    # In Anomalib v2.4+, image resizing is handled via augmentations
    resize_transform = Resize(image_size)

    datamodule = Folder(
        name=name,
        root=str(root),
        normal_dir=normal_dir,
        abnormal_dir=abnormal_dir_arg,
        mask_dir=mask_dir_arg,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        augmentations=resize_transform,
    )

    return datamodule


def validate_dataset_structure(root: str | Path) -> dict[str, bool]:
    """Validate that a dataset directory has the expected structure.

    Args:
        root: Dataset root directory.

    Returns:
        Dictionary indicating which components are present.
    """
    root = Path(root)
    result = {
        "root_exists": root.exists(),
        "has_normal": (root / "good").exists(),
        "has_abnormal": (root / "abnormal").exists(),
        "has_masks": (root / "mask").exists(),
    }

    if result["has_normal"]:
        normal_images = list((root / "good").glob("*"))
        result["normal_count"] = len([
            f for f in normal_images
            if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        ])

    if result["has_abnormal"]:
        abnormal_images = list((root / "abnormal").glob("*"))
        result["abnormal_count"] = len([
            f for f in abnormal_images
            if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        ])

    return result
