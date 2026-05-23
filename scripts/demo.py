"""Interactive Gradio demo for visual anomaly detection.

Provides a polished web interface for:
- Uploading images and detecting anomalies in real-time
- Selecting between different trained models
- Visualizing anomaly heatmaps overlaid on input images
- Adjusting detection threshold with a slider
- Viewing pre-loaded example images

Usage:
    python scripts/demo.py
    python scripts/demo.py --port 7861 --share
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr

from src.evaluation.visualize import overlay_anomaly_map
from src.models.autoencoder import AnomalyAutoencoder
from src.utils.config import RESULTS_DIR


# ── Globals ──────────────────────────────────────────────────
_loaded_models: dict[str, dict] = {}


def load_autoencoder_model(category: str = "bottle") -> dict | None:
    """Load a trained custom autoencoder model.

    Args:
        category: MVTec AD category the model was trained on.

    Returns:
        Dict with model, threshold, device. None if not found.
    """
    cache_key = f"autoencoder_{category}"
    if cache_key in _loaded_models:
        return _loaded_models[cache_key]

    ckpt_path = RESULTS_DIR / "custom_autoencoder" / category / "best_model.pt"
    if not ckpt_path.exists():
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    # Compute threshold from training data stats
    threshold_config = config.get("threshold", {})
    threshold = threshold_config.get("k_sigma", 3.0) * 0.01  # Approximate

    result = {"model": model, "threshold": threshold, "device": device}
    _loaded_models[cache_key] = result
    return result


def predict_anomaly(
    image: np.ndarray,
    model_name: str,
    category: str,
    threshold_multiplier: float,
) -> tuple[np.ndarray, str, float]:
    """Run anomaly detection on an uploaded image.

    Args:
        image: Input image from Gradio (H, W, 3) in [0, 255].
        model_name: Model to use.
        category: Category the model was trained on.
        threshold_multiplier: Multiplier for the base threshold.

    Returns:
        Tuple of (heatmap_overlay, verdict_text, anomaly_score).
    """
    if image is None:
        return None, "No image uploaded", 0.0

    if model_name == "Custom Autoencoder":
        return _predict_with_autoencoder(image, category, threshold_multiplier)
    else:
        return _predict_with_anomalib(image, model_name, category, threshold_multiplier)


def _predict_with_autoencoder(
    image: np.ndarray,
    category: str,
    threshold_multiplier: float,
) -> tuple[np.ndarray, str, float]:
    """Run prediction with the custom autoencoder."""
    model_data = load_autoencoder_model(category)
    if model_data is None:
        return image, f"⚠️ No trained autoencoder found for '{category}'", 0.0

    model = model_data["model"]
    device = model_data["device"]
    base_threshold = model_data["threshold"]

    # Preprocess: resize and normalize
    img_resized = cv2.resize(image, (256, 256))
    img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Get anomaly map and score
    with torch.no_grad():
        anomaly_map = model.compute_anomaly_map(img_tensor)
        anomaly_score = model.compute_anomaly_score(img_tensor)

    score = float(anomaly_score.cpu().item())
    amap = anomaly_map.squeeze().cpu().numpy()

    # Resize anomaly map to original image size
    amap_full = cv2.resize(amap, (image.shape[1], image.shape[0]))

    # Create overlay
    overlay = overlay_anomaly_map(image / 255.0, amap_full, alpha=0.4)

    # Determine verdict
    threshold = base_threshold * threshold_multiplier
    if score >= threshold:
        verdict = f"🔴 ANOMALY DETECTED\nScore: {score:.4f} (threshold: {threshold:.4f})"
    else:
        verdict = f"🟢 NORMAL\nScore: {score:.4f} (threshold: {threshold:.4f})"

    return overlay, verdict, score


def _predict_with_anomalib(
    image: np.ndarray,
    model_name: str,
    category: str,
    threshold_multiplier: float,
) -> tuple[np.ndarray, str, float]:
    """Run prediction with an Anomalib model (placeholder for trained models)."""
    # For Anomalib models, we'd need a checkpoint. This is a fallback.
    model_key = model_name.lower().replace(" ", "")
    ckpt_dir = RESULTS_DIR / model_key / category

    if not ckpt_dir.exists():
        return (
            image,
            f"⚠️ No trained {model_name} found for '{category}'\n"
            f"Train it first: python scripts/train.py --model {model_key} --category {category}",
            0.0,
        )

    return image, f"ℹ️ Anomalib inference — use the CLI for full results", 0.0


def find_available_models() -> list[str]:
    """Scan results directory for trained models."""
    available = []

    # Check custom autoencoder
    ae_dir = RESULTS_DIR / "custom_autoencoder"
    if ae_dir.exists() and any(ae_dir.iterdir()):
        available.append("Custom Autoencoder")

    # Check Anomalib models
    for model_name in ["patchcore", "efficientad", "stfpm", "padim"]:
        model_dir = RESULTS_DIR / model_name
        if model_dir.exists() and any(model_dir.iterdir()):
            available.append(model_name.title())

    if not available:
        available = ["Custom Autoencoder"]  # Default option

    return available


def find_available_categories(model_name: str) -> list[str]:
    """Find categories for which a model has been trained."""
    model_key = model_name.lower().replace(" ", "_")
    if "autoencoder" in model_key:
        model_key = "custom_autoencoder"

    model_dir = RESULTS_DIR / model_key
    if not model_dir.exists():
        return ["bottle"]  # Default

    categories = [d.name for d in model_dir.iterdir() if d.is_dir()]
    return categories if categories else ["bottle"]


def create_demo() -> gr.Blocks:
    """Create the Gradio demo interface."""
    available_models = find_available_models()

    with gr.Blocks(
        title="🔬 Visual Anomaly Detection",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="orange",
        ),
    ) as demo:
        gr.Markdown(
            """
            # 🔬 Visual Anomaly Detection
            ### Detect defects and anomalies in images using deep learning

            Upload an image to check for anomalies. The model will generate a heatmap
            showing suspected defect regions and provide an anomaly score.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # Input controls
                model_selector = gr.Dropdown(
                    choices=available_models,
                    value=available_models[0],
                    label="🤖 Model",
                    info="Select the anomaly detection model",
                )
                category_selector = gr.Dropdown(
                    choices=["bottle", "carpet", "hazelnut", "screw", "tile"],
                    value="bottle",
                    label="📂 Category",
                    info="MVTec AD category the model was trained on",
                )
                threshold_slider = gr.Slider(
                    minimum=0.1, maximum=5.0, value=1.0, step=0.1,
                    label="🎚️ Threshold Sensitivity",
                    info="Lower = more sensitive (more detections), Higher = stricter",
                )
                input_image = gr.Image(
                    label="📸 Upload Image",
                    type="numpy",
                )
                detect_btn = gr.Button(
                    "🔍 Detect Anomalies",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=1):
                # Output
                output_image = gr.Image(
                    label="🗺️ Anomaly Heatmap",
                    type="numpy",
                )
                verdict_text = gr.Textbox(
                    label="📋 Verdict",
                    lines=3,
                    interactive=False,
                )
                score_display = gr.Number(
                    label="📊 Anomaly Score",
                    interactive=False,
                )

        # Connect button
        detect_btn.click(
            fn=predict_anomaly,
            inputs=[input_image, model_selector, category_selector, threshold_slider],
            outputs=[output_image, verdict_text, score_display],
        )

        gr.Markdown(
            """
            ---
            ### ℹ️ How It Works
            - **Normal training**: Models learn what "normal" looks like from defect-free images
            - **Anomaly detection**: Regions that deviate from "normal" are highlighted as anomalies
            - **Heatmap**: Warmer colors (red/yellow) indicate higher anomaly probability
            - **Threshold**: Adjust to balance between catching all defects vs. reducing false alarms
            """
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch anomaly detection demo")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()

    demo = create_demo()
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
