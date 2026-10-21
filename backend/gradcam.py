"""
Grad-CAM explainability for LeafGuard AI.
Generates attention heatmaps for model predictions.
"""

import torch
import torch.nn as nn
from typing import Optional
from PIL import Image
import numpy as np
import logging

from preprocessing import preprocess_image, IMAGENET_MEAN, IMAGENET_STD


logger = logging.getLogger(__name__)

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    logging.warning("pytorch-grad-cam not installed. Grad-CAM endpoint will be disabled.")


def _find_last_conv_layer(model: nn.Module) -> nn.Module:
    """Find the last Conv2d layer in the model for Grad-CAM target."""
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("No Conv2d layer found in model for Grad-CAM")
    return last_conv


def generate_gradcam(
    model: nn.Module,
    image: Image.Image,
    predicted_label: int,
    device: torch.device,
    target_layer: Optional[nn.Module] = None
) -> Optional[np.ndarray]:
    """
    Generate Grad-CAM heatmap for an image.

    Args:
        model: PyTorch model
        image: PIL Image
        predicted_label: Predicted class label
        device: Device to run on
        target_layer: Target layer for Grad-CAM (auto-detected if None)

    Returns:
        Heatmap as numpy array or None if Grad-CAM unavailable
    """
    if not GRADCAM_AVAILABLE:
        return None

    try:
        # Preprocess image
        input_tensor = preprocess_image(image).to(device)

        # Find target layer if not specified
        if target_layer is None:
            target_layer = _find_last_conv_layer(model)

        # Create GradCAM
        cam = GradCAM(model=model, target_layers=[target_layer])

        # Create target
        targets = [ClassifierOutputTarget(predicted_label)]

        # Generate heatmap
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]  # Remove batch dimension

        return grayscale_cam

    except Exception as e:
        logger.error(f"Grad-CAM generation failed: {e}")
        return None


def overlay_heatmap(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4
) -> Image.Image:
    """
    Overlay Grad-CAM heatmap on original image.

    Args:
        image: Original PIL Image
        heatmap: Grad-CAM heatmap
        alpha: Transparency of heatmap overlay

    Returns:
        PIL Image with heatmap overlay
    """
    # Resize heatmap to match image size
    heatmap_resized = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
    heatmap_resized = np.array(heatmap_resized).astype(float) / 255

    # Apply colormap.
    # matplotlib >= 3.9 removed cm.get_cmap(); use the top-level registry instead.
    import matplotlib
    if hasattr(matplotlib, "colormaps"):
        colormap = matplotlib.colormaps["jet"]
    else:  # pragma: no cover - older matplotlib
        import matplotlib.cm as cm
        colormap = cm.get_cmap("jet")

    heatmap_colored = colormap(heatmap_resized)[:, :, :3]

    # Convert to PIL
    heatmap_image = Image.fromarray((heatmap_colored * 255).astype(np.uint8))

    # Overlay
    overlay = Image.blend(image, heatmap_image, alpha)

    return overlay
