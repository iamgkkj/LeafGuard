"""
Inference module for LeafGuard AI.
Handles prediction logic for both models.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
import logging

from preprocessing import preprocess_image


logger = logging.getLogger(__name__)


def predict(
    model: nn.Module,
    image: Image.Image,
    class_names: List[str],
    device: torch.device,
    top_k: int = 3
) -> Dict:
    """
    Run inference on a single image.
    
    Args:
        model: PyTorch model
        image: PIL Image
        class_names: List of class names
        device: Device to run inference on
        top_k: Number of top predictions to return
        
    Returns:
        Dictionary with prediction results
    """
    # Preprocess
    input_tensor = preprocess_image(image).to(device)
    
    # Inference
    with torch.inference_mode():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, dim=0)
    
    # Get top-k predictions
    top_k_probs, top_k_indices = torch.topk(probabilities, min(top_k, len(class_names)))
    
    # Convert to lists
    top_k_probs = top_k_probs.cpu().numpy().tolist()
    top_k_indices = top_k_indices.cpu().numpy().tolist()
    
    top_predictions = [
        {
            "class": class_names[idx],
            "confidence": float(prob),
            "label": int(idx)
        }
        for prob, idx in zip(top_k_probs, top_k_indices)
    ]
    
    return {
        "predicted_class": class_names[predicted_idx.item()],
        "predicted_label": int(predicted_idx.item()),
        "confidence": float(confidence.item()),
        "top_predictions": top_predictions
    }


def compare_models(
    baseline_model: nn.Module,
    cbam_model: nn.Module,
    image: Image.Image,
    class_names: List[str],
    device: torch.device
) -> Dict:
    """
    Run both models on the same image and compare results.
    
    Args:
        baseline_model: Baseline model
        cbam_model: CBAM model
        image: PIL Image
        class_names: List of class names
        device: Device to run inference on
        
    Returns:
        Dictionary with comparison results
    """
    baseline_result = predict(baseline_model, image, class_names, device)
    cbam_result = predict(cbam_model, image, class_names, device)
    
    return {
        "baseline": baseline_result,
        "cbam": cbam_result,
        "agreement": baseline_result["predicted_class"] == cbam_result["predicted_class"]
    }
