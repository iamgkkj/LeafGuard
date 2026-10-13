"""
Image preprocessing for LeafGuard AI.
Matches the evaluation pipeline from the research notebook.
"""

from PIL import Image
import torch
from torchvision import transforms


# ImageNet normalization constants (from notebook)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


# Evaluation transform (exact match to notebook)
eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess a PIL image for model inference.
    
    Args:
        image: PIL Image in RGB format
        
    Returns:
        Preprocessed tensor with shape (1, 3, 224, 224)
    """
    # Ensure RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transform
    tensor = eval_transform(image)
    
   # Add batch dimension
    tensor = tensor.unsqueeze(0)
    
    return tensor


def denormalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize a tensor for visualization.
    
    Args:
        tensor: Normalized tensor
        
    Returns:
        Denormalized tensor in [0, 1] range
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)
