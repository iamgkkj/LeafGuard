"""
Basic tests for preprocessing module.
"""

from PIL import Image
import numpy as np
import torch
from preprocessing import preprocess_image, denormalize_tensor, IMAGE_SIZE


def test_preprocess_image():
    """Test image preprocessing."""
    # Create a simple RGB image
    img = Image.new('RGB', (300, 200), color='red')
    
    # Preprocess
    tensor = preprocess_image(img)
    
    # Check shape
    assert tensor.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)
    
    # Check it's normalized (values should be in reasonable range)
    assert tensor.min() >= -3  # ImageNet normalization range
    assert tensor.max() <= 3


def test_preprocess_grayscale():
    """Test that grayscale images are converted to RGB."""
    img = Image.new('L', (300, 200), color=128)
    tensor = preprocess_image(img)
    
    assert tensor.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)


def test_denormalize_tensor():
    """Test tensor denormalization."""
    # Create a normalized tensor
    tensor = torch.randn(1, 3, 224, 224)
    
    # Denormalize
    denorm = denormalize_tensor(tensor)
    
    # Check values are in [0, 1] range
    assert denorm.min() >= 0
    assert denorm.max() <= 1


if __name__ == '__main__':
    test_preprocess_image()
    test_preprocess_grayscale()
    test_denormalize_tensor()
    print("All preprocessing tests passed!")
