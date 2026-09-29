"""
Baseline: plain MobileNetV3-Small (torchvision) for comparison.
No CBAM, no extra blocks. Fine-tuned head for 38 classes.
"""
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


def get_baseline_model(num_classes: int = 38, pretrained: bool = True) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    # replace classifier last layer (Linear)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def get_model_info(model: nn.Module):
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params": params, "trainable": trainable}


if __name__ == "__main__":
    import torch
    m = get_baseline_model(38, pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    y = m(x)
    assert y.shape == (1, 38)
    info = get_model_info(m)
    print(f"Baseline OK: {y.shape}, params={info['params']:,}")
