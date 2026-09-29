"""
MobileNetV3-Small with CBAM inserted after each InvertedResidual block.
Uses torchvision backbone (pretrained) and injects CBAM modules.
Target layers for Grad-CAM: features[-1] (last CBAM-augmented block) or features[12] etc.
"""
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models.mobilenetv3 import InvertedResidual
from .cbam import CBAM


class MobilenetV3CBAM(nn.Module):
    def __init__(self, num_classes: int = 38, pretrained: bool = True, reduction: int = 16):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)

        # Wrap features to insert CBAM after each InvertedResidual
        new_features = nn.Sequential()
        for idx, layer in enumerate(backbone.features):
            new_features.add_module(str(idx), layer)
            if isinstance(layer, InvertedResidual):
                # CBAM channels = layer's output channels
                # InvertedResidual stores out_channels; for Small variant we infer from layer
                # Use layer.conv's output or try to infer via layer.out_channels if exists
                out_ch = None
                # Typical InvertedResidual has attribute `out_channels` in torchvision >=0.15
                if hasattr(layer, "out_channels"):
                    out_ch = layer.out_channels
                else:
                    # fallback: try to get from last conv
                    try:
                        # layer.block is Sequential; last 1x1 projection conv
                        # find last Conv2d
                        for m in reversed(list(layer.modules())):
                            if isinstance(m, nn.Conv2d):
                                out_ch = m.out_channels
                                break
                    except Exception:
                        out_ch = 64
                if out_ch is None:
                    out_ch = 64
                new_features.add_module(f"cbam_{idx}", CBAM(out_ch, reduction=reduction))

        self.features = new_features
        self.avgpool = backbone.avgpool
        self.classifier = backbone.classifier
        # replace last linear for num_classes
        in_features = self.classifier[-1].in_features
        self.classifier[-1] = nn.Linear(in_features, num_classes)

        self.num_classes = num_classes

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def get_target_layers(self):
        """Return last CBAM-augmented conv for Grad-CAM.
        We return the last InvertedResidual block's conv.
        """
        # Find last InvertedResidual in features
        candidates = []
        for name, m in self.features.named_modules():
            if isinstance(m, InvertedResidual):
                candidates.append(m)
        if candidates:
            # last InvertedResidual's depthwise or project conv
            last = candidates[-1]
            # try to return its last conv layer
            for mod in reversed(list(last.modules())):
                if isinstance(mod, nn.Conv2d):
                    return [mod]
            return [last]
        # fallback: last feature conv
        for m in reversed(list(self.features.modules())):
            if isinstance(m, nn.Conv2d):
                return [m]
        return [self.features[-1]]


def get_cbam_model(num_classes: int = 38, pretrained: bool = True) -> MobilenetV3CBAM:
    return MobilenetV3CBAM(num_classes=num_classes, pretrained=pretrained)


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    import torch
    m = get_cbam_model(38, pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    y = m(x)
    assert y.shape == (1, 38), y.shape
    print(f"CBAM-MobileNet OK: {y.shape}, params={count_parameters(m):,}")
    tl = m.get_target_layers()
    print(f" Target layers: {tl}")
    # also test baseline interop
    from .baseline import get_baseline_model
    b = get_baseline_model(38, pretrained=False)
    y2 = b(x)
    assert y2.shape == (1, 38)
    print(" Baseline also OK")
