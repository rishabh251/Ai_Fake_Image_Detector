# model.py
"""
Model utilities for AI vs Real image classifier using transfer learning (PyTorch).

Functions:
- build_model(backbone='resnet50', pretrained=True, num_classes=1, dropout=0.5)
- save_checkpoint(model, optimizer, epoch, path, extra=None)
- load_checkpoint(path, device=None) -> dict with keys: model, optimizer (maybe), epoch, extra

Model outputs raw logits. Use sigmoid(logit) for probability.
"""

import torch
import torch.nn as nn
from typing import Optional

# optional efficientnet import
try:
    from torchvision.models import resnet50, resnet18, resnet34, efficientnet_b0
    from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights
except Exception:
    # for older torchvision versions, import generic models
    from torchvision import models
    resnet50 = models.resnet50
    resnet18 = models.resnet18
    resnet34 = models.resnet34
    try:
        efficientnet_b0 = models.efficientnet_b0
    except Exception:
        efficientnet_b0 = None
    ResNet50_Weights = None
    EfficientNet_B0_Weights = None


def build_model(backbone: str = "resnet50",
                pretrained: bool = True,
                num_classes: int = 1,
                dropout: float = 0.5) -> torch.nn.Module:
    """
    Build a binary classifier using a pretrained backbone and a small head.
    Args:
        backbone: 'resnet50' | 'resnet34' | 'resnet18' | 'efficientnet_b0'
        pretrained: whether to use pretrained ImageNet weights
        num_classes: 1 for binary (logit)
        dropout: head dropout prob
    Returns:
        model: torch.nn.Module
    """
    backbone = backbone.lower()
    if backbone == "resnet50":
        if ResNet50_Weights is not None and pretrained:
            # newer torchvision weight enum (optional)
            model_base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            model_base = resnet50(pretrained=pretrained)
        in_features = model_base.fc.in_features
        # remove original fc
        model_base.fc = nn.Identity()
    elif backbone == "resnet34":
        model_base = resnet34(pretrained=pretrained)
        in_features = model_base.fc.in_features
        model_base.fc = nn.Identity()
    elif backbone == "resnet18":
        model_base = resnet18(pretrained=pretrained)
        in_features = model_base.fc.in_features
        model_base.fc = nn.Identity()
    elif backbone == "efficientnet_b0" and efficientnet_b0 is not None:
        if EfficientNet_B0_Weights is not None and pretrained:
            model_base = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            model_base = efficientnet_b0(pretrained=pretrained)
        # newer efficientnet structure uses classifier[1]
        try:
            in_features = model_base.classifier[1].in_features
            model_base.classifier = nn.Identity()
        except Exception:
            # fallback - try .classifier
            in_features = model_base.classifier.in_features
            model_base.classifier = nn.Identity()
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    # head for binary classification -> single logit
    head = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout * 0.5),
        nn.Linear(256, num_classes)
    )

    # Full model wrapper
    class Classifier(nn.Module):
        def __init__(self, base, head):
            super().__init__()
            self.backbone = base
            self.head = head

        def forward(self, x):
            feat = self.backbone(x)
            out = self.head(feat)
            return out.squeeze(1) if out.shape[-1] == 1 else out

    model = Classifier(model_base, head)
    return model


# ---------------- checkpoint helpers ----------------
def save_checkpoint(model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer],
                    epoch: int, path: str, extra: Optional[dict] = None):
    """
    Save model + optimizer + epoch + extra to path.
    """
    state = {
        "epoch": epoch,
        "model_state": model.state_dict()
    }
    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    if extra is not None:
        state["extra"] = extra
    torch.save(state, path)


def load_checkpoint(path: str, device: Optional[str] = None):
    """
    Load checkpoint and return the raw dict. Does not rebuild model.
    Use load_checkpoint then model.load_state_dict(checkpoint['model_state']).

    Args:
        path: checkpoint file path
        device: 'cpu' or 'cuda' or None
    Returns:
        checkpoint dict
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    chk = torch.load(path, map_location=device)
    return chk


# ---------------- quick local test helper ----------------
if __name__ == "__main__":
    # small smoke test to verify model builds
    m = build_model("resnet50", pretrained=False)
    print("Model built:", type(m))
    import torch
    x = torch.randn(2, 3, 224, 224)
    y = m(x)
    print("Output shape:", y.shape)
