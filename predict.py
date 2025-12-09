# predict.py
"""
Prediction utility that loads a checkpoint and exposes predict_image(image, ckpt_path)
Returns float probability in 0..100 (percentage AI-likelihood).

Usage:
from predict import predict_image
prob = predict_image("/path/to/img.jpg", ckpt_path="checkpoints/best.pth")
"""

import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from typing import Union, Optional
import os

# import model builder
from model import build_model, load_checkpoint

# Preprocessing transform: match ImageNet normalization & input size 224x224
def get_transforms(size: int = 224):
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])


class Predictor:
    def __init__(self, ckpt_path: Optional[str] = None, device: Optional[str] = None,
                 backbone: str = "resnet50", input_size: int = 224):
        """
        If ckpt_path is None, model is randomly-initialized (useful for testing).
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.input_size = input_size
        self.transforms = get_transforms(size=input_size)

        # instantiate model
        self.model = build_model(backbone=backbone, pretrained=False, num_classes=1)
        self.model.to(self.device)
        self.model.eval()

        self.ckpt_path = ckpt_path
        if ckpt_path and os.path.exists(ckpt_path):
            chk = load_checkpoint(ckpt_path, device=self.device)
            # support key names 'model_state' or 'state_dict'
            state = chk.get("model_state", chk.get("state_dict", chk))
            try:
                self.model.load_state_dict(state)
            except RuntimeError:
                # try non-strict loading if shapes mismatch
                self.model.load_state_dict(state, strict=False)

    @torch.no_grad()
    def predict(self, pil_img: Union[str, Image.Image]) -> float:
        """
        Returns probability in 0..100
        """
        if isinstance(pil_img, str):
            pil_img = Image.open(pil_img).convert("RGB")

        x = self.transforms(pil_img).unsqueeze(0).to(self.device)  # 1x3xHxW
        logits = self.model(x)  # shape: (1,) or (1,1)
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy().ravel()
        # handle shapes robustly
        val = float(np.atleast_1d(logits)[0])
        prob = 1.0 / (1.0 + np.exp(-val))  # sigmoid
        return float(prob * 100.0)


# convenience function (module-level)
_default_predictor = None
def predict_image(img: Union[str, Image.Image], ckpt_path: Optional[str] = None,
                  device: Optional[str] = None, backbone: str = "resnet50", input_size: int = 224) -> float:
    global _default_predictor
    if _default_predictor is None or (_default_predictor.ckpt_path != ckpt_path):
        _default_predictor = Predictor(ckpt_path=ckpt_path, device=device, backbone=backbone, input_size=input_size)
    return _default_predictor.predict(img)


# simple CLI
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", "-i", required=True, help="Image path")
    parser.add_argument("--ckpt", "-c", default=None, help="Checkpoint path (optional)")
    parser.add_argument("--backbone", default="resnet50", help="Backbone name")
    parser.add_argument("--size", type=int, default=224, help="Input size")
    args = parser.parse_args()

    prob = predict_image(args.image, ckpt_path=args.ckpt, backbone=args.backbone, input_size=args.size)
    print(f"AI probability: {prob:.2f}%")
