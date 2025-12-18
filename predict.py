# predict.py
"""
Prediction utility that loads a checkpoint and exposes predict_image(image, ckpt_path)
Returns float probability in 0..100 (percentage AI-likelihood).

Usage:
from predict import predict_image
prob = predict_image("/path/to/img.jpg", ckpt_path="checkpoints/best.pth")
"""

import argparse
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models


def load_model(ckpt_path, device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict(image_path, model, device, img_size):
    tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
transforms.ToTensor(),
transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
    ])

    img = Image.open(image_path).convert("RGB")
    img = tfms(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)[0]

    real_prob = probs[0].item()
    ai_prob = probs[1].item()

    return real_prob, ai_prob


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.ckpt, device)

    real_p, ai_p = predict(args.image, model, device, args.img_size)

    print("\n===== PREDICTION RESULT =====")
    print(f"Real Image Probability : {real_p*100:.2f}%")
    print(f"AI Image Probability   : {ai_p*100:.2f}%")

    if ai_p > real_p:
        print("Prediction: AI Generated Image")
    else:
        print("Prediction: Real Image")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--ckpt", default="checkpoints/best_model.pth")
    parser.add_argument("--img_size", type=int, default=224)

    args = parser.parse_args()
    main(args)
