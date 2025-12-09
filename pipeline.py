# pipeline.py
"""
Orchestrator pipeline for AI-image detection.
Expected to live next to:
- prnu_utils.py         (functions: extract_prnu(image) -> prnu_map, prnu_score(prnu_map) -> float)
- save_suspicious_patches.py (function: extract_suspicious_patches(image, out_dir=None) -> list of PIL.Image or paths)
- predict.py (optional) with function: predict_image(image) -> probability (0-100)
If predict.py is missing, a placeholder predictor is used so pipeline runs.
"""

import os
import sys
import json
import argparse
from PIL import Image
import numpy as np

# Try to import your tools — adjust names if needed.
try:
    import prnu_utils
except Exception:
    prnu_utils = None

try:
    import save_suspicious_patches as ssp
except Exception:
    ssp = None

# Try to import a real predictor
try:
    from predict import predict_image  # should accept PIL.Image or path, return probability 0-100
except Exception:
    predict_image = None

# ---------- Helper / fallback implementations ----------
def _placeholder_predict_image(img):
    """
    Fallback predictor until you have a trained model.
    Returns 50.0 for neutral, or uses a simple heuristic:
    - If image is mostly synthetic-looking (low entropy), raise prob slightly.
    This is intentionally simple and only for pipeline testing.
    """
    if isinstance(img, str):
        img = Image.open(img).convert("RGB")
    arr = np.array(img.convert("L"))
    entropy = _image_entropy(arr)
    # Normalize entropy roughly between 0..8 for 8-bit images
    # Lower entropy -> more likely AI (heuristic)
    prob = float(max(0, min(100, 60 - (entropy * 8))))  # just a weak heuristic
    return prob

def _image_entropy(arr):
    """Compute Shannon entropy of grayscale image array (simple)."""
    if arr.size == 0:
        return 0.0
    hist = np.bincount(arr.ravel(), minlength=256).astype(float)
    prob = hist / hist.sum()
    prob = prob[prob > 0]
    return -np.sum(prob * np.log2(prob))

# ---------- Pipeline core ----------
def analyze_image(input_path, aggregate_method="weighted", patch_weight=0.6, full_weight=0.4, temp_patch_dir="/tmp/suspicious_patches"):
    """
    Main pipeline.
    Args:
        input_path (str): path to image file
        aggregate_method (str): "weighted" or "avg"
        patch_weight (float): weight for patch-level score when using weighted aggregation
        full_weight (float): weight for full-image score when using weighted aggregation
        temp_patch_dir (str): where to save patches (if your patch extractor writes files)
    Returns:
        dict: { "ai_prob": float, "full_prob": float, "patch_probs": [...], "prnu_score": float }
    """
    # load image
    img = Image.open(input_path).convert("RGB")

    # 1) PRNU analysis (optional)
    prnu_score_val = None
    try:
        if prnu_utils is not None:
            # adjust function names to your prnu_utils
            prnu_map = prnu_utils.extract_prnu(img) if hasattr(prnu_utils, "extract_prnu") else None
            if prnu_map is not None and hasattr(prnu_utils, "prnu_score"):
                prnu_score_val = prnu_utils.prnu_score(prnu_map)
            elif prnu_map is not None:
                # fallback: use variance of prnu_map as simple score
                prnu_score_val = float(np.var(np.array(prnu_map)))
    except Exception as e:
        print(f"[warning] PRNU step failed: {e}", file=sys.stderr)
        prnu_score_val = None

    # 2) Suspicious patch extraction
    patch_images = []
    try:
        if ssp is not None and hasattr(ssp, "extract_suspicious_patches"):
            # prefer returning PIL.Image list; some implementations save files instead
            out = ssp.extract_suspicious_patches(img, out_dir=temp_patch_dir)
            # out may be list of paths or PIL.Image objects
            for o in out:
                if isinstance(o, str):
                    patch_images.append(Image.open(o).convert("RGB"))
                else:
                    patch_images.append(o.convert("RGB"))
        else:
            # fallback: create 4 center crops as "patches"
            w, h = img.size
            patch_size = (w // 4, h // 4)
            cx, cy = w // 2, h // 2
            left = max(0, cx - patch_size[0] // 2)
            upper = max(0, cy - patch_size[1] // 2)
            patch_images = [img.crop((left, upper, left + patch_size[0], upper + patch_size[1]))]
    except Exception as e:
        print(f"[warning] Patch extraction failed: {e}", file=sys.stderr)
        patch_images = []

    # 3) Full-image prediction
    try:
        if predict_image is not None:
            full_prob = float(predict_image(img))
        else:
            full_prob = float(_placeholder_predict_image(img))
    except Exception as e:
        print(f"[warning] Full-image prediction failed: {e}", file=sys.stderr)
        full_prob = float(_placeholder_predict_image(img))

    # 4) Patch predictions
    patch_probs = []
    for p_img in patch_images:
        try:
            if predict_image is not None:
                p_prob = float(predict_image(p_img))
            else:
                p_prob = float(_placeholder_predict_image(p_img))
        except Exception:
            p_prob = float(_placeholder_predict_image(p_img))
        patch_probs.append(p_prob)

    # 5) Aggregate probabilities
    if aggregate_method == "avg":
        all_probs = [full_prob] + patch_probs
        ai_prob = float(np.mean(all_probs)) if len(all_probs) > 0 else full_prob
    else:  # weighted: give patch-level and full-image weights
        patch_part = float(np.mean(patch_probs)) if patch_probs else full_prob
        ai_prob = float(full_weight * full_prob + patch_weight * patch_part)

    # normalize to 0..100
    ai_prob = max(0.0, min(100.0, ai_prob))

    result = {
        "ai_prob": ai_prob,
        "full_prob": full_prob,
        "patch_probs": patch_probs,
        "prnu_score": prnu_score_val
    }
    return result

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Run full pipeline: PRNU + patches + model prediction")
    parser.add_argument("--input", "-i", required=True, help="input image path")
    parser.add_argument("--aggregate", choices=["avg", "weighted"], default="weighted")
    parser.add_argument("--patch_weight", type=float, default=0.6)
    parser.add_argument("--full_weight", type=float, default=0.4)
    parser.add_argument("--out_json", "-o", default=None, help="save output JSON to file")
    args = parser.parse_args()

    res = analyze_image(args.input, aggregate_method=args.aggregate, patch_weight=args.patch_weight, full_weight=args.full_weight)
    pretty = json.dumps(res, indent=2)
    print(pretty)
    if args.out_json:
        with open(args.out_json, "w") as f:
            f.write(pretty)
        print(f"Wrote results to {args.out_json}")

if __name__ == "__main__":
    main()
