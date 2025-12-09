# tools/prnu_patch_debugger.py
"""
Patch-based PRNU debugger that re-uses functions from prnu_utils.py.

Usage (from project root):
    python -m tools.prnu_patch_debugger /path/to/image.png --out analysis_out --patch 64 --stride 32 --top 12

This script is safe to add — it does not modify your main files.
"""

import os
import sys
import argparse
import numpy as np
import cv2
from matplotlib import pyplot as plt, cm

# Ensure project root is in path so we can import prnu_utils (works when run as module)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import your existing utilities (must exist in project root)
try:
    from prnu_utils import denoise_image, extract_residual, load_image  # use your functions
except Exception:
    # fallback: implement minimal loader/denoiser if import fails (shouldn't normally be needed)
    from skimage.restoration import denoise_wavelet
    def load_image(path):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.float32) / 255.0
    def denoise_image(img, method='wavelet'):
        den = np.zeros_like(img)
        for c in range(img.shape[2]):
            den[..., c] = denoise_wavelet(img[..., c], method='BayesShrink', mode='soft',
                                          rescale_sigma=True, channel_axis=None)
        return den
    def extract_residual(img, den):
        residual = img - den
        residual_gray = residual.mean(axis=2)
        hp = cv2.GaussianBlur(residual_gray, (0,0), sigmaX=1.0)
        return residual_gray - hp

from scipy import stats

def local_noise_brightness_corr(patch_img_gray, patch_residual):
    flat_b = patch_img_gray.ravel()
    flat_r = np.abs(patch_residual).ravel()
    if flat_r.size < 2 or np.allclose(flat_r, flat_r[0]) or np.allclose(flat_b, flat_b[0]):
        return 0.0
    r, _ = stats.pearsonr(flat_b, flat_r)
    return float(r)

def compute_patch_features(img, residual, patch_size=64, stride=32):
    H, W = residual.shape
    img_gray = 0.2989 * img[...,0] + 0.5870 * img[...,1] + 0.1140 * img[...,2]
    coords = []
    energies = []
    rs = []
    ys = list(range(0, H - patch_size + 1, stride))
    xs = list(range(0, W - patch_size + 1, stride))
    for y in ys:
        for x in xs:
            rp = residual[y:y+patch_size, x:x+patch_size]
            bp = img_gray[y:y+patch_size, x:x+patch_size]
            energy = float(np.std(rp))
            r = local_noise_brightness_corr(bp, rp)
            coords.append((y,x))
            energies.append(energy)
            rs.append(r)
    return np.array(coords), np.array(energies), np.array(rs), (ys, xs)

def build_dense_map(coords, values, image_shape, patch_size=64):
    H, W = image_shape[:2]
    accum = np.zeros((H,W), dtype=np.float32)
    count = np.zeros((H,W), dtype=np.float32)
    for val, (y,x) in zip(values, coords):
        accum[y:y+patch_size, x:x+patch_size] += float(val)
        count[y:y+patch_size, x:x+patch_size] += 1.0
    count = np.maximum(count, 1.0)
    return accum / count

def save_top_patches(img, coords, energies, rs, top_n, patch_size, out_dir, prefix="susp"):
    e_z = (energies - energies.mean()) / (energies.std() + 1e-12)
    r_z = (rs - rs.mean()) / (rs.std() + 1e-12)
    # suspicion score: lower zscore of energy and lower zscore of r -> suspicious
    score = (-e_z) + (-r_z)
    idx_sorted = np.argsort(-score)  # descending (most suspicious first)
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for rank, idx in enumerate(idx_sorted[:top_n]):
        y,x = coords[idx]
        crop = (img[y:y+patch_size, x:x+patch_size, :] * 255.0).astype(np.uint8)
        fname = os.path.join(out_dir, f"{prefix}_{rank:02d}_y{y}_x{x}_e{energies[idx]:.6f}_r{rs[idx]:.3f}.png")
        cv2.imwrite(fname, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        saved.append(fname)
    return saved

def visualize_and_save(img, residual, energy_map, r_map, comb_norm, overlay, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(14,8))
    plt.subplot(2,3,1); plt.title("Original (RGB)"); plt.imshow(img); plt.axis('off')
    plt.subplot(2,3,2); plt.title("Residual (grayscale)"); plt.imshow(residual, cmap='gray'); plt.colorbar(shrink=0.6); plt.axis('off')
    plt.subplot(2,3,3); plt.title("Patch energy map"); plt.imshow(energy_map, cmap='magma'); plt.colorbar(shrink=0.6); plt.axis('off')
    plt.subplot(2,3,4); plt.title("Patch noise-brightness r"); plt.imshow(r_map, cmap='RdBu', vmin=-1.0, vmax=1.0); plt.colorbar(shrink=0.6); plt.axis('off')
    plt.subplot(2,3,5); plt.title("Combined PRNU strength (norm)"); plt.imshow(comb_norm, cmap='inferno'); plt.colorbar(shrink=0.6); plt.axis('off')
    plt.subplot(2,3,6); plt.title("Overlay (suspicious bright)"); plt.imshow(img); plt.imshow(overlay, alpha=0.45); plt.axis('off')
    plt.tight_layout()
    out_png = os.path.join(out_dir, "analysis_overlay.png")
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    return out_png

def run_analysis(image_path: str, outdir: str, patch: int=64, stride: int=32, top: int=12, denoise_method='wavelet'):
    img = load_image(image_path)
    den = denoise_image(img, method=denoise_method)
    res = extract_residual(img, den)
    coords, energies, rs, _ = compute_patch_features(img, res, patch_size=patch, stride=stride)
    energy_map = build_dense_map(coords, energies, img.shape, patch_size=patch)
    r_map = build_dense_map(coords, rs, img.shape, patch_size=patch)
    # combine zscores into patch-level value and build dense map
    e_z = (energies - energies.mean()) / (energies.std() + 1e-12)
    r_z = (rs - rs.mean()) / (rs.std() + 1e-12)
    comb_patch = 0.5 * e_z + 0.5 * r_z
    comb_dense = build_dense_map(coords, comb_patch, img.shape, patch_size=patch)
    mn, mx = comb_dense.min(), comb_dense.max()
    comb_norm = (comb_dense - mn) / (mx - mn + 1e-12)
    overlay = cm.jet(1.0 - comb_norm)
    saved = save_top_patches(img, coords, energies, rs, top, patch, outdir, prefix="susp")
    vis = visualize_and_save(img, res, energy_map, r_map, comb_norm, overlay, outdir)
    stats = {
        "patch": patch, "stride": stride, "num_patches": len(coords),
        "energy_mean": float(energies.mean()), "energy_std": float(energies.std()),
        "r_mean": float(rs.mean()), "r_min": float(rs.min()), "r_max": float(rs.max())
    }
    return {"saved": saved, "overlay": vis, "stats": stats}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("image", help="path to image")
    p.add_argument("--out", default="analysis_out", help="output folder")
    p.add_argument("--patch", type=int, default=64, help="patch size")
    p.add_argument("--stride", type=int, default=32, help="stride")
    p.add_argument("--top", type=int, default=12, help="top suspicious patches to save")
    p.add_argument("--denoise", choices=('wavelet','nl_means'), default='wavelet')
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    result = run_analysis(args.image, args.out, patch=args.patch, stride=args.stride, top=args.top, denoise_method=args.denoise)
    print("Saved patches:", result["saved"])
    print("Overlay image:", result["overlay"])
    print("Stats:", result["stats"])
