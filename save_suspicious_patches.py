#!/usr/bin/env python3
"""
save_suspicious_patches.py
- Compute patch PRNU features (residual energy, noise-brightness Pearson r)
- Build a combined suspicion score per patch
- Save top-N suspicious patches (cropped images) to an output folder
- Save a visualization (overlay) and heatmap images

Usage:
    python save_suspicious_patches.py /path/to/image.jpg --out out_dir --patch 64 --stride 32 --top 12
"""

import os
import argparse
import numpy as np
import cv2
from skimage.restoration import denoise_wavelet
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import cm

def load_image(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0

def denoise_image(img, method='wavelet'):
    den = np.zeros_like(img)
    for c in range(img.shape[2]):
        channel = img[..., c]
        if method == 'wavelet':
            den[..., c] = denoise_wavelet(channel, method='BayesShrink', mode='soft',
                                          rescale_sigma=True, channel_axis=None)
        else:
            ch8 = np.clip(channel * 255.0, 0, 255).astype(np.uint8)
            den8 = cv2.fastNlMeansDenoising(ch8, h=10)
            den[..., c] = den8.astype(np.float32) / 255.0
    return den

def extract_residual(img, denoised):
    residual = img - denoised
    residual_gray = residual.mean(axis=2)
    hp = cv2.GaussianBlur(residual_gray, (0,0), sigmaX=1.0)
    residual_hp = residual_gray - hp
    return residual_hp

def local_noise_brightness_corr(patch_img_gray, patch_residual):
    flat_b = patch_img_gray.ravel()
    flat_r = np.abs(patch_residual).ravel()
    # degenerate case
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
    ys = list(range(0, H-patch_size+1, stride))
    xs = list(range(0, W-patch_size+1, stride))
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

def build_dense_maps(coords, values, image_shape, patch_size=64):
    H, W = image_shape[:2]
    accum = np.zeros((H,W), dtype=np.float32)
    count = np.zeros((H,W), dtype=np.float32)
    for val, (y,x) in zip(values, coords):
        accum[y:y+patch_size, x:x+patch_size] += float(val)
        count[y:y+patch_size, x:x+patch_size] += 1.0
    count = np.maximum(count, 1.0)
    return accum / count

def save_top_patches(img, coords, energies, rs, top_n, patch_size, out_dir, prefix="susp"):
    # compute a suspicion score: lower energy and lower r -> more suspicious
    # normalize energies and rs first (zscore)
    e_z = (energies - energies.mean()) / (energies.std() + 1e-12)
    r_z = (rs - rs.mean()) / (rs.std() + 1e-12)
    # combine: make lower r (negative) suspicious; lower energy suspicious
    # We'll compute score = -e_z + -r_z  (higher => more suspicious)
    score = (-e_z) + (-r_z)
    idx_sorted = np.argsort(-score)  # descending
    saved = []
    os.makedirs(out_dir, exist_ok=True)
    for rank, idx in enumerate(idx_sorted[:top_n]):
        y,x = coords[idx]
        crop = (img[y:y+patch_size, x:x+patch_size, :] * 255.0).astype(np.uint8)
        fname = os.path.join(out_dir, f"{prefix}_{rank:02d}_y{y}_x{x}_e{energies[idx]:.6f}_r{rs[idx]:.3f}.png")
        cv2.imwrite(fname, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        saved.append(fname)
    return saved

def visualize_and_save(img, residual, energy_map, r_map, comb_map, overlay, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(14,8))
    plt.subplot(2,3,1); plt.title("Original (RGB)"); plt.imshow(img); plt.axis('off')
    plt.subplot(2,3,2); plt.title("Residual (grayscale)"); plt.imshow(residual, cmap='gray'); plt.colorbar(shrink=0.6); plt.axis('off')
    plt.subplot(2,3,3); plt.title("Patch energy map"); plt.imshow(energy_map, cmap='magma'); plt.colorbar(shrink=0.6); plt.axis('off')
    plt.subplot(2,3,4); plt.title("Patch noise-brightness r"); plt.imshow(r_map, cmap='RdBu', vmin=-1.0, vmax=1.0); plt.colorbar(shrink=0.6); plt.axis('off')
    plt.subplot(2,3,5); plt.title("Combined PRNU strength (norm)"); plt.imshow(comb_map, cmap='inferno'); plt.colorbar(shrink=0.6); plt.axis('off')
    plt.subplot(2,3,6); plt.title("Overlay (suspicious bright)"); plt.imshow(img); plt.imshow(overlay, alpha=0.45); plt.axis('off')
    plt.tight_layout()
    out_png = os.path.join(out_dir, "analysis_overlay.png")
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.show()
    return out_png

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to image")
    parser.add_argument("--out", default="out_patches", help="Output folder")
    parser.add_argument("--patch", type=int, default=64, help="Patch size")
    parser.add_argument("--stride", type=int, default=32, help="Patch stride")
    parser.add_argument("--top", type=int, default=12, help="Top N suspicious patches to save")
    parser.add_argument("--denoise", choices=('wavelet','nl_means'), default='wavelet')
    args = parser.parse_args()

    IMAGE_PATH = args.image
    OUTDIR = args.out
    PATCH = args.patch
    STRIDE = args.stride
    TOPN = args.top

    print("Loading image:", IMAGE_PATH)
    img = load_image(IMAGE_PATH)
    den = denoise_image(img, method=args.denoise)
    res = extract_residual(img, den)

    coords, energies, rs, _ = compute_patch_features(img, res, patch_size=PATCH, stride=STRIDE)
    # build dense maps for visualization
    energy_map = build_dense_maps(coords, energies, img.shape, patch_size=PATCH)
    r_map = build_dense_maps(coords, rs, img.shape, patch_size=PATCH)
    # normalize patch-level combination to [0,1] (zscore based fusion)
    e_z = (energies - energies.mean()) / (energies.std() + 1e-12)
    r_z = (rs - rs.mean()) / (rs.std() + 1e-12)
    # comb zscore: high value = strong PRNU; we want suspicious = low PRNU, so invert later
    comb_patch = 0.5 * e_z + 0.5 * r_z
    # create dense comb map (put patch-level comb values back to image canvas)
    comb_dense = build_dense_maps(coords, comb_patch, img.shape, patch_size=PATCH)
    # normalize to [0,1]
    mn, mx = comb_dense.min(), comb_dense.max()
    comb_norm = (comb_dense - mn) / (mx - mn + 1e-12)
    # overlay suspicious = inverted comb_norm
    overlay = cm.jet(1.0 - comb_norm)  # RGBA
    # Save top patches
    saved = save_top_patches(img, coords, energies, rs, TOPN, PATCH, OUTDIR, prefix="susp")
    print("Saved top patches:", saved)
    # Visualization & save overlay
    vis_png = visualize_and_save(img, res, energy_map, r_map, comb_norm, overlay, OUTDIR)
    print("Saved visualization:", vis_png)
    # Print summary stats
    print(f"PATCH={PATCH} STRIDE={STRIDE} patches={len(coords)}")
    print(f"Energy: mean={energies.mean():.6f} std={energies.std():.6f}")
    print(f"r (pearson): mean={rs.mean():.4f} min={rs.min():.4f} max={rs.max():.4f}")

if __name__ == "__main__":
    main()
