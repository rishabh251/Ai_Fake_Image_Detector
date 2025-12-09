"""
prnu_utils.py
First step: extract noise residual from images and compute simple PRNU cues.

Functions:
- load_image(path, as_gray=False)
- denoise_image(img, method='wavelet')
- extract_residual(img, denoised)
- residual_energy(residual) -> scalar
- noise_brightness_correlation(img, residual) -> scalar (pearson r)
- extract_patches(img, patch_size=128, stride=64)
- visualize_results(img, denoised, residual, show_fft=True)
- simple_pce(reference_residual, test_residual) -> correlation peak metric (optional)
"""

import numpy as np
import cv2
from skimage import img_as_float32
from skimage.restoration import denoise_wavelet
from scipy import fftpack, stats, signal
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_image(path, as_gray=False):
    img_bgr = cv2.imread(path, cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if not as_gray:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_as_float32(img_rgb)
    else:
        return img_as_float32(img_bgr)


def denoise_image(img, method='wavelet'):
    """
    Denoise image. Returns denoised image same dtype float32 range [0,1]
    Methods supported: 'wavelet', 'nl_means'
    """
    if img.ndim == 3:
        # process each channel separately
        den = np.zeros_like(img)
        for c in range(3):
            channel = img[..., c]
            if method == 'wavelet':
                den[..., c] = denoise_wavelet(channel, method='BayesShrink', mode='soft',
                                             rescale_sigma=True, channel_axis=-1)
            else:
                # convert to 8-bit temporarily for cv2 fastNlMeansDenoising
                ch8 = np.clip(channel * 255.0, 0, 255).astype(np.uint8)
                den8 = cv2.fastNlMeansDenoising(ch8, h=10)
                den[..., c] = den8.astype(np.float32) / 255.0
        return den
    else:
        if method == 'wavelet':
            return denoise_wavelet(img, method='BayesShrink', mode='soft', rescale_sigma=True, channel_axis=-1)
        else:
            ch8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            den8 = cv2.fastNlMeansDenoising(ch8, h=10)
            return den8.astype(np.float32) / 255.0


def extract_residual(img, denoised):
    """
    residual = original - denoised
    We also apply a mild high-pass to emphasize PRNU
    """
    residual = img - denoised
    # convert to grayscale residual magnitude if color
    if residual.ndim == 3:
        # average channels for residual analysis
        residual_gray = residual.mean(axis=2)
    else:
        residual_gray = residual
    # apply small high-pass (laplacian-ish) to emphasize sensor noise
    hp = cv2.GaussianBlur(residual_gray, (0, 0), sigmaX=1.0)
    residual_hp = residual_gray - hp
    return residual_hp


def residual_energy(residual):
    """Return energy / std of residual (higher for images with PRNU)"""
    return float(np.std(residual))


def noise_brightness_correlation(img, residual, patch_size=32):
    """
    Compute correlation between local brightness and local residual magnitude.
    For camera PRNU, brighter patches often have higher noise variance.
    Returns Pearson r.
    """
    if img.ndim == 3:
        gray = 0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]
    else:
        gray = img
    h, w = gray.shape
    mags = []
    br = []
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            patch_b = gray[y:y+patch_size, x:x+patch_size]
            patch_r = residual[y:y+patch_size, x:x+patch_size]
            br.append(np.mean(patch_b))
            mags.append(np.std(patch_r))
    if len(br) < 2:
        return 0.0
    r, p = stats.pearsonr(br, mags)
    return float(r)


def extract_patches(img, patch_size=128, stride=64):
    """
    Returns list of patches and their (y,x) coords
    """
    if img.ndim == 2:
        H, W = img.shape
    else:
        H, W, _ = img.shape
    patches = []
    coords = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            if img.ndim == 2:
                patch = img[y:y+patch_size, x:x+patch_size]
            else:
                patch = img[y:y+patch_size, x:x+patch_size, :]
            patches.append(patch.copy())
            coords.append((y, x))
    return patches, coords


def visualize_results(img, denoised, residual, show_fft=True):
    """Quick visualization for debugging"""
    plt.figure(figsize=(12,6))
    plt.subplot(2,3,1); plt.title("Original (RGB)"); plt.imshow(img); plt.axis('off')
    plt.subplot(2,3,2); plt.title("Denoised"); plt.imshow(np.clip(denoised,0,1)); plt.axis('off')
    plt.subplot(2,3,3); plt.title("Residual (grayscale)"); plt.imshow(residual, cmap='gray'); plt.colorbar(shrink=0.5); plt.axis('off')

    # residual histogram
    plt.subplot(2,3,4); plt.title("Residual histogram"); plt.hist(residual.ravel(), bins=256);

    if show_fft:
        # FFT magnitude of residual
        F = fftpack.fftshift(fftpack.fft2(residual))
        mag = np.log1p(np.abs(F))
        plt.subplot(2,3,5); plt.title("Residual FFT (log mag)"); plt.imshow(mag, cmap='inferno'); plt.axis('off')

        # show 1D radial average of spectrum
        center = np.array(mag.shape)//2
        y, x = np.indices(mag.shape)
        r = np.hypot(x-center[1], y-center[0]).astype(int)   # <--- use builtin int
        tbin = np.bincount(r.ravel(), mag.ravel())
        nr = np.bincount(r.ravel())
        radialprofile = tbin / (nr + 1e-8)
        plt.subplot(2,3,6); plt.title("Radial FFT profile"); plt.plot(radialprofile); plt.xlabel('radius'); plt.ylabel('avg log mag')

    plt.tight_layout()
    plt.show()


def simple_pce(reference_residual, test_residual):
    """
    A very simplified Peak-to-Correlation Energy (PCE) like measure:
    1. cross correlate test_residual with reference
    2. find peak and measure prominence vs background
    This is for camera-matching scenarios. If no reference, skip.
    """
    # zero-mean both
    ref = reference_residual - reference_residual.mean()
    test = test_residual - test_residual.mean()
    # normalized cross-correlation using FFT
    # pad to next power of two for speed
    size = [ref.shape[0] + test.shape[0], ref.shape[1] + test.shape[1]]
    corr = signal.fftconvolve(test, ref[::-1, ::-1], mode='valid')
    peak = np.max(np.abs(corr))
    avg = np.mean(np.abs(corr))
    std = np.std(corr)
    pce = (peak - avg) / (std + 1e-12)
    return float(pce)
