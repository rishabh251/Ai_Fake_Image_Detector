from prnu_utils import load_image, denoise_image, extract_residual, residual_energy, noise_brightness_correlation, visualize_results

img = load_image("my_photo2.jpg")                 # RGB float32 [0..1]
den = denoise_image(img, method='wavelet')       # or 'nl_means'
res = extract_residual(img, den)
energy = residual_energy(res)
nbcorr = noise_brightness_correlation(img, res)
print("Residual energy:", energy)
print("Noise-brightness Pearson r:", nbcorr)
visualize_results(img, den, res, show_fft=True)
