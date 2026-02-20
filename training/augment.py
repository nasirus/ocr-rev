"""
Data augmentation for training.

Applies random transformations to synthetic text images to improve
model robustness: noise, blur, erosion/dilation, slight rotation.
All implemented in pure NumPy (no OpenCV needed).
"""

from __future__ import annotations

import numpy as np


def augment(img: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Apply random augmentations to a grayscale image.

    Args:
        img: 2-D float32 array in [0, 1].
        rng: NumPy random generator for reproducibility.

    Returns:
        Augmented 2-D float32 array in [0, 1].
    """
    if rng is None:
        rng = np.random.default_rng()

    # Apply each augmentation with some probability
    if rng.random() < 0.5:
        img = add_gaussian_noise(img, rng, sigma_range=(0.01, 0.08))

    if rng.random() < 0.3:
        img = gaussian_blur(img, rng)

    if rng.random() < 0.3:
        img = adjust_brightness(img, rng, delta_range=(-0.15, 0.15))

    if rng.random() < 0.3:
        img = adjust_contrast(img, rng, factor_range=(0.7, 1.3))

    if rng.random() < 0.2:
        img = erode_or_dilate(img, rng)

    return np.clip(img, 0.0, 1.0)


def add_gaussian_noise(
    img: np.ndarray,
    rng: np.random.Generator,
    sigma_range: tuple[float, float] = (0.01, 0.05),
) -> np.ndarray:
    """Add Gaussian noise."""
    sigma = rng.uniform(*sigma_range)
    noise = rng.normal(0, sigma, size=img.shape).astype(np.float32)
    return img + noise


def gaussian_blur(
    img: np.ndarray,
    rng: np.random.Generator,
    kernel_size: int = 3,
) -> np.ndarray:
    """Simple box blur (approximates Gaussian for small kernels)."""
    # Use a simple mean filter as a lightweight blur
    k = kernel_size
    pad = k // 2
    padded = np.pad(img, pad, mode="edge")

    h, w = img.shape
    out = np.zeros_like(img)
    for dy in range(k):
        for dx in range(k):
            out += padded[dy : dy + h, dx : dx + w]
    return out / (k * k)


def adjust_brightness(
    img: np.ndarray,
    rng: np.random.Generator,
    delta_range: tuple[float, float] = (-0.1, 0.1),
) -> np.ndarray:
    """Randomly shift brightness."""
    delta = rng.uniform(*delta_range)
    return img + delta


def adjust_contrast(
    img: np.ndarray,
    rng: np.random.Generator,
    factor_range: tuple[float, float] = (0.8, 1.2),
) -> np.ndarray:
    """Randomly adjust contrast around the mean."""
    factor = rng.uniform(*factor_range)
    mean = img.mean()
    return (img - mean) * factor + mean


def erode_or_dilate(
    img: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Morphological erosion or dilation (makes text thinner or thicker).

    Operates on a binary-ish image by using min/max filters.
    """
    h, w = img.shape
    padded = np.pad(img, 1, mode="edge")

    if rng.random() < 0.5:
        # Erosion (darken / thicken text) — min filter
        out = np.minimum(
            np.minimum(padded[:-2, 1:-1], padded[2:, 1:-1]),
            np.minimum(padded[1:-1, :-2], padded[1:-1, 2:]),
        )
    else:
        # Dilation (lighten / thin text) — max filter
        out = np.maximum(
            np.maximum(padded[:-2, 1:-1], padded[2:, 1:-1]),
            np.maximum(padded[1:-1, :-2], padded[1:-1, 2:]),
        )
    return out
