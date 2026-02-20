"""
Data augmentation for training.

Applies random transformations to synthetic text images to improve
model robustness with lightweight perturbations:
- noise / blur / morphology
- brightness / contrast / illumination shifts
- slight rotation and perspective warp
- JPEG compression artifacts
"""

from __future__ import annotations

import io

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None


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

    # Apply each augmentation with some probability.
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

    if rng.random() < 0.25:
        img = add_illumination_gradient(img, rng)

    if rng.random() < 0.25:
        img = gamma_adjust(img, rng)

    if Image is not None and rng.random() < 0.25:
        img = random_rotate(img, rng, max_degrees=4.0)

    if Image is not None and rng.random() < 0.2:
        img = random_perspective(img, rng, max_warp_ratio=0.08)

    if Image is not None and rng.random() < 0.2:
        img = jpeg_compress(img, rng, quality_range=(35, 90))

    return np.clip(img, 0.0, 1.0).astype(np.float32, copy=False)


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


def add_illumination_gradient(
    img: np.ndarray,
    rng: np.random.Generator,
    strength_range: tuple[float, float] = (-0.18, 0.18),
) -> np.ndarray:
    """Add a smooth illumination gradient across the image."""
    h, w = img.shape
    start = rng.uniform(*strength_range)
    end = rng.uniform(*strength_range)
    if rng.random() < 0.5:
        grad = np.linspace(start, end, w, dtype=np.float32)[np.newaxis, :]
    else:
        grad = np.linspace(start, end, h, dtype=np.float32)[:, np.newaxis]
    return img + grad


def gamma_adjust(
    img: np.ndarray,
    rng: np.random.Generator,
    gamma_range: tuple[float, float] = (0.75, 1.35),
) -> np.ndarray:
    """Apply random gamma correction."""
    gamma = float(rng.uniform(*gamma_range))
    return np.power(np.clip(img, 0.0, 1.0), gamma)


def random_rotate(
    img: np.ndarray,
    rng: np.random.Generator,
    max_degrees: float = 4.0,
) -> np.ndarray:
    """Rotate by a small random angle."""
    if Image is None:
        return img
    angle = float(rng.uniform(-max_degrees, max_degrees))
    resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
    pil = Image.fromarray((np.clip(img, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
    out = pil.rotate(angle, resample=resample, fillcolor=255)
    return np.array(out, dtype=np.float32) / 255.0


def random_perspective(
    img: np.ndarray,
    rng: np.random.Generator,
    max_warp_ratio: float = 0.08,
) -> np.ndarray:
    """Apply a mild random perspective warp."""
    if Image is None:
        return img

    h, w = img.shape
    if h < 4 or w < 4:
        return img

    dx = max_warp_ratio * w
    dy = max_warp_ratio * h
    src = np.array(
        [
            [0.0, 0.0],
            [w - 1.0, 0.0],
            [w - 1.0, h - 1.0],
            [0.0, h - 1.0],
        ],
        dtype=np.float32,
    )
    dst = src + np.array(
        [
            [rng.uniform(-dx, dx), rng.uniform(-dy, dy)],
            [rng.uniform(-dx, dx), rng.uniform(-dy, dy)],
            [rng.uniform(-dx, dx), rng.uniform(-dy, dy)],
            [rng.uniform(-dx, dx), rng.uniform(-dy, dy)],
        ],
        dtype=np.float32,
    )
    coeffs = _find_perspective_coeffs(src, dst)
    resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR

    pil = Image.fromarray((np.clip(img, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
    warped = pil.transform(
        (w, h),
        Image.Transform.PERSPECTIVE,
        coeffs.tolist(),
        resample=resample,
        fillcolor=255,
    )
    return np.array(warped, dtype=np.float32) / 255.0


def jpeg_compress(
    img: np.ndarray,
    rng: np.random.Generator,
    quality_range: tuple[int, int] = (35, 90),
) -> np.ndarray:
    """Round-trip through JPEG to simulate compression artifacts."""
    if Image is None:
        return img
    q_low, q_high = quality_range
    quality = int(rng.integers(q_low, q_high + 1))
    pil = Image.fromarray((np.clip(img, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    decoded = Image.open(buf).convert("L")
    return np.array(decoded, dtype=np.float32) / 255.0


def _find_perspective_coeffs(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Compute Pillow perspective coeffs for mapping dst->src."""
    matrix = []
    target = []
    for (sx, sy), (dx, dy) in zip(src, dst, strict=True):
        matrix.append([dx, dy, 1.0, 0.0, 0.0, 0.0, -sx * dx, -sx * dy])
        matrix.append([0.0, 0.0, 0.0, dx, dy, 1.0, -sy * dx, -sy * dy])
        target.extend([sx, sy])
    a = np.asarray(matrix, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    coeffs, *_ = np.linalg.lstsq(a, b, rcond=None)
    return coeffs.astype(np.float32)
