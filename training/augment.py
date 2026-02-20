"""
Data augmentation for training.

Applies random transformations to synthetic text images to improve
model robustness with lightweight perturbations:
- noise / blur / morphology
- brightness / contrast / illumination shifts
- slight rotation and perspective warp
- JPEG compression artifacts
- random erasing / cutout
- salt-and-pepper noise
- elastic distortion
- paper texture and stains (document realism)
- horizontal line artifacts (underlines, strikethroughs, scan lines)
- scanner border shadows
- variable background textures (Perlin-style noise)
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

    # ── Document-realism augmentations (applied early) ────────────────
    if rng.random() < 0.15:
        img = paper_texture(img, rng)

    if rng.random() < 0.06:
        img = add_stains(img, rng)

    if rng.random() < 0.08:
        img = scanner_shadow(img, rng)

    if rng.random() < 0.08:
        img = line_artifact(img, rng)

    # ── Standard augmentations ────────────────────────────────────────
    if rng.random() < 0.45:
        img = add_gaussian_noise(img, rng, sigma_range=(0.005, 0.05))

    if rng.random() < 0.2:
        img = gaussian_blur(img, rng)

    if rng.random() < 0.22:
        img = adjust_brightness(img, rng, delta_range=(-0.15, 0.15))

    if rng.random() < 0.22:
        img = adjust_contrast(img, rng, factor_range=(0.7, 1.3))

    if rng.random() < 0.14:
        img = erode_or_dilate(img, rng)

    if rng.random() < 0.16:
        img = add_illumination_gradient(img, rng)

    if rng.random() < 0.16:
        img = gamma_adjust(img, rng)

    if Image is not None and rng.random() < 0.22:
        img = random_rotate(img, rng, max_degrees=2.5)

    if Image is not None and rng.random() < 0.15:
        img = random_perspective(img, rng, max_warp_ratio=0.05)

    if Image is not None and rng.random() < 0.15:
        img = jpeg_compress(img, rng, quality_range=(50, 95))

    if rng.random() < 0.10:
        img = random_erasing(img, rng)

    if rng.random() < 0.08:
        img = salt_and_pepper(img, rng, amount=0.004)

    if rng.random() < 0.08:
        img = elastic_distortion(img, rng, alpha=1.5, sigma=0.8)

    return np.clip(img, 0.0, 1.0).astype(np.float32, copy=False)


# ── Standard augmentations ────────────────────────────────────────────


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
    resample = (
        Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
    )
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
    resample = (
        Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
    )

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


def random_erasing(
    img: np.ndarray,
    rng: np.random.Generator,
    area_range: tuple[float, float] = (0.01, 0.05),
    fill: float | None = None,
) -> np.ndarray:
    """Random erasing / cutout augmentation.

    Erases a small random rectangular region, filled with near-white.
    """
    h, w = img.shape
    area = h * w
    erase_area = rng.uniform(*area_range) * area
    aspect_ratio = rng.uniform(0.3, 3.3)

    eh = int(np.sqrt(erase_area * aspect_ratio))
    ew = int(np.sqrt(erase_area / aspect_ratio))
    eh = min(eh, h - 1)
    ew = min(ew, w - 1)
    if eh < 1 or ew < 1:
        return img

    y = int(rng.integers(0, h - eh))
    x = int(rng.integers(0, w - ew))

    out = img.copy()
    if fill is None:
        # Fill with near-white (0.9 - 1.0)
        fill_val = rng.uniform(0.9, 1.0)
    else:
        fill_val = fill
    out[y : y + eh, x : x + ew] = fill_val
    return out


def salt_and_pepper(
    img: np.ndarray,
    rng: np.random.Generator,
    amount: float = 0.01,
) -> np.ndarray:
    """Add salt-and-pepper noise."""
    out = img.copy()
    n_pixels = img.size
    n_salt = int(amount * n_pixels / 2)
    n_pepper = int(amount * n_pixels / 2)

    # Salt (white)
    if n_salt > 0:
        coords = (
            rng.integers(0, img.shape[0], size=n_salt),
            rng.integers(0, img.shape[1], size=n_salt),
        )
        out[coords] = 1.0

    # Pepper (black)
    if n_pepper > 0:
        coords = (
            rng.integers(0, img.shape[0], size=n_pepper),
            rng.integers(0, img.shape[1], size=n_pepper),
        )
        out[coords] = 0.0

    return out


def elastic_distortion(
    img: np.ndarray,
    rng: np.random.Generator,
    alpha: float = 3.0,
    sigma: float = 0.5,
) -> np.ndarray:
    """Apply elastic distortion with small displacement fields.

    Uses random displacement fields smoothed by a simple box filter.
    """
    h, w = img.shape
    if h < 4 or w < 4:
        return img

    # Generate random displacement fields
    dx = rng.uniform(-1.0, 1.0, size=(h, w)).astype(np.float32)
    dy = rng.uniform(-1.0, 1.0, size=(h, w)).astype(np.float32)

    # Smooth with box filter (approximating Gaussian)
    k = max(3, int(sigma * 6) | 1)  # ensure odd
    dx = _box_filter_2d(dx, k) * alpha
    dy = _box_filter_2d(dy, k) * alpha

    # Create coordinate grids
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = np.clip(x_coords + dx, 0, w - 1)
    map_y = np.clip(y_coords + dy, 0, h - 1)

    # Bilinear interpolation
    x0 = np.floor(map_x).astype(np.int32)
    y0 = np.floor(map_y).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)

    wx = map_x - x0
    wy = map_y - y0

    out = (
        img[y0, x0] * (1 - wx) * (1 - wy)
        + img[y0, x1] * wx * (1 - wy)
        + img[y1, x0] * (1 - wx) * wy
        + img[y1, x1] * wx * wy
    )
    return out.astype(np.float32, copy=False)


# ── Document-realism augmentations ────────────────────────────────────


def paper_texture(
    img: np.ndarray,
    rng: np.random.Generator,
    strength_range: tuple[float, float] = (0.02, 0.08),
) -> np.ndarray:
    """Add subtle paper-like texture noise to simulate scanned paper grain.

    Uses multi-scale noise to mimic the fibrous texture of real paper.
    """
    h, w = img.shape
    strength = rng.uniform(*strength_range)

    # Fine grain (pixel-level)
    fine = rng.normal(0, 1, size=(h, w)).astype(np.float32)

    # Coarse grain (downsampled then upscaled) — simulates paper fiber
    coarse_h = max(2, h // 4)
    coarse_w = max(2, w // 4)
    coarse_small = rng.normal(0, 1, size=(coarse_h, coarse_w)).astype(np.float32)

    # Nearest-neighbor upscale
    row_idx = np.clip((np.arange(h) * coarse_h / h).astype(int), 0, coarse_h - 1)
    col_idx = np.clip((np.arange(w) * coarse_w / w).astype(int), 0, coarse_w - 1)
    coarse = coarse_small[np.ix_(row_idx, col_idx)]

    # Blend fine + coarse
    texture = 0.6 * fine + 0.4 * coarse
    texture = texture * strength

    return img + texture


def add_stains(
    img: np.ndarray,
    rng: np.random.Generator,
    n_stains_range: tuple[int, int] = (1, 4),
) -> np.ndarray:
    """Add subtle circular stain marks that simulate coffee stains, aging spots.

    Stains are light enough to not destroy text but add realism.
    """
    h, w = img.shape
    out = img.copy()
    n_stains = int(rng.integers(*n_stains_range))

    for _ in range(n_stains):
        # Random center
        cy = int(rng.integers(0, h))
        cx = int(rng.integers(0, w))

        # Random radius (small relative to image)
        max_radius = max(3, min(h, w) // 4)
        radius = int(rng.integers(2, max_radius + 1))

        # Intensity of the stain (darkening effect, subtle)
        intensity = rng.uniform(-0.08, -0.02)

        # Create circular mask with smooth falloff
        yy, xx = np.ogrid[0:h, 0:w]
        dist_sq = (yy - cy) ** 2 + (xx - cx) ** 2
        radius_sq = radius * radius

        # Smooth gaussian-like falloff
        mask = np.exp(-dist_sq.astype(np.float32) / (2.0 * radius_sq))
        out = out + mask * intensity

    return out


def scanner_shadow(
    img: np.ndarray,
    rng: np.random.Generator,
    shadow_width_range: tuple[float, float] = (0.05, 0.15),
    shadow_strength_range: tuple[float, float] = (-0.12, -0.03),
) -> np.ndarray:
    """Add dark shadow bands along edges to simulate scanner artifacts.

    Real scanners often produce dark shadows along the edges of the page,
    especially near the spine of bound documents.
    """
    h, w = img.shape
    out = img.copy()

    # Pick 1-2 edges to shadow
    edges = []
    if rng.random() < 0.5:
        edges.append("left")
    if rng.random() < 0.5:
        edges.append("right")
    if rng.random() < 0.3:
        edges.append("top")
    if rng.random() < 0.3:
        edges.append("bottom")

    if not edges:
        edges.append("left" if rng.random() < 0.5 else "right")

    for edge in edges:
        strength = rng.uniform(*shadow_strength_range)
        rel_width = rng.uniform(*shadow_width_range)

        if edge in ("left", "right"):
            shadow_w = max(1, int(w * rel_width))
            grad = np.linspace(strength, 0.0, shadow_w, dtype=np.float32)
            if edge == "left":
                out[:, :shadow_w] += grad[np.newaxis, :]
            else:
                out[:, -shadow_w:] += grad[np.newaxis, ::-1]
        else:
            shadow_h = max(1, int(h * rel_width))
            grad = np.linspace(strength, 0.0, shadow_h, dtype=np.float32)
            if edge == "top":
                out[:shadow_h, :] += grad[:, np.newaxis]
            else:
                out[-shadow_h:, :] += grad[::-1, np.newaxis]

    return out


def line_artifact(
    img: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add horizontal line artifacts simulating underlines, strikethroughs,
    scan lines, or fold creases.

    These are common in real scanned documents:
    - Underlines beneath text
    - Strikethroughs across text
    - Scanner feeder lines (thin horizontal streaks)
    - Paper fold creases
    """
    h, w = img.shape
    out = img.copy()

    style = int(rng.integers(0, 4))

    if style == 0:
        # Underline: dark line near the bottom third
        y = int(rng.integers(h * 2 // 3, h))
        thickness = int(rng.integers(1, 3))
        darkness = rng.uniform(0.0, 0.3)
        y_end = min(h, y + thickness)
        # Span most of the width
        x_start = int(rng.integers(0, max(1, w // 8)))
        x_end = w - int(rng.integers(0, max(1, w // 8)))
        out[y:y_end, x_start:x_end] = darkness

    elif style == 1:
        # Strikethrough: line through the middle
        y = int(rng.integers(h // 3, 2 * h // 3))
        thickness = int(rng.integers(1, 3))
        darkness = rng.uniform(0.0, 0.25)
        y_end = min(h, y + thickness)
        x_start = int(rng.integers(0, max(1, w // 6)))
        x_end = w - int(rng.integers(0, max(1, w // 6)))
        out[y:y_end, x_start:x_end] = darkness

    elif style == 2:
        # Scan line: thin faint horizontal streak across full width
        n_lines = int(rng.integers(1, 4))
        for _ in range(n_lines):
            y = int(rng.integers(0, h))
            faintness = rng.uniform(-0.08, -0.02)
            out[y, :] += faintness

    else:
        # Fold crease: subtle brightness change across a horizontal band
        y = int(rng.integers(h // 4, 3 * h // 4))
        crease_h = int(rng.integers(1, max(2, h // 6)))
        y_end = min(h, y + crease_h)
        # Crease slightly darkens the band
        crease_strength = rng.uniform(-0.06, -0.01)
        grad = np.abs(np.linspace(-1, 1, crease_h, dtype=np.float32))
        # V-shaped profile (darkest in center)
        profile = (1.0 - grad) * crease_strength
        out[y:y_end, :] += profile[:, np.newaxis]

    return out


# ── Helper functions ──────────────────────────────────────────────────


def _box_filter_2d(img: np.ndarray, k: int) -> np.ndarray:
    """Apply a box filter (mean filter) of size k."""
    pad = k // 2
    padded = np.pad(img, pad, mode="reflect")
    h, w = img.shape
    out = np.zeros_like(img)
    for dy in range(k):
        for dx in range(k):
            out += padded[dy : dy + h, dx : dx + w]
    return out / (k * k)


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
