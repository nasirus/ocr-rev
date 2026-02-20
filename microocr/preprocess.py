"""
Image preprocessing pipeline for MicroOCR.

Takes a grayscale image and produces a clean, normalized strip
ready for the recognition model:
    grayscale → binarize → deskew (optional) → normalize to fixed height
"""

from __future__ import annotations

from typing import Literal

import numpy as np

# Target height for model input (width is variable)
TARGET_HEIGHT = 32


def preprocess(
    gray: np.ndarray,
    target_height: int = TARGET_HEIGHT,
    already_binary: bool = False,
    resize_mode: Literal["nearest", "bilinear"] = "bilinear",
) -> np.ndarray:
    """Full preprocessing pipeline.

    Args:
        gray: 2-D uint8 grayscale image (H, W).
        target_height: Fixed height to normalize to.
        already_binary: If True, skip thresholding and treat input as
            pre-binarized (0 ink, 255 background).
        resize_mode: Interpolation mode for resize. ``"bilinear"`` gives
            smoother glyph edges; ``"nearest"`` preserves hard binary edges.

    Returns:
        2-D float32 array of shape (target_height, W') normalized to [0, 1].
    """
    binary = gray if already_binary else binarize(gray)
    cropped = crop_to_content(binary)
    if cropped.size == 0:
        # Empty image — return a blank strip
        return np.zeros((target_height, target_height), dtype=np.float32)
    resized = resize_height(cropped, target_height, mode=resize_mode)
    normalized = resized.astype(np.float32) / 255.0
    return normalized


def binarize(gray: np.ndarray, block_size: int = 15, c: float = 10.0) -> np.ndarray:
    """Adaptive thresholding (simplified Sauvola-style).

    Uses a local mean computed via box filter. Pixels darker than
    (local_mean - c) are set to 0 (ink), others to 255 (background).

    Args:
        gray: 2-D uint8 array.
        block_size: Size of the local neighborhood (must be odd).
        c: Constant subtracted from the local mean.

    Returns:
        2-D uint8 binary image (0 or 255).
    """
    if gray.ndim != 2:
        raise ValueError(f"Expected 2-D grayscale image, got shape {gray.shape}")

    # Ensure odd block size
    if block_size % 2 == 0:
        block_size += 1

    h, w = gray.shape
    img = gray.astype(np.float64)

    # Compute local mean via integral image (cumulative sum with a
    # prepended zero row/column so indices never go out of bounds).
    pad = block_size // 2
    padded = np.pad(img, pad, mode="edge")
    # Prepend a zero row and column → shape (ph+1, pw+1)
    ph, pw = padded.shape
    integral = np.zeros((ph + 1, pw + 1), dtype=np.float64)
    integral[1:, 1:] = padded.cumsum(axis=0).cumsum(axis=1)

    # For each pixel (y, x) in the original image, the local window in
    # the padded image starts at (y, x) and ends at (y+block_size, x+block_size).
    # In the integral image (which is shifted by +1), the corners are:
    y1 = np.arange(h)
    y2 = y1 + block_size
    x1 = np.arange(w)
    x2 = x1 + block_size

    local_sum = (
        integral[np.ix_(y2, x2)]
        - integral[np.ix_(y1, x2)]
        - integral[np.ix_(y2, x1)]
        + integral[np.ix_(y1, x1)]
    )
    local_mean = local_sum / (block_size * block_size)

    # Threshold
    binary = np.where(img > local_mean - c, np.uint8(255), np.uint8(0))
    return binary.astype(np.uint8)


def crop_to_content(binary: np.ndarray, margin: int = 2) -> np.ndarray:
    """Crop a binary image to its ink bounding box plus a small margin.

    Args:
        binary: 2-D uint8 binary image (0=ink, 255=bg).
        margin: Pixels of padding to add around the bounding box.

    Returns:
        Cropped 2-D uint8 array.
    """
    # Ink pixels are 0
    ink_rows = np.where(np.any(binary < 128, axis=1))[0]
    ink_cols = np.where(np.any(binary < 128, axis=0))[0]

    if len(ink_rows) == 0 or len(ink_cols) == 0:
        return np.array([], dtype=np.uint8)

    y_min = max(0, ink_rows[0] - margin)
    y_max = min(binary.shape[0], ink_rows[-1] + 1 + margin)
    x_min = max(0, ink_cols[0] - margin)
    x_max = min(binary.shape[1], ink_cols[-1] + 1 + margin)

    return binary[y_min:y_max, x_min:x_max]


def resize_height(
    img: np.ndarray,
    target_height: int,
    mode: Literal["nearest", "bilinear"] = "nearest",
) -> np.ndarray:
    """Resize image to a fixed height, preserving aspect ratio.

    Uses nearest-neighbor or bilinear interpolation.

    Args:
        img: 2-D uint8 array.
        target_height: Desired height in pixels.
        mode: ``"nearest"`` or ``"bilinear"`` interpolation.

    Returns:
        Resized 2-D uint8 array of shape (target_height, new_width).
    """
    h, w = img.shape
    if h == 0 or w == 0:
        return np.zeros((target_height, target_height), dtype=img.dtype)

    scale = target_height / h
    new_w = max(1, int(round(w * scale)))

    if mode == "nearest":
        # Nearest-neighbor resize via index mapping
        row_idx = (np.arange(target_height) * h / target_height).astype(int)
        col_idx = (np.arange(new_w) * w / new_w).astype(int)
        row_idx = np.clip(row_idx, 0, h - 1)
        col_idx = np.clip(col_idx, 0, w - 1)
        return img[np.ix_(row_idx, col_idx)]

    if mode == "bilinear":
        return _resize_bilinear(img, target_height, new_w)

    raise ValueError(f"Unsupported resize mode: {mode}")


def _resize_bilinear(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear resize for 2-D arrays without external dependencies."""
    in_h, in_w = img.shape
    if in_h == 1 or in_w == 1:
        # Degenerate case: bilinear reduces to nearest.
        row_idx = (np.arange(out_h) * in_h / out_h).astype(int)
        col_idx = (np.arange(out_w) * in_w / out_w).astype(int)
        row_idx = np.clip(row_idx, 0, in_h - 1)
        col_idx = np.clip(col_idx, 0, in_w - 1)
        return img[np.ix_(row_idx, col_idx)]

    # Pixel-center mapping (matches common image library behavior).
    y = (np.arange(out_h, dtype=np.float32) + 0.5) * (in_h / out_h) - 0.5
    x = (np.arange(out_w, dtype=np.float32) + 0.5) * (in_w / out_w) - 0.5
    y = np.clip(y, 0.0, in_h - 1.0)
    x = np.clip(x, 0.0, in_w - 1.0)

    y0 = np.floor(y).astype(np.int32)
    x0 = np.floor(x).astype(np.int32)
    y1 = np.minimum(y0 + 1, in_h - 1)
    x1 = np.minimum(x0 + 1, in_w - 1)

    wy = (y - y0).reshape(-1, 1)
    wx = (x - x0).reshape(1, -1)

    top_left = img[np.ix_(y0, x0)].astype(np.float32)
    top_right = img[np.ix_(y0, x1)].astype(np.float32)
    bot_left = img[np.ix_(y1, x0)].astype(np.float32)
    bot_right = img[np.ix_(y1, x1)].astype(np.float32)

    top = top_left * (1.0 - wx) + top_right * wx
    bottom = bot_left * (1.0 - wx) + bot_right * wx
    out = top * (1.0 - wy) + bottom * wy
    return np.rint(out).astype(img.dtype)
