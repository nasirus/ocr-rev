"""
Image preprocessing pipeline for MicroOCR.

Takes a grayscale image and produces a clean, normalized strip
ready for the recognition model:
    grayscale → binarize → deskew (optional) → normalize to fixed height
"""

from __future__ import annotations

import numpy as np

# Target height for model input (width is variable)
TARGET_HEIGHT = 32


def preprocess(gray: np.ndarray, target_height: int = TARGET_HEIGHT) -> np.ndarray:
    """Full preprocessing pipeline.

    Args:
        gray: 2-D uint8 grayscale image (H, W).
        target_height: Fixed height to normalize to.

    Returns:
        2-D float32 array of shape (target_height, W') normalized to [0, 1].
    """
    binary = binarize(gray)
    cropped = crop_to_content(binary)
    if cropped.size == 0:
        # Empty image — return a blank strip
        return np.zeros((target_height, target_height), dtype=np.float32)
    resized = resize_height(cropped, target_height)
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


def resize_height(img: np.ndarray, target_height: int) -> np.ndarray:
    """Resize image to a fixed height, preserving aspect ratio.

    Uses nearest-neighbor interpolation (fast, no dependencies).

    Args:
        img: 2-D uint8 array.
        target_height: Desired height in pixels.

    Returns:
        Resized 2-D uint8 array of shape (target_height, new_width).
    """
    h, w = img.shape
    if h == 0 or w == 0:
        return np.zeros((target_height, target_height), dtype=img.dtype)

    scale = target_height / h
    new_w = max(1, int(round(w * scale)))

    # Nearest-neighbor resize via index mapping
    row_idx = (np.arange(target_height) * h / target_height).astype(int)
    col_idx = (np.arange(new_w) * w / new_w).astype(int)
    row_idx = np.clip(row_idx, 0, h - 1)
    col_idx = np.clip(col_idx, 0, w - 1)

    return img[np.ix_(row_idx, col_idx)]
