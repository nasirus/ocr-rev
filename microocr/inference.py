"""
MicroOCR inference engine — pure NumPy, zero framework dependency.

This module implements:
    1. NumPy-only CNN forward pass (matching MicroOCRModel architecture)
    2. End-to-end ``read()`` function: base64 → text

The model weights are loaded from a ``.npz`` file (exported from PyTorch
via ``export.py``).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from microocr.ctc import greedy_decode
from microocr.decode import decode_base64
from microocr.preprocess import preprocess, TARGET_HEIGHT
from microocr.segment import segment_lines

# Default model weights path
_DEFAULT_WEIGHTS = Path(__file__).parent / "weights" / "microocr.npz"

# Cached weights (loaded once)
_cached_weights: dict[str, np.ndarray] | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read(
    b64_string: str,
    weights_path: str | Path | None = None,
) -> str:
    """Read text from a base64-encoded image.

    This is the main entry point for MicroOCR.

    Args:
        b64_string: Base64-encoded image (PNG or JPEG).
        weights_path: Path to ``.npz`` model weights. If None, uses
            the bundled default weights.

    Returns:
        Recognized text string.

    Example::

        import microocr
        text = microocr.read("iVBORw0KGgoAAAANSUhEUg...")
    """
    weights = _load_weights(weights_path)

    # Decode base64 → grayscale pixels
    gray = decode_base64(b64_string)

    # Segment into lines
    from microocr.preprocess import binarize

    binary = binarize(gray)
    lines = segment_lines(binary)

    # Recognize each line
    results: list[str] = []
    for line_img in lines:
        processed = preprocess(line_img, target_height=TARGET_HEIGHT)
        logits = _forward(processed, weights)
        text = greedy_decode(logits)
        if text:
            results.append(text)

    return "\n".join(results)


def read_file(
    filepath: str | Path,
    weights_path: str | Path | None = None,
) -> str:
    """Read text from an image file.

    Convenience wrapper that loads a file, base64-encodes it,
    then calls :func:`read`.

    Args:
        filepath: Path to a PNG or JPEG image file.
        weights_path: Path to ``.npz`` model weights.

    Returns:
        Recognized text string.
    """
    import base64

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {filepath}")

    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return read(b64, weights_path=weights_path)


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


def _load_weights(path: str | Path | None = None) -> dict[str, np.ndarray]:
    """Load model weights from a .npz file, with caching."""
    global _cached_weights

    if path is None:
        path = _DEFAULT_WEIGHTS

    path = Path(path)

    if _cached_weights is not None:
        return _cached_weights

    if not path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {path}. "
            "Train a model first with: python -m training.train"
        )

    data = np.load(str(path))
    weights = {k: data[k] for k in data.files}
    _cached_weights = weights
    return weights


# ---------------------------------------------------------------------------
# Pure NumPy forward pass
# ---------------------------------------------------------------------------

# The architecture mirrors MicroOCRModel from model.py:
#   conv1(1→16, 3x3, pad=1) → relu → maxpool(2x2)
#   conv2(16→32, 3x3, pad=1) → relu → maxpool(2x2)
#   conv3(32→64, 3x3, pad=1) → relu
#   conv4(64→64, 3x3, pad=1) → relu
#   reshape → fc1(512→128) → relu → fc2(128→63)


def _forward(img: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
    """Run the CNN forward pass in pure NumPy.

    Args:
        img: 2-D float32 array (H, W) normalized to [0, 1].
        weights: Dictionary of weight arrays from .npz file.

    Returns:
        2-D array of shape (T, num_classes) — logits per timestep.
    """
    # Add batch and channel dims: (H, W) → (1, 1, H, W)
    x = img[np.newaxis, np.newaxis, :, :]

    # Conv block 1
    x = _conv2d(x, weights["conv1.weight"], weights["conv1.bias"], padding=1)
    x = _relu(x)
    x = _maxpool2d(x, 2)

    # Conv block 2
    x = _conv2d(x, weights["conv2.weight"], weights["conv2.bias"], padding=1)
    x = _relu(x)
    x = _maxpool2d(x, 2)

    # Conv block 3
    x = _conv2d(x, weights["conv3.weight"], weights["conv3.bias"], padding=1)
    x = _relu(x)

    # Conv block 4
    x = _conv2d(x, weights["conv4.weight"], weights["conv4.bias"], padding=1)
    x = _relu(x)

    # Reshape: (1, 64, 8, T) → (1, T, 512)
    b, c, h, w = x.shape
    x = np.transpose(x, (0, 3, 1, 2))  # (1, T, 64, 8)
    x = x.reshape(b, w, c * h)  # (1, T, 512)

    # FC1
    x = x @ weights["fc1.weight"].T + weights["fc1.bias"]
    x = _relu(x)

    # FC2
    x = x @ weights["fc2.weight"].T + weights["fc2.bias"]

    # Remove batch dim: (1, T, C) → (T, C)
    return x[0]


# ---------------------------------------------------------------------------
# NumPy ops (matching PyTorch semantics)
# ---------------------------------------------------------------------------


def _conv2d(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    padding: int = 0,
) -> np.ndarray:
    """2-D convolution using im2col for efficiency.

    Args:
        x: Input (B, C_in, H, W).
        weight: Kernel (C_out, C_in, kH, kW).
        bias: Bias (C_out,).
        padding: Zero-padding added to both sides.

    Returns:
        Output (B, C_out, H_out, W_out).
    """
    B, C_in, H, W = x.shape
    C_out, _, kH, kW = weight.shape

    if padding > 0:
        x = np.pad(
            x,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode="constant",
            constant_values=0,
        )
        H += 2 * padding
        W += 2 * padding

    H_out = H - kH + 1
    W_out = W - kW + 1

    # im2col: extract patches as columns
    cols = _im2col(x, kH, kW, H_out, W_out)  # (B, C_in*kH*kW, H_out*W_out)
    w_flat = weight.reshape(C_out, -1)  # (C_out, C_in*kH*kW)

    # Matrix multiply
    out = w_flat @ cols  # (B, C_out, H_out*W_out) — broadcast over batch via loop
    # Actually: cols is (B, patch, spatial), we need per-batch matmul
    out = np.einsum("ij,bjk->bik", w_flat, cols)  # (B, C_out, H_out*W_out)
    out = out.reshape(B, C_out, H_out, W_out)
    out += bias[np.newaxis, :, np.newaxis, np.newaxis]
    return out


def _im2col(x: np.ndarray, kH: int, kW: int, H_out: int, W_out: int) -> np.ndarray:
    """Extract sliding local blocks (im2col).

    Produces a column matrix matching PyTorch's weight layout where
    the kernel weight is stored as (C_out, C_in, kH, kW). Each column
    must therefore be ordered as [C_in, kH, kW] flattened.

    Args:
        x: (B, C, H, W).

    Returns:
        (B, C*kH*kW, H_out*W_out).
    """
    B, C, H, W = x.shape
    strides = x.strides  # (stride_B, stride_C, stride_H, stride_W)

    # Build a 6-D view: (B, C, kH, kW, H_out, W_out)
    # so that reshaping to (B, C*kH*kW, H_out*W_out) gives the correct
    # interleaving that matches PyTorch's (C_out, C_in, kH, kW) weight layout.
    shape = (B, C, kH, kW, H_out, W_out)
    strides_6d = (
        strides[0],  # B
        strides[1],  # C
        strides[2],  # kH (within kernel, step by 1 row)
        strides[3],  # kW (within kernel, step by 1 col)
        strides[2],  # H_out (step by 1 row)
        strides[3],  # W_out (step by 1 col)
    )

    patches = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides_6d)
    # Reshape: (B, C, kH, kW, H_out, W_out) → (B, C*kH*kW, H_out*W_out)
    cols = np.ascontiguousarray(patches).reshape(B, C * kH * kW, H_out * W_out)
    return cols


def _maxpool2d(x: np.ndarray, pool_size: int) -> np.ndarray:
    """2x2 max pooling with stride = pool_size.

    Args:
        x: (B, C, H, W).

    Returns:
        (B, C, H//pool_size, W//pool_size).
    """
    B, C, H, W = x.shape
    pH = H // pool_size
    pW = W // pool_size

    # Trim to exact multiple
    x = x[:, :, : pH * pool_size, : pW * pool_size]

    # Reshape and take max
    x = x.reshape(B, C, pH, pool_size, pW, pool_size)
    return x.max(axis=(3, 5))


def _relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(x, 0)
