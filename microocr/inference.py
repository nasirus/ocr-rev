"""
MicroOCR inference engine — pure NumPy, zero framework dependency.

This module implements:
    1. NumPy-only CNN forward pass (matching MicroOCRModel architecture)
    2. End-to-end ``read()`` function: base64 → text

The model weights are loaded from a ``.npz`` file (exported from PyTorch
via ``export.py``).
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np

from microocr.ctc import BLANK_IDX, beam_decode, greedy_decode
from microocr.decode import decode_base64
from microocr.preprocess import TARGET_HEIGHT, binarize, preprocess
from microocr.segment import segment_lines

# Default model weights path
_DEFAULT_WEIGHTS = Path(__file__).parent / "weights" / "microocr.npz"

# Cached weights keyed by absolute path
_cached_weights: dict[Path, dict[str, np.ndarray]] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read(
    b64_string: str,
    weights_path: str | Path | None = None,
    *,
    decode_mode: Literal["greedy", "beam"] = "greedy",
    reject_blank: bool = True,
    reject_blank_ratio: float = 0.90,
    reject_nonblank_peak: float = 0.55,
    max_line_width: int | None = None,
    low_confidence_beam_fallback: bool = False,
    fallback_margin: float = 0.05,
    beam_width: int = 10,
) -> str:
    """Read text from a base64-encoded image.

    This is the main entry point for MicroOCR.

    Args:
        b64_string: Base64-encoded image (PNG or JPEG).
        weights_path: Path to ``.npz`` model weights. If None, uses
            the bundled default weights.
        decode_mode: ``"greedy"`` (fast) or ``"beam"`` (more exhaustive).
        reject_blank: If True, suppress line output when logits indicate
            mostly blank content.
        reject_blank_ratio: Minimum argmax-blank ratio to treat as blank line.
        reject_nonblank_peak: Maximum allowed nonblank probability peak when
            rejecting blank lines.
        max_line_width: Optional cap on preprocessed line width to reduce
            latency spikes on very long lines.
        low_confidence_beam_fallback: If True, run beam decode on lines where
            greedy confidence margin is low.
        fallback_margin: Mean top1-top2 probability margin threshold for
            greedy-to-beam fallback.
        beam_width: Beam width used by beam decoder.

    Returns:
        Recognized text string.

    Example::

        import microocr
        text = microocr.read("iVBORw0KGgoAAAANSUhEUg...")
    """
    text, _ = _read_impl(
        b64_string=b64_string,
        weights_path=weights_path,
        decode_mode=decode_mode,
        reject_blank=reject_blank,
        reject_blank_ratio=reject_blank_ratio,
        reject_nonblank_peak=reject_nonblank_peak,
        max_line_width=max_line_width,
        low_confidence_beam_fallback=low_confidence_beam_fallback,
        fallback_margin=fallback_margin,
        beam_width=beam_width,
        collect_timing=False,
    )
    return text


def read_file(
    filepath: str | Path,
    weights_path: str | Path | None = None,
    *,
    decode_mode: Literal["greedy", "beam"] = "greedy",
    reject_blank: bool = True,
    reject_blank_ratio: float = 0.90,
    reject_nonblank_peak: float = 0.55,
    max_line_width: int | None = None,
    low_confidence_beam_fallback: bool = False,
    fallback_margin: float = 0.05,
    beam_width: int = 10,
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
    return read(
        b64,
        weights_path=weights_path,
        decode_mode=decode_mode,
        reject_blank=reject_blank,
        reject_blank_ratio=reject_blank_ratio,
        reject_nonblank_peak=reject_nonblank_peak,
        max_line_width=max_line_width,
        low_confidence_beam_fallback=low_confidence_beam_fallback,
        fallback_margin=fallback_margin,
        beam_width=beam_width,
    )


def _read_with_timing(
    b64_string: str,
    weights_path: str | Path | None = None,
    *,
    decode_mode: Literal["greedy", "beam"] = "greedy",
    reject_blank: bool = True,
    reject_blank_ratio: float = 0.90,
    reject_nonblank_peak: float = 0.55,
    max_line_width: int | None = None,
    low_confidence_beam_fallback: bool = False,
    fallback_margin: float = 0.05,
    beam_width: int = 10,
) -> tuple[str, dict[str, float]]:
    """Internal helper that runs OCR and returns stage timings in ms."""
    text, timings = _read_impl(
        b64_string=b64_string,
        weights_path=weights_path,
        decode_mode=decode_mode,
        reject_blank=reject_blank,
        reject_blank_ratio=reject_blank_ratio,
        reject_nonblank_peak=reject_nonblank_peak,
        max_line_width=max_line_width,
        low_confidence_beam_fallback=low_confidence_beam_fallback,
        fallback_margin=fallback_margin,
        beam_width=beam_width,
        collect_timing=True,
    )
    assert timings is not None
    return text, timings


def _read_impl(
    b64_string: str,
    weights_path: str | Path | None,
    *,
    decode_mode: Literal["greedy", "beam"],
    reject_blank: bool,
    reject_blank_ratio: float,
    reject_nonblank_peak: float,
    max_line_width: int | None,
    low_confidence_beam_fallback: bool,
    fallback_margin: float,
    beam_width: int,
    collect_timing: bool,
) -> tuple[str, dict[str, float] | None]:
    if decode_mode not in ("greedy", "beam"):
        raise ValueError(f"Unsupported decode mode: {decode_mode}")
    if beam_width < 1:
        raise ValueError("beam_width must be >= 1")
    if max_line_width is not None and max_line_width < 1:
        raise ValueError("max_line_width must be >= 1")

    timings: dict[str, float] | None = None
    total_start = perf_counter()
    if collect_timing:
        timings = {
            "decode_ms": 0.0,
            "segment_ms": 0.0,
            "preprocess_ms": 0.0,
            "forward_ms": 0.0,
            "decode_ctc_ms": 0.0,
            "total_ms": 0.0,
        }

    weights = _load_weights(weights_path)

    # Decode base64 → grayscale pixels
    t = perf_counter()
    gray = decode_base64(b64_string)
    _accum_ms(timings, "decode_ms", t)

    # Segment into lines (single binarization pass)
    t = perf_counter()
    binary = binarize(gray)
    lines = segment_lines(binary)
    _accum_ms(timings, "segment_ms", t)

    # Recognize each line
    results: list[str] = []
    for line_img in lines:
        t = perf_counter()
        processed = preprocess(
            line_img,
            target_height=TARGET_HEIGHT,
            already_binary=True,
            resize_mode="bilinear",
        )
        if max_line_width is not None and processed.shape[1] > max_line_width:
            processed = _downsample_width(processed, max_line_width)
        _accum_ms(timings, "preprocess_ms", t)

        t = perf_counter()
        logits = _forward(processed, weights)
        _accum_ms(timings, "forward_ms", t)

        t = perf_counter()
        text = _decode_line(
            logits=logits,
            decode_mode=decode_mode,
            reject_blank=reject_blank,
            reject_blank_ratio=reject_blank_ratio,
            reject_nonblank_peak=reject_nonblank_peak,
            low_confidence_beam_fallback=low_confidence_beam_fallback,
            fallback_margin=fallback_margin,
            beam_width=beam_width,
        )
        _accum_ms(timings, "decode_ctc_ms", t)

        if text:
            results.append(text)

    out = "\n".join(results)
    if timings is not None:
        timings["total_ms"] = (perf_counter() - total_start) * 1000.0
    return out, timings


def _accum_ms(
    timings: dict[str, float] | None,
    key: str,
    start_time: float,
) -> None:
    if timings is not None:
        timings[key] += (perf_counter() - start_time) * 1000.0


def _decode_line(
    logits: np.ndarray,
    decode_mode: Literal["greedy", "beam"],
    reject_blank: bool,
    reject_blank_ratio: float,
    reject_nonblank_peak: float,
    low_confidence_beam_fallback: bool,
    fallback_margin: float,
    beam_width: int,
) -> str:
    if reject_blank:
        blank_ratio, nonblank_peak = _blank_line_stats(logits)
        if blank_ratio >= reject_blank_ratio and nonblank_peak < reject_nonblank_peak:
            return ""

    if decode_mode == "beam":
        return beam_decode(logits, beam_width=beam_width)

    text = greedy_decode(logits)
    if low_confidence_beam_fallback and _mean_top1_margin(logits) < fallback_margin:
        return beam_decode(logits, beam_width=beam_width)
    return text


def _blank_line_stats(logits: np.ndarray) -> tuple[float, float]:
    """Return (blank argmax ratio, max nonblank posterior peak)."""
    best = np.argmax(logits, axis=1)
    blank_ratio = float(np.mean(best == BLANK_IDX))

    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    nonblank_peak = float(probs[:, :BLANK_IDX].max()) if BLANK_IDX > 0 else 0.0
    return blank_ratio, nonblank_peak


def _mean_top1_margin(logits: np.ndarray) -> float:
    """Average top1-top2 posterior margin across timesteps."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    top2 = np.partition(probs, -2, axis=1)[:, -2:]
    margin = top2.max(axis=1) - top2.min(axis=1)
    return float(margin.mean())


def _downsample_width(img: np.ndarray, target_width: int) -> np.ndarray:
    """Downsample width only using linear interpolation."""
    h, w = img.shape
    if w <= target_width:
        return img
    if target_width == 1:
        return img[:, :1]

    x = np.linspace(0.0, w - 1.0, target_width, dtype=np.float32)
    x0 = np.floor(x).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    wx = x - x0
    out = img[:, x0] * (1.0 - wx)[np.newaxis, :] + img[:, x1] * wx[np.newaxis, :]
    return out.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


def _load_weights(path: str | Path | None = None) -> dict[str, np.ndarray]:
    """Load model weights from a .npz file, with caching."""
    if path is None:
        path = _DEFAULT_WEIGHTS

    path = Path(path).expanduser()

    if not path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {path}. "
            "Train a model first with: python -m training.train"
        )

    resolved = path.resolve()
    cached = _cached_weights.get(resolved)
    if cached is not None:
        return cached

    with np.load(str(resolved)) as data:
        weights = {k: data[k] for k in data.files}
    _cached_weights[resolved] = weights
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

    # Per-batch matrix multiply.
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
