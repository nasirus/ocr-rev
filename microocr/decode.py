"""
Decode base64-encoded images into NumPy pixel arrays.

Supports PNG and JPEG without heavy image libraries by using
Python's built-in base64 + zlib for PNG, and optional Pillow
fallback for JPEG.
"""

from __future__ import annotations

import base64
import io
import struct
import zlib

import numpy as np


def decode_base64(b64_string: str) -> np.ndarray:
    """Decode a base64-encoded image into a HxW uint8 grayscale array.

    Supports PNG natively (pure Python) and JPEG via Pillow fallback.

    Args:
        b64_string: Base64-encoded image data. May include a
            ``data:image/...;base64,`` prefix which is stripped automatically.

    Returns:
        2-D NumPy array of shape (H, W) with dtype uint8 (grayscale).
    """
    # Strip optional data-URI prefix
    if "," in b64_string[:80]:
        b64_string = b64_string.split(",", 1)[1]

    raw = base64.b64decode(b64_string)

    # Detect format from magic bytes
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return _decode_png_gray(raw)

    if raw[:2] in (b"\xff\xd8",):
        return _decode_jpeg_gray(raw)

    # BMP magic
    if raw[:2] == b"BM":
        return _decode_with_pillow(raw)

    raise ValueError("Unsupported image format. Provide PNG or JPEG base64 data.")


# ---------------------------------------------------------------------------
# PNG decoder (pure Python + zlib) — no Pillow needed
# ---------------------------------------------------------------------------


def _decode_png_gray(data: bytes) -> np.ndarray:
    """Minimal PNG decoder that returns a grayscale HxW uint8 array."""
    pos = 8  # skip signature
    ihdr = None
    idat_chunks: list[bytes] = []

    while pos < len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        chunk_type = data[pos + 4 : pos + 8]
        chunk_data = data[pos + 8 : pos + 8 + length]
        pos += 12 + length  # 4 len + 4 type + data + 4 crc

        if chunk_type == b"IHDR":
            ihdr = _parse_ihdr(chunk_data)
        elif chunk_type == b"IDAT":
            idat_chunks.append(chunk_data)
        elif chunk_type == b"IEND":
            break

    if ihdr is None:
        raise ValueError("Invalid PNG: missing IHDR chunk")

    width, height, bit_depth, color_type = ihdr

    # Only handle 8-bit depth for simplicity
    if bit_depth != 8:
        # Fallback to Pillow for exotic bit depths
        return _decode_with_pillow(data)

    raw_pixels = zlib.decompress(b"".join(idat_chunks))

    # Determine bytes per pixel
    bpp = {0: 1, 2: 3, 4: 2, 6: 4}.get(color_type)
    if bpp is None:
        return _decode_with_pillow(data)

    stride = 1 + width * bpp  # 1 byte filter + pixel data per row
    rows = []
    prev_row = np.zeros(width * bpp, dtype=np.uint8)

    for y in range(height):
        row_start = y * stride
        filter_byte = raw_pixels[row_start]
        scanline = np.frombuffer(
            raw_pixels, dtype=np.uint8, count=width * bpp, offset=row_start + 1
        ).copy()
        scanline = _png_unfilter(filter_byte, scanline, prev_row, bpp)
        prev_row = scanline
        rows.append(scanline)

    img = np.stack(rows)  # (H, W*bpp)

    # Convert to grayscale
    if color_type == 0:
        # Already grayscale
        return img.reshape(height, width)
    elif color_type == 2:
        # RGB → gray  (0.299R + 0.587G + 0.114B)
        img = img.reshape(height, width, 3)
        return _rgb_to_gray(img)
    elif color_type == 4:
        # Gray + Alpha → drop alpha
        img = img.reshape(height, width, 2)
        return img[:, :, 0]
    elif color_type == 6:
        # RGBA → gray
        img = img.reshape(height, width, 4)
        return _rgb_to_gray(img[:, :, :3])

    return _decode_with_pillow(data)


def _parse_ihdr(data: bytes) -> tuple[int, int, int, int]:
    width = struct.unpack(">I", data[0:4])[0]
    height = struct.unpack(">I", data[4:8])[0]
    bit_depth = data[8]
    color_type = data[9]
    return width, height, bit_depth, color_type


def _png_unfilter(
    filter_type: int,
    scanline: np.ndarray,
    prev_row: np.ndarray,
    bpp: int,
) -> np.ndarray:
    """Apply PNG inverse filter to a scanline."""
    if filter_type == 0:
        return scanline
    out = scanline.astype(np.int16)
    if filter_type == 1:  # Sub
        for i in range(bpp, len(out)):
            out[i] = (out[i] + out[i - bpp]) % 256
    elif filter_type == 2:  # Up
        out = (out + prev_row.astype(np.int16)) % 256
    elif filter_type == 3:  # Average
        for i in range(len(out)):
            left = int(out[i - bpp]) if i >= bpp else 0
            up = int(prev_row[i])
            out[i] = (int(out[i]) + (left + up) // 2) % 256
    elif filter_type == 4:  # Paeth
        for i in range(len(out)):
            left = int(out[i - bpp]) if i >= bpp else 0
            up = int(prev_row[i])
            up_left = int(prev_row[i - bpp]) if i >= bpp else 0
            out[i] = (int(out[i]) + _paeth_predictor(left, up, up_left)) % 256
    return out.astype(np.uint8)


def _paeth_predictor(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    elif pb <= pc:
        return b
    return c


def _rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB array to grayscale using luminance weights."""
    return (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(
        np.uint8
    )


# ---------------------------------------------------------------------------
# JPEG / fallback via Pillow
# ---------------------------------------------------------------------------


def _decode_jpeg_gray(data: bytes) -> np.ndarray:
    return _decode_with_pillow(data)


def _decode_with_pillow(data: bytes) -> np.ndarray:
    """Fallback decoder using Pillow (required for JPEG)."""
    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "Pillow is required to decode JPEG images. Install with: pip install pillow"
        )
    img = Image.open(io.BytesIO(data)).convert("L")
    return np.array(img, dtype=np.uint8)
