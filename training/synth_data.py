"""
Synthetic training data generator.

Renders random alphanumeric strings onto images using Pillow,
producing (image, label) pairs for CTC training. No external
labeled datasets needed — infinite free training data.
"""

from __future__ import annotations

import os
import string
import sys
from pathlib import Path

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print(
        "Pillow is required for synthetic data generation.\n"
        "Install with: pip install pillow",
        file=sys.stderr,
    )
    raise

from microocr.preprocess import TARGET_HEIGHT
from training.augment import augment


# Characters we generate
CHARS = string.ascii_letters + string.digits  # a-z A-Z 0-9


def generate_sample(
    rng: np.random.Generator | None = None,
    min_len: int = 1,
    max_len: int = 20,
    font_size_range: tuple[int, int] = (20, 40),
    target_height: int = TARGET_HEIGHT,
    apply_augment: bool = True,
) -> tuple[np.ndarray, str]:
    """Generate a single synthetic (image, label) pair.

    Args:
        rng: Random generator for reproducibility.
        min_len: Minimum text length.
        max_len: Maximum text length.
        font_size_range: Range of font sizes to sample from.
        target_height: Target image height after preprocessing.
        apply_augment: Whether to apply data augmentation.

    Returns:
        Tuple of:
            - 2-D float32 array of shape (target_height, W), values in [0, 1]
            - Label string
    """
    if rng is None:
        rng = np.random.default_rng()

    # Random text
    length = rng.integers(min_len, max_len + 1)
    label = "".join(rng.choice(list(CHARS)) for _ in range(length))

    # Random font size
    font_size = int(rng.integers(*font_size_range))

    # Render text to image
    img = _render_text(label, font_size, rng)

    # Convert to float32 [0, 1]
    img_f = img.astype(np.float32) / 255.0

    # Resize to target height
    from microocr.preprocess import resize_height

    img_f = (
        resize_height((img_f * 255).astype(np.uint8), target_height).astype(np.float32)
        / 255.0
    )

    # Augment
    if apply_augment:
        img_f = augment(img_f, rng)

    return img_f, label


def generate_batch(
    batch_size: int,
    rng: np.random.Generator | None = None,
    **kwargs,
) -> tuple[list[np.ndarray], list[str]]:
    """Generate a batch of synthetic samples.

    Args:
        batch_size: Number of samples to generate.
        rng: Random generator.
        **kwargs: Passed to :func:`generate_sample`.

    Returns:
        Tuple of (list of images, list of labels).
    """
    if rng is None:
        rng = np.random.default_rng()

    images = []
    labels = []
    for _ in range(batch_size):
        img, label = generate_sample(rng=rng, **kwargs)
        images.append(img)
        labels.append(label)

    return images, labels


def _render_text(
    text: str,
    font_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Render text to a grayscale numpy array using Pillow.

    Args:
        text: The string to render.
        font_size: Font size in pixels.
        rng: Random generator (for font selection, padding).

    Returns:
        2-D uint8 grayscale array.
    """
    font = _get_font(font_size, rng)

    # Create a temporary image to measure text size
    tmp = Image.new("L", (1, 1), color=255)
    draw = ImageDraw.Draw(tmp)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Add random padding
    pad_x = int(rng.integers(2, 10))
    pad_y = int(rng.integers(2, 8))
    img_w = text_w + 2 * pad_x
    img_h = text_h + 2 * pad_y

    # Render
    img = Image.new("L", (img_w, img_h), color=255)
    draw = ImageDraw.Draw(img)
    draw.text((pad_x - bbox[0], pad_y - bbox[1]), text, fill=0, font=font)

    return np.array(img, dtype=np.uint8)


# Cache discovered fonts
_font_cache: list[str] = []
_default_font_cache: ImageFont.FreeTypeFont | None = None


def _get_font(
    size: int, rng: np.random.Generator
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Get a random system font, or fall back to Pillow's default."""
    global _font_cache, _default_font_cache

    # Try to discover system fonts (first call only)
    if not _font_cache:
        _font_cache = _discover_fonts()

    if _font_cache:
        font_path = rng.choice(_font_cache)
        try:
            return ImageFont.truetype(font_path, size)
        except (OSError, IOError):
            pass

    # Fallback: Pillow's built-in default
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except (OSError, IOError):
        pass

    return ImageFont.load_default()


def _discover_fonts() -> list[str]:
    """Find TrueType fonts on the system."""
    font_dirs = [
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        os.path.expanduser("~/.fonts"),
        os.path.expanduser("~/.local/share/fonts"),
        # macOS
        "/System/Library/Fonts",
        "/Library/Fonts",
        os.path.expanduser("~/Library/Fonts"),
        # Windows
        r"C:\Windows\Fonts",
    ]

    fonts: list[str] = []
    for d in font_dirs:
        p = Path(d)
        if p.is_dir():
            for f in p.rglob("*.ttf"):
                fonts.append(str(f))
            for f in p.rglob("*.TTF"):
                fonts.append(str(f))
            for f in p.rglob("*.otf"):
                fonts.append(str(f))

    return fonts
