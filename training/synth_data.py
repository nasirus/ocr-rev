"""
Synthetic training data generator.

Renders synthetic text lines (letters/digits/whitespace/special chars)
onto images using Pillow, producing (image, label) pairs for CTC
training. No external labeled datasets needed — infinite free data.
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

from microocr.model import CHARS
from microocr.preprocess import TARGET_HEIGHT, preprocess
from training.augment import augment

_ALNUM_CHARS = [c for c in CHARS if c.isalnum()]
_SPACE_CHAR = " "
_SPECIAL_CHARS = [c for c in CHARS if (not c.isalnum()) and c != _SPACE_CHAR]

# Rare characters that need frequency boosting
_RARE_CHARS = list("QZXqzxJKjk@#%")
# Common character pool (for mixing)
_COMMON_CHARS = list(CHARS)

# Pattern generators for diverse text
_CAMEL_WORDS = [
    "getData", "setName", "isValid", "hasError", "toString", "valueOf",
    "parseInt", "getItem", "onClick", "forEach", "indexOf", "toFixed",
    "charAt", "endsWith", "replace", "concat", "isEmpty", "toArray",
    "readFile", "sendMsg", "runTest", "logInfo", "mapKeys", "zipWith",
]
_SEPARATORS = [" ", "-", "_", "/", "."]
_TAIL_PUNCT = ["", "!", "?", ":", ";", "."]


def _generate_text(rng: np.random.Generator, min_len: int, max_len: int) -> str:
    """Generate diverse text patterns.

    Produces a mix of:
    - Pure random alphanumeric strings
    - CamelCase words
    - Alphanumeric codes (e.g., AB12cd)
    - Date-like strings
    - Multi-token text with spaces and separators
    - Strings with boosted rare characters
    """
    choice = rng.random()

    if choice < 0.45:
        # Standard random alphanumeric
        length = int(rng.integers(min_len, max_len + 1))
        return "".join(str(rng.choice(_ALNUM_CHARS)) for _ in range(length))

    elif choice < 0.60:
        # CamelCase words
        word = str(rng.choice(_CAMEL_WORDS))
        # Optionally prepend/append digits
        if rng.random() < 0.3:
            word = word + str(int(rng.integers(0, 100)))
        return word[:max_len]

    elif choice < 0.75:
        # Alphanumeric codes (e.g., AB12cd, X7y9Z)
        length = int(rng.integers(max(min_len, 4), min(max_len + 1, 12)))
        code = []
        for _ in range(length):
            pool_choice = rng.random()
            if pool_choice < 0.35:
                code.append(str(rng.choice(list(string.ascii_uppercase))))
            elif pool_choice < 0.65:
                code.append(str(rng.choice(list(string.digits))))
            else:
                code.append(str(rng.choice(list(string.ascii_lowercase))))
        return "".join(code)

    elif choice < 0.85:
        # Date-like digits only (keeps labels inside model alphabet).
        mm = int(rng.integers(1, 13))
        dd = int(rng.integers(1, 29))
        yyyy = int(rng.integers(1990, 2030))
        yy = yyyy % 100
        style = int(rng.integers(0, 3))
        if style == 0:
            return f"{mm:02d}{dd:02d}{yyyy:04d}"  # MMDDYYYY
        if style == 1:
            return f"{yyyy:04d}{mm:02d}{dd:02d}"  # YYYYMMDD
        return f"{dd:02d}{mm:02d}{yy:02d}"  # DDMMYY

    elif choice < 0.95:
        # Word-like strings with spaces/separators and optional punctuation.
        n_words = int(rng.integers(2, 5))
        parts = [str(rng.choice(_CAMEL_WORDS)) for _ in range(n_words)]
        sep = str(rng.choice(_SEPARATORS))
        text = sep.join(parts)
        text = text + str(rng.choice(_TAIL_PUNCT))
        if len(text) > max_len:
            text = text[:max_len]
        return text

    else:
        # Rare/special boosted strings
        length = int(rng.integers(min_len, max_len + 1))
        text = []
        for _ in range(length):
            p = rng.random()
            if p < 0.35:
                text.append(str(rng.choice(_RARE_CHARS)))
            elif p < 0.65 and _SPECIAL_CHARS:
                text.append(str(rng.choice(_SPECIAL_CHARS)))
            elif p < 0.75:
                text.append(_SPACE_CHAR)
            else:
                text.append(str(rng.choice(_COMMON_CHARS)))
        out = "".join(text).strip()
        return out or "0"


def generate_sample(
    rng: np.random.Generator | None = None,
    min_len: int = 1,
    max_len: int = 20,
    font_size_range: tuple[int, int] = (20, 40),
    target_height: int = TARGET_HEIGHT,
    apply_augment: bool = True,
    align_with_inference: bool = True,
) -> tuple[np.ndarray, str]:
    """Generate a single synthetic (image, label) pair.

    Args:
        rng: Random generator for reproducibility.
        min_len: Minimum text length.
        max_len: Maximum text length.
        font_size_range: Range of font sizes to sample from.
        target_height: Target image height after preprocessing.
        apply_augment: Whether to apply data augmentation.
        align_with_inference: If True, run the same crop/resize pipeline
            as runtime inference.

    Returns:
        Tuple of:
            - 2-D float32 array of shape (target_height, W), values in [0, 1]
            - Label string
    """
    if rng is None:
        rng = np.random.default_rng()

    # Generate diverse text
    label = _generate_text(rng, min_len, max_len)

    # Random font size
    font_size = int(rng.integers(*font_size_range))

    # Render text to image (with optional variable kerning)
    use_variable_kerning = rng.random() < 0.3
    if use_variable_kerning:
        img = _render_text_variable_kerning(label, font_size, rng)
    else:
        img = _render_text(label, font_size, rng)

    # Augment
    img_f = img.astype(np.float32) / 255.0
    if apply_augment:
        img_f = augment(img_f, rng)

    img_u8 = np.clip(img_f * 255.0, 0.0, 255.0).astype(np.uint8)
    if align_with_inference:
        img_out = preprocess(
            img_u8,
            target_height=target_height,
            already_binary=False,
            resize_mode="bilinear",
        )
    else:
        from microocr.preprocess import resize_height

        img_out = (
            resize_height(img_u8, target_height, mode="bilinear").astype(np.float32) / 255.0
        )

    return img_out, label


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


def _render_text_variable_kerning(
    text: str,
    font_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Render text character-by-character with random inter-character spacing.

    This simulates variable kerning found in real-world text.
    """
    font = _get_font(font_size, rng)
    tmp = Image.new("L", (1, 1), color=255)
    draw = ImageDraw.Draw(tmp)

    # Measure each character
    char_widths = []
    char_heights = []
    for ch in text:
        bbox = draw.textbbox((0, 0), ch, font=font)
        char_widths.append(bbox[2] - bbox[0])
        char_heights.append(bbox[3] - bbox[1])

    if not char_widths:
        return np.full((font_size + 4, 4), 255, dtype=np.uint8)

    max_h = max(char_heights) if char_heights else font_size

    # Compute total width with variable spacing
    spacings = [int(rng.integers(-1, 4)) for _ in range(max(0, len(text) - 1))]
    total_w = sum(char_widths) + sum(spacings)

    pad_x = int(rng.integers(2, 10))
    pad_y = int(rng.integers(2, 8))
    img_w = max(total_w + 2 * pad_x, 4)
    img_h = max_h + 2 * pad_y

    img = Image.new("L", (img_w, img_h), color=255)
    draw = ImageDraw.Draw(img)

    x_cursor = pad_x
    for i, ch in enumerate(text):
        bbox = draw.textbbox((0, 0), ch, font=font)
        draw.text((x_cursor - bbox[0], pad_y - bbox[1]), ch, fill=0, font=font)
        x_cursor += char_widths[i]
        if i < len(spacings):
            x_cursor += spacings[i]

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
