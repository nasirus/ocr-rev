"""
Tests for the MicroOCR pipeline.

Tests each component independently and the end-to-end pipeline.
Run with: pytest tests/
"""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Test: decode.py
# ---------------------------------------------------------------------------


class TestDecode:
    """Tests for base64 image decoding."""

    def test_decode_png_grayscale(self):
        """Create a minimal grayscale PNG and decode it."""
        from microocr.decode import decode_base64

        img = self._make_gray_png(10, 20, fill=128)
        b64 = base64.b64encode(img).decode("ascii")
        result = decode_base64(b64)

        assert result.ndim == 2
        assert result.shape == (20, 10)  # width=10, height=20
        assert result.dtype == np.uint8

    def test_decode_png_rgb(self):
        """Create a minimal RGB PNG and verify it gets converted to grayscale."""
        from microocr.decode import decode_base64

        # Use Pillow to create a known RGB PNG
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        img = Image.new("RGB", (15, 10), color=(100, 150, 200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        result = decode_base64(b64)
        assert result.ndim == 2
        assert result.shape == (10, 15)

    def test_decode_strips_data_uri(self):
        """Ensure data:image/png;base64, prefix is stripped."""
        from microocr.decode import decode_base64

        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        img = Image.new("L", (5, 5), color=200)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        data_uri = f"data:image/png;base64,{b64}"

        result = decode_base64(data_uri)
        assert result.shape == (5, 5)

    def test_decode_invalid_format(self):
        """Unsupported format raises ValueError."""
        from microocr.decode import decode_base64

        garbage = base64.b64encode(b"not an image").decode("ascii")
        with pytest.raises(ValueError, match="Unsupported"):
            decode_base64(garbage)

    @staticmethod
    def _make_gray_png(width: int, height: int, fill: int = 128) -> bytes:
        """Create a minimal valid grayscale PNG in pure Python."""
        import struct
        import zlib

        def _chunk(chunk_type: bytes, data: bytes) -> bytes:
            crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
            return (
                struct.pack(">I", len(data))
                + chunk_type
                + data
                + struct.pack(">I", crc)
            )

        # IHDR
        ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
        # IDAT
        raw_rows = b""
        for _ in range(height):
            raw_rows += b"\x00" + bytes([fill] * width)  # filter=None
        compressed = zlib.compress(raw_rows)

        png = b"\x89PNG\r\n\x1a\n"
        png += _chunk(b"IHDR", ihdr_data)
        png += _chunk(b"IDAT", compressed)
        png += _chunk(b"IEND", b"")
        return png


# ---------------------------------------------------------------------------
# Test: preprocess.py
# ---------------------------------------------------------------------------


class TestPreprocess:
    """Tests for image preprocessing."""

    def test_binarize(self):
        """Binarization produces only 0 and 255 values."""
        from microocr.preprocess import binarize

        gray = np.random.randint(0, 256, size=(50, 100), dtype=np.uint8)
        binary = binarize(gray)

        assert binary.shape == gray.shape
        assert binary.dtype == np.uint8
        assert set(np.unique(binary)).issubset({0, 255})

    def test_crop_to_content(self):
        """Cropping removes blank borders."""
        from microocr.preprocess import crop_to_content

        # White image with a small dark rectangle
        img = np.full((100, 200), 255, dtype=np.uint8)
        img[30:60, 50:150] = 0  # ink block

        cropped = crop_to_content(img, margin=0)
        assert cropped.shape[0] == 30  # height of ink block
        assert cropped.shape[1] == 100  # width of ink block

    def test_resize_height(self):
        """Resize preserves aspect ratio."""
        from microocr.preprocess import resize_height

        img = np.zeros((64, 128), dtype=np.uint8)
        resized = resize_height(img, target_height=32)

        assert resized.shape[0] == 32
        assert resized.shape[1] == 64  # halved (aspect ratio preserved)

    def test_resize_height_bilinear_mode(self):
        """Bilinear mode runs and differs from nearest on nontrivial input."""
        from microocr.preprocess import resize_height

        img = np.tile(np.arange(9, dtype=np.uint8) * 28, (5, 1))
        nearest = resize_height(img, target_height=11, mode="nearest")
        bilinear = resize_height(img, target_height=11, mode="bilinear")

        assert nearest.shape == bilinear.shape
        assert nearest.dtype == np.uint8
        assert bilinear.dtype == np.uint8
        assert not np.array_equal(nearest, bilinear)

    def test_resize_height_invalid_mode(self):
        """Unsupported resize mode raises ValueError."""
        from microocr.preprocess import resize_height

        img = np.zeros((8, 12), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unsupported resize mode"):
            resize_height(img, target_height=16, mode="unknown")

    def test_preprocess_pipeline(self):
        """Full pipeline returns correct shape and dtype."""
        from microocr.preprocess import preprocess, TARGET_HEIGHT

        gray = np.random.randint(0, 256, size=(60, 200), dtype=np.uint8)
        result = preprocess(gray)

        assert result.shape[0] == TARGET_HEIGHT
        assert result.dtype == np.float32
        assert 0.0 <= result.min()
        assert result.max() <= 1.0

    def test_preprocess_empty_image(self):
        """All-white image returns a blank strip without crashing."""
        from microocr.preprocess import preprocess, TARGET_HEIGHT

        white = np.full((50, 100), 255, dtype=np.uint8)
        result = preprocess(white)

        assert result.shape[0] == TARGET_HEIGHT

    def test_preprocess_already_binary(self):
        """Preprocess supports skipping thresholding when input is already binary."""
        from microocr.preprocess import preprocess, TARGET_HEIGHT

        binary = np.full((50, 120), 255, dtype=np.uint8)
        binary[15:35, 20:100] = 0
        result = preprocess(binary, target_height=TARGET_HEIGHT, already_binary=True)

        assert result.shape[0] == TARGET_HEIGHT
        assert result.dtype == np.float32
        assert result.min() <= 0.05
        assert result.max() >= 0.95


# ---------------------------------------------------------------------------
# Test: segment.py
# ---------------------------------------------------------------------------


class TestSegment:
    """Tests for line segmentation."""

    def test_single_line(self):
        """Image with one text line returns one segment."""
        from microocr.segment import segment_lines

        img = np.full((50, 200), 255, dtype=np.uint8)
        img[15:35, 20:180] = 0  # one line of ink

        lines = segment_lines(img)
        assert len(lines) == 1

    def test_multiple_lines(self):
        """Image with three text lines returns three segments."""
        from microocr.segment import segment_lines

        img = np.full((150, 200), 255, dtype=np.uint8)
        img[10:30, 20:180] = 0  # line 1
        img[50:70, 20:180] = 0  # line 2
        img[100:120, 20:180] = 0  # line 3

        lines = segment_lines(img, min_line_height=8)
        assert len(lines) == 3

    def test_blank_image(self):
        """All-white image returns the full image as one line."""
        from microocr.segment import segment_lines

        img = np.full((50, 200), 255, dtype=np.uint8)
        lines = segment_lines(img)
        assert len(lines) == 1


# ---------------------------------------------------------------------------
# Test: ctc.py
# ---------------------------------------------------------------------------


class TestCTC:
    """Tests for CTC decoding."""

    def test_greedy_decode_simple(self):
        """Greedy decode collapses repeated chars and removes blanks."""
        from microocr.ctc import greedy_decode, BLANK_IDX

        # Simulate logits for "Hi" (H=33, i=8 in CHARS)
        # CHARS = abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789
        # H is at index 33, i is at index 8
        T = 8
        C = 63
        logits = np.full((T, C), -10.0, dtype=np.float32)

        # Timesteps: blank, H, H, blank, i, i, i, blank
        logits[0, BLANK_IDX] = 10.0
        logits[1, 33] = 10.0  # H
        logits[2, 33] = 10.0  # H (repeated, should collapse)
        logits[3, BLANK_IDX] = 10.0
        logits[4, 8] = 10.0  # i
        logits[5, 8] = 10.0  # i (repeated)
        logits[6, 8] = 10.0  # i (repeated)
        logits[7, BLANK_IDX] = 10.0

        result = greedy_decode(logits)
        assert result == "Hi"

    def test_greedy_decode_empty(self):
        """All-blank logits produce empty string."""
        from microocr.ctc import greedy_decode, BLANK_IDX

        logits = np.full((10, 63), -10.0, dtype=np.float32)
        logits[:, BLANK_IDX] = 10.0  # all blanks

        result = greedy_decode(logits)
        assert result == ""

    def test_greedy_decode_repeated_with_blank(self):
        """Blank between same chars produces two of that char."""
        from microocr.ctc import greedy_decode, BLANK_IDX

        # "aa" requires: a, blank, a
        T = 3
        C = 63
        logits = np.full((T, C), -10.0, dtype=np.float32)
        logits[0, 0] = 10.0  # a
        logits[1, BLANK_IDX] = 10.0  # blank
        logits[2, 0] = 10.0  # a

        result = greedy_decode(logits)
        assert result == "aa"

    def test_beam_decode_repeated_with_blank(self):
        """Beam decode handles repeated characters correctly."""
        from microocr.ctc import beam_decode, BLANK_IDX

        T = 8
        C = 63
        logits = np.full((T, C), -10.0, dtype=np.float32)
        logits[0, BLANK_IDX] = 10.0
        logits[1, 33] = 10.0  # H
        logits[2, 33] = 10.0  # H repeat
        logits[3, BLANK_IDX] = 10.0
        logits[4, 8] = 10.0  # i
        logits[5, 8] = 10.0
        logits[6, 8] = 10.0
        logits[7, BLANK_IDX] = 10.0

        result = beam_decode(logits, beam_width=5)
        assert result == "Hi"


# ---------------------------------------------------------------------------
# Test: model.py (requires PyTorch)
# ---------------------------------------------------------------------------


class TestModel:
    """Tests for the PyTorch model definition."""

    def test_model_output_shape(self):
        """Model output has correct shape (T, B, C)."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")

        from microocr.model import MicroOCRModel, NUM_CLASSES

        model = MicroOCRModel()
        x = torch.randn(2, 1, 32, 64)  # batch=2, width=64
        out = model(x)

        assert out.shape == (64 // 4, 2, NUM_CLASSES)  # T=16, B=2, C=63

    def test_model_parameter_count(self):
        """Model has roughly ~100K parameters."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")

        from microocr.model import MicroOCRModel

        model = MicroOCRModel()
        n_params = model.count_parameters()

        assert 50_000 < n_params < 200_000, f"Unexpected param count: {n_params}"


# ---------------------------------------------------------------------------
# Test: inference.py (NumPy forward pass)
# ---------------------------------------------------------------------------


class TestInference:
    """Tests for the pure NumPy inference engine."""

    def test_conv2d_shape(self):
        """Conv2d produces correct output shape."""
        from microocr.inference import _conv2d

        x = np.random.randn(1, 1, 32, 64).astype(np.float32)
        w = np.random.randn(16, 1, 3, 3).astype(np.float32)
        b = np.zeros(16, dtype=np.float32)

        out = _conv2d(x, w, b, padding=1)
        assert out.shape == (1, 16, 32, 64)

    def test_maxpool2d_shape(self):
        """MaxPool2d halves spatial dimensions."""
        from microocr.inference import _maxpool2d

        x = np.random.randn(1, 16, 32, 64).astype(np.float32)
        out = _maxpool2d(x, 2)
        assert out.shape == (1, 16, 16, 32)

    def test_numpy_matches_pytorch(self):
        """NumPy forward pass matches PyTorch forward pass."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")

        from microocr.model import MicroOCRModel
        from microocr.inference import _forward

        model = MicroOCRModel()
        model.eval()

        # Get weights as numpy
        state = model.state_dict()
        weights = {k: v.numpy() for k, v in state.items()}

        # Create test input
        rng = np.random.default_rng(42)
        img_np = rng.random((32, 64), dtype=np.float64).astype(np.float32)

        # NumPy forward
        np_out = _forward(img_np, weights)

        # PyTorch forward
        with torch.no_grad():
            pt_input = torch.tensor(img_np).unsqueeze(0).unsqueeze(0)
            pt_out = model(pt_input)  # (T, 1, C)
            pt_out = pt_out[:, 0, :].numpy()  # (T, C)

        # Compare
        assert np_out.shape == pt_out.shape, (
            f"Shape mismatch: numpy={np_out.shape} vs pytorch={pt_out.shape}"
        )
        np.testing.assert_allclose(np_out, pt_out, atol=1e-4, rtol=1e-4)

    def test_load_weights_cache_respects_path(self):
        """Loading one path does not mask missing/other paths."""
        from microocr import inference as inf

        inf._cached_weights.clear()
        _ = inf._load_weights("microocr/weights/microocr.npz")
        with pytest.raises(FileNotFoundError):
            inf._load_weights("does/not/exist.npz")

    def test_blank_image_rejected(self):
        """All-white image should decode as empty text by default."""
        import microocr

        img = TestDecode._make_gray_png(200, 80, fill=255)
        b64 = base64.b64encode(img).decode("ascii")
        result = microocr.read(b64, weights_path="microocr/weights/microocr.npz")
        assert result == ""

    def test_read_with_timing_reports_stages(self):
        """Internal timing helper returns expected stage keys."""
        from microocr.inference import _read_with_timing

        img = TestDecode._make_gray_png(64, 32, fill=255)
        b64 = base64.b64encode(img).decode("ascii")
        _, timing = _read_with_timing(b64, weights_path="microocr/weights/microocr.npz")
        for key in (
            "decode_ms",
            "segment_ms",
            "preprocess_ms",
            "forward_ms",
            "decode_ctc_ms",
            "total_ms",
        ):
            assert key in timing
            assert timing[key] >= 0.0


# ---------------------------------------------------------------------------
# Test: training/synth_data.py
# ---------------------------------------------------------------------------


class TestSynthData:
    """Tests for synthetic data generation."""

    def test_generate_sample(self):
        """Single sample has correct shape and label."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        from training.synth_data import generate_sample
        from microocr.preprocess import TARGET_HEIGHT

        rng = np.random.default_rng(123)
        img, label = generate_sample(rng=rng, apply_augment=False)

        assert img.ndim == 2
        assert img.shape[0] == TARGET_HEIGHT
        assert img.dtype == np.float32
        assert len(label) > 0
        assert all(c.isalnum() for c in label)

    def test_generate_batch(self):
        """Batch generation returns correct number of samples."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        from training.synth_data import generate_batch

        rng = np.random.default_rng(456)
        images, labels = generate_batch(5, rng=rng, apply_augment=False)

        assert len(images) == 5
        assert len(labels) == 5


# ---------------------------------------------------------------------------
# Test: augment.py
# ---------------------------------------------------------------------------


class TestAugment:
    """Tests for data augmentation."""

    def test_augment_preserves_shape(self):
        """Augmentation doesn't change image dimensions."""
        from training.augment import augment

        rng = np.random.default_rng(789)
        img = np.random.rand(32, 100).astype(np.float32)
        augmented = augment(img, rng)

        assert augmented.shape == img.shape
        assert augmented.dtype == np.float32

    def test_augment_clamps_values(self):
        """Augmented values stay in [0, 1]."""
        from training.augment import augment

        rng = np.random.default_rng(101)
        img = np.random.rand(32, 100).astype(np.float32)
        augmented = augment(img, rng)

        assert augmented.min() >= 0.0
        assert augmented.max() <= 1.0
