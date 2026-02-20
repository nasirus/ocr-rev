"""
MicroOCR - Lightweight OCR: base64 image → text tokens.

Usage:
    import microocr
    text = microocr.read(base64_string)
"""

from microocr.inference import read, read_file

__version__ = "0.1.0"
__all__ = ["read", "read_file"]
