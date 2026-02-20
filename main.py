"""MicroOCR — Lightweight OCR: base64 image → text tokens."""

import microocr


def main():
    print(f"MicroOCR v{microocr.__version__}")
    print("Usage: microocr.read(base64_string) → text")
    print()
    print("To train a model:")
    print("  python -m training.train --epochs 50")
    print()
    print("To run inference:")
    print("  import microocr")
    print('  text = microocr.read("iVBORw0KGgo...")')


if __name__ == "__main__":
    main()
