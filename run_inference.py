"""
Run MicroOCR inference on an image.

Run:
    python run_inference.py

No arguments needed. Generates a sample test image with known text,
encodes it as base64, runs the MicroOCR pipeline, and prints the result.

If you want to test on your own image, edit IMAGE_PATH below.
"""

import base64
from pathlib import Path

# -- Configuration ----------------------------------------------------------
# Set to a .png or .jpg file path to run on a real image.
# Leave as None to auto-generate a test image.
IMAGE_PATH = None
WEIGHTS_PATH = "output/microocr.npz"
# ---------------------------------------------------------------------------


def generate_test_image() -> tuple[str, str]:
    """Render a known string to a PNG and return (base64, expected_text)."""
    from PIL import Image, ImageDraw, ImageFont

    text = """
    if we need to build a new ocr from scratch, i mean let's forget all the actual technic of converting image to text.
the main idea is to start from a base64 content from image contening a text and convert it to token.
be creative and let's brainstomre how can we build a new framework in python to be efficient , capacle to run in constrained hardware liike phone or web browser.
keep it simple and only focused on coverting base64 content to text toekn, we dont need to have an llm that understand or interpret just converting image to text as OCR
    """

    # Render
    font_size = 32
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    tmp = Image.new("L", (1, 1), color=255)
    draw = ImageDraw.Draw(tmp)
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0] + 20
    h = bbox[3] - bbox[1] + 20

    img = Image.new("L", (w, h), color=255)
    draw = ImageDraw.Draw(img)
    draw.text((10 - bbox[0], 10 - bbox[1]), text, fill=0, font=font)

    # Save to a temp file and encode
    tmp_path = Path("test_sample.png")
    img.save(tmp_path)

    raw = tmp_path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")

    print(f"Generated test image: {tmp_path} ({w}x{h})")
    print(f"Expected text: '{text}'")
    return b64, text


def main():
    weights = Path(WEIGHTS_PATH)
    if not weights.exists():
        print(f"ERROR: Weights not found at '{WEIGHTS_PATH}'.")
        print("Train the model first:  python run_training.py")
        return

    import microocr

    if IMAGE_PATH is not None:
        # Run on a real image file
        path = Path(IMAGE_PATH)
        if not path.exists():
            print(f"ERROR: Image not found at '{IMAGE_PATH}'.")
            return

        print(f"Running inference on: {path}")
        result = microocr.read_file(
            str(path),
            weights_path=str(weights),
            decode_mode="beam",
            beam_width=12,
            case_normalization="mixed",
        )
        print(f"\nRecognized text:\n{result}")
    else:
        # Generate a test image and run
        b64, expected = generate_test_image()
        print(f"\nRunning inference...")
        result = microocr.read(
            b64,
            weights_path=str(weights),
            decode_mode="beam",
            beam_width=12,
            case_normalization="mixed",
        )
        print(f"\nRecognized text: '{result}'")
        print(f"Expected text:   '{expected}'")
        if result == expected:
            print("MATCH")
        else:
            print("MISMATCH (model may need more training)")


if __name__ == "__main__":
    main()
