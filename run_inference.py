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
from typing import Literal

# -- Configuration ----------------------------------------------------------
# Set to a .png or .jpg file path to run on a real image.
# Leave as None to auto-generate a test image.
IMAGE_PATH = None
NUMPY_WEIGHTS_PATH = "output/microocr.npz"
TORCH_WEIGHTS_PATH = "output/microocr_best.pth"
# ---------------------------------------------------------------------------


def generate_test_image() -> tuple[str, str]:
    """Render a known string to a PNG and return (base64, expected_text)."""
    from PIL import Image, ImageDraw, ImageFont

    text = """
                                        ABSTRACT
We present the results of the Exploration of Local VolumE Survey - Field (ELVES-Field), a survey of
the dwarf galaxies in the Local Volume (LV; D < 10 Mpc) over roughly 3, 000 square degrees, focusing
on the field dwarf population. Candidates are detected using a semi-automated algorithm tailored
for low surface brightness dwarfs. Using tests with injected galaxies, we show the detection is 50%
complete to mg ∼ 20 mag and M⋆ ∼ 106 M⊙. Candidates are confirmed to be true nearby dwarfs
through distance measurements including redshift, tip of the red giant branch, and surface brightness
fluctuations. We identify isolated, field dwarfs using various environmental criteria. Over the survey
footprint, we detect and confirm 95 LV dwarfs, 44 of which we consider isolated. Using this sample,
we infer the field dwarf mass function and find good agreement at the high-mass end with previous
redshift surveys and with the predictions of the IllustrisTNG simulation. This sample of isolated, field
dwarfs represents a powerful dataset to investigate aspects of small-scale structure and the effect of
environment on dwarf galaxy evolution.
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


def _choose_backend() -> tuple[Literal["torch", "numpy"], Path, str]:
    """Prefer CUDA torch backend when available, otherwise use NumPy."""
    npz = Path(NUMPY_WEIGHTS_PATH)
    pth = Path(TORCH_WEIGHTS_PATH)

    try:
        import torch  # noqa: F401
    except Exception:
        if not npz.exists():
            raise FileNotFoundError(
                f"Neither torch backend nor NumPy weights are usable. "
                f"Missing: {NUMPY_WEIGHTS_PATH}"
            )
        return "numpy", npz, "torch not installed"

    import torch

    if torch.cuda.is_available() and pth.exists():
        return "torch", pth, "cuda available"
    if torch.cuda.is_available() and not pth.exists():
        if npz.exists():
            return "numpy", npz, f"missing torch weights: {TORCH_WEIGHTS_PATH}"
        raise FileNotFoundError(
            f"CUDA is available but torch weights are missing: {TORCH_WEIGHTS_PATH}. "
            f"NumPy fallback weights are also missing: {NUMPY_WEIGHTS_PATH}"
        )
    if npz.exists():
        return "numpy", npz, "cuda not available"
    if pth.exists():
        # Keep the "cuda only" policy for torch backend, but provide a clear
        # error so users can still run by exporting/using npz.
        raise FileNotFoundError(
            f"NumPy weights missing at {NUMPY_WEIGHTS_PATH}. "
            f"Found torch weights at {TORCH_WEIGHTS_PATH}, but torch backend "
            "is enabled only when CUDA is available."
        )
    raise FileNotFoundError(
        f"Weights not found. Expected either {NUMPY_WEIGHTS_PATH} or {TORCH_WEIGHTS_PATH}."
    )


def _load_torch_model(weights: Path):
    """Load PyTorch checkpoint onto CUDA for fast inference."""
    import torch

    from microocr.model import NUM_CLASSES, MicroOCRModel

    device = torch.device("cuda")
    model = MicroOCRModel(num_classes=NUM_CLASSES).to(device)
    state = torch.load(str(weights), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _read_torch(
    b64_string: str,
    model,
    *,
    decode_mode: Literal["greedy", "beam"] = "beam",
    beam_width: int = 12,
    case_normalization: Literal["none", "mixed", "lower"] = "mixed",
) -> str:
    """Run OCR with PyTorch model forward on CUDA."""
    import torch

    from microocr.inference import _decode_line, _normalize_case_text
    from microocr.decode import decode_base64
    from microocr.preprocess import TARGET_HEIGHT, binarize, preprocess
    from microocr.segment import segment_lines

    gray = decode_base64(b64_string)
    binary = binarize(gray)
    lines = segment_lines(binary)

    out_lines: list[str] = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for line_img in lines:
            processed = preprocess(
                line_img,
                target_height=TARGET_HEIGHT,
                already_binary=True,
                resize_mode="bilinear",
            )
            x = (
                torch.from_numpy(processed)
                .to(device=device, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            logits = model(x)[:, 0, :].detach().cpu().numpy()
            text = _decode_line(
                logits=logits,
                decode_mode=decode_mode,
                reject_blank=True,
                reject_blank_ratio=0.90,
                reject_nonblank_peak=0.55,
                low_confidence_beam_fallback=False,
                fallback_margin=0.05,
                beam_width=beam_width,
                weights=None,
            )
            if text:
                out_lines.append(text)

    return _normalize_case_text("\n".join(out_lines), mode=case_normalization)


def _read_file_torch(
    path: Path,
    model,
    *,
    decode_mode: Literal["greedy", "beam"] = "beam",
    beam_width: int = 12,
    case_normalization: Literal["none", "mixed", "lower"] = "mixed",
) -> str:
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return _read_torch(
        b64,
        model,
        decode_mode=decode_mode,
        beam_width=beam_width,
        case_normalization=case_normalization,
    )


def main():
    try:
        backend, weights, why = _choose_backend()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        print("Train first: python run_training.py")
        return

    print(f"Backend: {backend} ({why})")
    print(f"Weights: {weights}")

    import microocr

    torch_model = _load_torch_model(weights) if backend == "torch" else None

    if IMAGE_PATH is not None:
        # Run on a real image file
        path = Path(IMAGE_PATH)
        if not path.exists():
            print(f"ERROR: Image not found at '{IMAGE_PATH}'.")
            return

        print(f"Running inference on: {path}")
        if backend == "torch":
            result = _read_file_torch(
                path,
                torch_model,
                decode_mode="beam",
                beam_width=12,
                case_normalization="mixed",
            )
        else:
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
        if backend == "torch":
            result = _read_torch(
                b64,
                torch_model,
                decode_mode="beam",
                beam_width=12,
                case_normalization="mixed",
            )
        else:
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
