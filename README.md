# MicroOCR

Lightweight OCR pipeline that reads **base64-encoded images** and returns text tokens.

The project is designed around:
- Training with PyTorch on synthetic text images.
- Deployment-time inference in pure NumPy (no torch dependency).
- Small model size suitable for constrained environments.

## What It Does

MicroOCR provides an end-to-end OCR flow:
1. Decode base64 image data (`PNG`/`JPEG`).
2. Convert image to grayscale and binarize it.
3. Segment text into lines.
4. Normalize each line to a fixed height.
5. Run a compact CNN + CTC decoder.
6. Return recognized text (newline-separated for multi-line input).

Primary API:
- `microocr.read(base64_string, weights_path=...)`
- `microocr.read_file(path, weights_path=...)`

## Repository Layout

```text
microocr/
  __init__.py       # Public API: read, read_file
  decode.py         # Base64 -> grayscale numpy array
  preprocess.py     # Binarize/crop/resize pipeline
  segment.py        # Line segmentation (projection profile)
  inference.py      # Pure NumPy model forward + OCR orchestration
  ctc.py            # Greedy and beam CTC decoding
  model.py          # PyTorch model definition (training side)
  export.py         # .pth -> .npz/.onnx export

training/
  synth_data.py     # Synthetic text rendering
  augment.py        # Numpy augmentations
  train.py          # Training loop and checkpoint export
  eval.py           # Internal deterministic CER/word-acc evaluation

tests/
  test_pipeline.py  # Unit/integration coverage across modules

run_training.py     # Opinionated training entrypoint
run_inference.py    # Quick inference demo script
benchmark_inference.py  # Stage-wise latency benchmark (p50/p95)
main.py             # Minimal CLI-style project info
```

## Model Overview

Architecture (`microocr/model.py`):
- Input: `(B, 1, 32, W)` grayscale line images.
- CNN blocks:
  - `Conv(1->16, 3x3)` + ReLU + MaxPool
  - `Conv(16->32, 3x3)` + ReLU + MaxPool
  - `Conv(32->64, 3x3)` + ReLU
  - `Conv(64->64, 3x3)` + ReLU
- Sequence projection:
  - reshape `(B, 64, 8, T)` -> `(B, T, 512)`
  - `Linear(512->128)` + ReLU
  - `Linear(128->63)` (62 alnum chars + CTC blank)
- Output for CTC: `(T, B, C)`.

Character set:
- `a-z`, `A-Z`, `0-9` (62 symbols), plus CTC blank.

## Requirements

- Python `>=3.10`
- Core dependency:
  - `numpy`
- Optional:
  - `torch` (training/export)
  - `pillow` (synthetic data generation and JPEG decode fallback)
  - `pytest` (tests)

Project metadata is in `pyproject.toml`.

## Setup

### Option 1: `uv` (recommended in this repo)

```bash
uv sync
```

For training extras:

```bash
uv sync --extra train
```

For development/testing extras:

```bash
uv sync --extra dev
```

### Option 2: pip

```bash
pip install -e .
pip install -e ".[train,dev]"
```

## Quick Start

### 1. Train a model

Simple wrapper:

```bash
python run_training.py
```

Or configurable training module:

```bash
python -m training.train --epochs 50 --batch-size 32 --lr 0.001 --batches-per-epoch 200 --output-dir output
```

Expected artifacts in `output/`:
- `microocr_best.pth`
- `microocr_epoch10.pth`, `microocr_epoch20.pth`, ...
- `microocr_final.pth`
- `microocr.npz` (NumPy weights from best validation CER checkpoint)
- `microocr_final.npz` (NumPy weights from final epoch)

Training also exports to package path:
- `microocr/weights/microocr.npz`

### 2. Run inference

Demo script:

```bash
python run_inference.py
```

By default this script:
- Checks for `output/microocr.npz`
- Generates a sample image (`Hello123`)
- Runs OCR and prints expected vs recognized output

To run on your own image, set `IMAGE_PATH` in `run_inference.py`.

### 3. Use as a library

```python
import microocr

# base64_string can be plain base64 or data URI
text = microocr.read(base64_string, weights_path="output/microocr.npz")

# Optional controls:
# - decode_mode: "greedy" (default) or "beam"
# - reject_blank: suppress false positives on blank/near-blank lines
# - max_line_width: optional latency guardrail for long lines
text = microocr.read(
    base64_string,
    weights_path="output/microocr.npz",
    decode_mode="greedy",
    reject_blank=True,
    max_line_width=None,
)

# or from file
text2 = microocr.read_file("sample.png", weights_path="output/microocr.npz")
```

## Inference Pipeline Details

`microocr.read(...)` performs:
1. `decode_base64`: base64 -> `np.uint8` grayscale image.
2. `binarize`: adaptive local thresholding.
3. `segment_lines`: horizontal projection line segmentation.
4. Per line:
   - `preprocess` (crop-to-content + resize to height 32 + normalize; no second binarization pass)
   - `_forward` (pure NumPy CNN)
   - `greedy_decode` or `beam_decode` (CTC decoding, configurable)
   - blank-line rejection gate (optional, enabled by default)
5. Join line results with `\n`.

## Evaluation and Benchmarking

Internal deterministic checkpoint evaluation (fixed synthetic set):

```bash
python -m training.eval output/microocr_best.pth --samples 256 --seed 1337
```

Inference stage benchmark (decode/segment/preprocess/forward/decode/total, p50/p95):

```bash
python benchmark_inference.py --weights-path output/microocr.npz --samples 64
```

## Exporting Trained Models

Use `microocr/export.py` to export from a `.pth` checkpoint:

```bash
python -m microocr.export output/microocr_best.pth --format both --output-dir output
```

Formats:
- `npz`: for pure NumPy inference in this project.
- `onnx`: for ONNX Runtime / mobile / web pipelines.

## Testing

Run tests with:

```bash
uv run pytest -q
```

Current repo baseline: `31 passed` (local run in this workspace).

Coverage includes:
- Image decode (PNG, data URI handling, invalid format behavior)
- Preprocessing utilities
- Line segmentation
- CTC decoding behavior
- PyTorch model shape/parameter sanity checks
- NumPy forward parity with PyTorch forward
- Synthetic data and augmentation

## Known Constraints

- Alphabet is strictly alphanumeric (`a-zA-Z0-9`).
- No language model or dictionary correction.
- Line segmentation is projection-based and can fail on complex layouts.
- JPEG decode requires `Pillow`.
- `microocr.read(...)` default weights path expects `microocr/weights/microocr.npz`; pass `weights_path` explicitly if using `output/microocr.npz`.

## Typical Development Flow

1. Train:
   - `python -m training.train ...`
2. Validate:
   - `uv run pytest -q`
3. Try inference:
   - `python run_inference.py`
4. (Optional) export ONNX:
   - `python -m microocr.export output/microocr_best.pth --format onnx`
