# Repository Guidelines

## Project Objective
MicroOCR is a lightweight OCR project that converts base64-encoded images into text with minimal runtime dependencies. The goal is accurate alphanumeric text recognition for constrained environments (edge/mobile/browser-friendly workflows), using PyTorch for training and NumPy/Torch for inference.

## Model Architecture
The recognizer is a compact CNN + CTC pipeline:
- Input line images are normalized to height 32.
- Convolution blocks extract visual features and reduce spatial dimensions.
- Features are reshaped into a time sequence and projected with linear layers.
- A CTC decoder (greedy or beam) maps logits to final text tokens.

## Project Structure & Module Organization
Core OCR code lives in `microocr/`:
- `inference.py`, `ctc.py`, `decode.py`, `preprocess.py`, `segment.py` handle runtime OCR.
- `model.py` and `export.py` support training/export workflows.

Training code is in `training/` (`train.py`, `eval.py`, `synth_data.py`, `augment.py`).  
Tests are in `tests/` (`test_pipeline.py`, `test_training_eval_schedule.py`).  
Entrypoints/scripts: `run_training.py`, `run_inference.py`, `benchmark_inference.py`.  
Generated artifacts are written to `output/` (checkpoints, exported `.npz`).

## Build, Test, and Development Commands
- `uv sync`: install project dependencies.
- `uv sync --extra train`: install training extras.
- `uv sync --extra dev`: install dev/test extras.
- `python run_training.py`: run default training flow.
- `python -m training.train --epochs 50 --batch-size 32 --output-dir output`: configurable training.
- `python run_inference.py`: run quick OCR demo.
- `python benchmark_inference.py --weights-path output/microocr.npz --samples 64`: latency benchmark.
- `uv run pytest -q`: run full test suite.

## Coding Style & Naming Conventions
Use Python 3.10+ with 4-space indentation and PEP 8 naming:
- `snake_case` for functions/variables/modules.
- `PascalCase` for classes.
- Keep public API changes explicit in `microocr/__init__.py`.

Prefer small, composable functions in pipeline stages and keep inference code NumPy-first (avoid adding torch dependency to runtime paths).

## Testing Guidelines
Use `pytest`. Name tests `test_*.py` and test functions `test_*`.  
For pipeline/model changes, add or update deterministic tests in `tests/test_pipeline.py`.  
Before opening a PR, run:
- `uv run pytest -q`
- target script checks (for example `python run_inference.py`) when behavior changes.

## Commit & Pull Request Guidelines
History includes short commits like `0.10 from colab` and descriptive ones like dependency refactors. Prefer descriptive, imperative commit subjects (for example `Refactor CTC beam search scoring`) and keep them focused.

PRs should include:
- What changed and why.
- Linked issue (if any).
- Validation evidence (test command + result).
- Sample OCR output or benchmark deltas when inference/training behavior is modified.
