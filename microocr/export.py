"""
Export utilities for MicroOCR models.

Converts trained PyTorch models to formats suitable for
lightweight inference:
    - ``.npz`` — NumPy archive (for pure NumPy inference)
    - ONNX — for ONNX Runtime (mobile / web)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from microocr.model import MicroOCRModel, NUM_CLASSES
from microocr.preprocess import TARGET_HEIGHT


def export_npz(pth_path: str | Path, npz_path: str | Path) -> None:
    """Export a PyTorch ``.pth`` checkpoint to a NumPy ``.npz`` file.

    The ``.npz`` file contains all weight arrays keyed by their
    PyTorch state_dict names (e.g., ``conv1.weight``, ``fc2.bias``).

    Args:
        pth_path: Path to the ``.pth`` file (PyTorch state_dict).
        npz_path: Output path for the ``.npz`` file.
    """
    state = torch.load(str(pth_path), map_location="cpu", weights_only=True)
    arrays = {k: v.numpy() for k, v in state.items()}
    np.savez(str(npz_path), **arrays)

    # Report size
    total_bytes = sum(a.nbytes for a in arrays.values())
    print(f"Exported {len(arrays)} arrays to {npz_path}")
    print(f"  Total size: {total_bytes:,} bytes ({total_bytes / 1024:.1f} KB)")
    print(f"  Parameters: {sum(a.size for a in arrays.values()):,}")


def export_onnx(
    pth_path: str | Path,
    onnx_path: str | Path,
    sample_width: int = 128,
) -> None:
    """Export a PyTorch model to ONNX format.

    The ONNX model accepts dynamic-width input and can be used with
    ONNX Runtime on mobile devices or in web browsers (via WASM).

    Args:
        pth_path: Path to the ``.pth`` file.
        onnx_path: Output path for the ``.onnx`` file.
        sample_width: Sample width for tracing (actual input is dynamic).
    """
    model = MicroOCRModel(num_classes=NUM_CLASSES)
    state = torch.load(str(pth_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Dummy input for tracing
    dummy = torch.randn(1, 1, TARGET_HEIGHT, sample_width)

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {3: "width"},
            "logits": {0: "timesteps"},
        },
        opset_version=13,
    )

    file_size = Path(onnx_path).stat().st_size
    print(f"Exported ONNX model to {onnx_path}")
    print(f"  File size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Export MicroOCR model")
    parser.add_argument("pth_path", help="Path to .pth checkpoint")
    parser.add_argument(
        "--format",
        choices=["npz", "onnx", "both"],
        default="both",
        help="Export format",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.format in ("npz", "both"):
        export_npz(args.pth_path, out / "microocr.npz")

    if args.format in ("onnx", "both"):
        export_onnx(args.pth_path, out / "microocr.onnx")


if __name__ == "__main__":
    main()
