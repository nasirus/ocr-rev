"""
Benchmark MicroOCR inference stage timings on a deterministic sample set.

Usage:
    python benchmark_inference.py --weights-path output/microocr.npz --samples 64
"""

from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path

import numpy as np

from microocr.inference import _read_with_timing
from training.synth_data import CHARS, _render_text


def _to_base64_png(gray: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(gray, mode="L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _build_samples(count: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    b64_samples: list[str] = []
    chars = list(CHARS)
    for _ in range(count):
        length = int(rng.integers(1, 21))
        label = "".join(rng.choice(chars) for _ in range(length))
        font_size = int(rng.integers(20, 40))
        raw = _render_text(label, font_size, rng)
        b64_samples.append(_to_base64_png(raw))
    return b64_samples


def _summarize(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return (
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 95)),
        float(arr.mean()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MicroOCR inference")
    parser.add_argument(
        "--weights-path",
        type=str,
        default="output/microocr.npz",
        help="Path to .npz weights",
    )
    parser.add_argument("--samples", type=int, default=64, help="Number of samples")
    parser.add_argument("--seed", type=int, default=2026, help="RNG seed")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument(
        "--decode-mode",
        type=str,
        choices=["greedy", "beam"],
        default="greedy",
        help="CTC decode mode",
    )
    args = parser.parse_args()

    if args.samples < 1:
        raise ValueError("--samples must be >= 1")

    weights = Path(args.weights_path)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    samples = _build_samples(args.samples, args.seed)

    # Warmup
    for i in range(min(args.warmup, len(samples))):
        _read_with_timing(
            samples[i],
            weights_path=weights,
            decode_mode=args.decode_mode,  # type: ignore[arg-type]
        )

    stage_values: dict[str, list[float]] = {
        "decode_ms": [],
        "segment_ms": [],
        "preprocess_ms": [],
        "forward_ms": [],
        "decode_ctc_ms": [],
        "total_ms": [],
    }
    for b64 in samples:
        _, timing = _read_with_timing(
            b64,
            weights_path=weights,
            decode_mode=args.decode_mode,  # type: ignore[arg-type]
        )
        for key in stage_values:
            stage_values[key].append(timing[key])

    print(
        f"MicroOCR benchmark: samples={args.samples} seed={args.seed} "
        f"decode_mode={args.decode_mode} weights={weights}"
    )
    print("stage,p50_ms,p95_ms,mean_ms")
    for key in ("decode_ms", "segment_ms", "preprocess_ms", "forward_ms", "decode_ctc_ms", "total_ms"):
        p50, p95, mean = _summarize(stage_values[key])
        print(f"{key},{p50:.3f},{p95:.3f},{mean:.3f}")


if __name__ == "__main__":
    main()
