"""
Internal evaluation utilities for MicroOCR checkpoints.

This module is intended for training-time validation and regression checks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from microocr.ctc import greedy_decode
from microocr.inference import _forward
from training.synth_data import generate_batch


def build_eval_set(
    num_samples: int = 256,
    seed: int = 1337,
) -> tuple[list[np.ndarray], list[str]]:
    """Build a deterministic synthetic eval set."""
    rng = np.random.default_rng(seed)
    return generate_batch(
        num_samples,
        rng=rng,
        apply_augment=False,
        align_with_inference=True,
    )


def evaluate_arrays(
    weights: dict[str, np.ndarray],
    images: list[np.ndarray],
    labels: list[str],
) -> dict[str, float]:
    """Evaluate OCR quality with greedy decoding."""
    if not images:
        raise ValueError("images must be non-empty")
    if len(images) != len(labels):
        raise ValueError("images and labels must have the same length")

    total_chars = 0
    char_errors = 0
    exact = 0

    for img, label in zip(images, labels, strict=True):
        logits = _forward(img, weights)
        pred = greedy_decode(logits)
        char_errors += edit_distance(label, pred)
        total_chars += len(label)
        if pred == label:
            exact += 1

    cer = char_errors / max(total_chars, 1)
    word_acc = exact / len(labels)
    return {"cer": float(cer), "word_acc": float(word_acc)}


def evaluate_npz(
    npz_path: str | Path,
    num_samples: int = 256,
    seed: int = 1337,
) -> dict[str, float]:
    """Evaluate an exported NumPy checkpoint."""
    with np.load(str(npz_path)) as data:
        weights = {k: data[k] for k in data.files}
    images, labels = build_eval_set(num_samples=num_samples, seed=seed)
    return evaluate_arrays(weights, images, labels)


def evaluate_pth(
    pth_path: str | Path,
    num_samples: int = 256,
    seed: int = 1337,
) -> dict[str, float]:
    """Evaluate a PyTorch checkpoint by running NumPy inference."""
    import torch

    state = torch.load(str(pth_path), map_location="cpu", weights_only=True)
    weights = {k: v.cpu().numpy() for k, v in state.items()}
    images, labels = build_eval_set(num_samples=num_samples, seed=seed)
    return evaluate_arrays(weights, images, labels)


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[n]


def main() -> None:
    parser = argparse.ArgumentParser(description="Internal MicroOCR checkpoint evaluator")
    parser.add_argument("checkpoint", help="Path to .pth or .npz file")
    parser.add_argument("--samples", type=int, default=256, help="Eval set size")
    parser.add_argument("--seed", type=int, default=1337, help="Eval RNG seed")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if ckpt.suffix == ".pth":
        metrics = evaluate_pth(ckpt, num_samples=args.samples, seed=args.seed)
    elif ckpt.suffix == ".npz":
        metrics = evaluate_npz(ckpt, num_samples=args.samples, seed=args.seed)
    else:
        raise ValueError("Checkpoint must be .pth or .npz")

    print(
        f"checkpoint={ckpt} samples={args.samples} seed={args.seed} "
        f"cer={metrics['cer']:.4f} word_acc={metrics['word_acc']:.4f}"
    )


if __name__ == "__main__":
    main()
