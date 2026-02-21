"""
Training script for MicroOCR.

Usage:
    python -m training.train [--epochs 50] [--batch-size 32] [--lr 0.01]

Trains the MicroOCR CNN with CTC loss on synthetically generated text
images. Exports trained weights as both ``.pth`` (PyTorch) and ``.npz``
(NumPy) for inference.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from microocr.model import (
    BLANK_IDX,
    NUM_CLASSES,
    MicroOCRModel,
    char_to_index,
    index_to_char,
)
from microocr.preprocess import TARGET_HEIGHT
from training.eval import build_eval_set, edit_distance, evaluate_arrays
from training.synth_data import generate_batch


def collate_batch(
    images: list[np.ndarray],
    labels: list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate variable-width images into a padded batch tensor.

    Args:
        images: List of 2-D float32 arrays (H, W_i).
        labels: List of label strings.

    Returns:
        Tuple of (batch_images, targets, input_lengths, target_lengths).
    """
    batch_size = len(images)
    max_w = max(img.shape[1] for img in images)

    # Pad all images to max width (with white = 1.0)
    batch = np.ones((batch_size, 1, TARGET_HEIGHT, max_w), dtype=np.float32)
    for i, img in enumerate(images):
        w = img.shape[1]
        batch[i, 0, :, :w] = img

    # Encode labels as integer sequences
    all_targets: list[int] = []
    target_lengths: list[int] = []
    for label in labels:
        encoded = [char_to_index(c) for c in label]
        all_targets.extend(encoded)
        target_lengths.append(len(encoded))

    # Input lengths: T = W_i // 4 per sample (two 2x2 maxpools reduce width by 4x)
    # Each image has its own actual width — using max_w for all would feed
    # padded blank regions as valid timesteps and bias the model toward blanks.
    input_lengths = [img.shape[1] // 4 for img in images]

    return (
        torch.tensor(batch, dtype=torch.float32),
        torch.tensor(all_targets, dtype=torch.long),
        torch.tensor(input_lengths, dtype=torch.long),
        torch.tensor(target_lengths, dtype=torch.long),
    )


def _parse_eval_each_epoch_cli(raw: str) -> int | bool:
    """Parse compatibility CLI values for --eval-each-epoch."""
    value = raw.strip().lower()
    if value in {"true", "t", "yes", "y", "on"}:
        return True
    if value in {"false", "f", "no", "n", "off"}:
        return False
    try:
        return int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "eval_each_epoch must be a boolean or integer >= 1"
        ) from exc


def _resolve_val_every(
    val_every: int,
    eval_each_epoch: int | bool | None,
) -> int:
    """Normalize validation cadence from val_every and eval_each_epoch alias."""
    if eval_each_epoch is None:
        resolved = val_every
    elif isinstance(eval_each_epoch, bool):
        resolved = 1 if eval_each_epoch else val_every
    elif isinstance(eval_each_epoch, int):
        resolved = eval_each_epoch
    else:
        raise TypeError("eval_each_epoch must be bool, int, or None")

    if resolved < 1:
        raise ValueError("validation interval must be >= 1")
    return resolved


def train(
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.01,
    batches_per_epoch: int = 200,
    output_dir: str = "output",
    seed: int = 42,
    val_samples: int = 256,
    val_seed: int = 1337,
    train_min_len: int = 2,
    train_max_len: int = 112,
    val_min_len: int = 2,
    val_max_len: int = 120,
    curriculum: bool = True,
    entropy_weight: float = 0.0,
    val_backend: Literal["torch", "numpy"] = "torch",
    val_batch_size: int = 128,
    val_every: int = 1,
    eval_each_epoch: int | bool | None = None,
) -> None:
    """Train the MicroOCR model.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Peak learning rate for OneCycleLR.
        batches_per_epoch: Number of batches per epoch.
        output_dir: Directory to save model weights.
        seed: Random seed.
        val_samples: Number of synthetic eval samples.
        val_seed: RNG seed for deterministic eval set.
        train_min_len: Minimum synthetic label length during training.
        train_max_len: Maximum synthetic label length during training.
        val_min_len: Minimum synthetic label length for validation.
        val_max_len: Maximum synthetic label length for validation.
        curriculum: If True, use curriculum learning — start with shorter/
            easier text and progressively increase max label length and
            font-size range over epochs.
        entropy_weight: Optional entropy regularization weight. Keep at
            0.0 for best exact-transcription accuracy on this task.
        val_backend: Validation backend. ``"torch"`` is much faster and runs
            model forward directly in PyTorch; ``"numpy"`` validates through
            exported NumPy inference parity path.
        val_batch_size: Batch size used by torch validation backend.
        val_every: Run validation every N epochs (always runs on final epoch).
        eval_each_epoch: Compatibility alias for validation cadence. Accepts
            bool/int and maps into ``val_every``.
    """
    if val_samples < 1:
        raise ValueError("val_samples must be >= 1")
    if train_min_len < 1 or val_min_len < 1:
        raise ValueError("min lengths must be >= 1")
    if train_max_len < train_min_len:
        raise ValueError("train_max_len must be >= train_min_len")
    if val_max_len < val_min_len:
        raise ValueError("val_max_len must be >= val_min_len")
    if val_backend not in ("torch", "numpy"):
        raise ValueError("val_backend must be 'torch' or 'numpy'")
    if val_batch_size < 1:
        raise ValueError("val_batch_size must be >= 1")
    val_every = _resolve_val_every(val_every=val_every, eval_each_epoch=eval_each_epoch)

    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Create model
    model = MicroOCRModel(num_classes=NUM_CLASSES).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # Optimizer: AdamW with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ctc_loss = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

    # OneCycleLR scheduler — stepped per batch
    total_steps = epochs * batches_per_epoch
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    eval_images, eval_labels = build_eval_set(
        num_samples=val_samples,
        seed=val_seed,
        min_len=val_min_len,
        max_len=val_max_len,
    )
    print(
        f"Validation set: {val_samples} synthetic samples (seed={val_seed}, "
        f"len={val_min_len}-{val_max_len})"
    )
    print(
        "Validation config: "
        f"backend={val_backend}, batch_size={val_batch_size}, every={val_every} epoch(s)"
    )

    best_val_cer = float("inf")
    best_val_word_acc = 0.0

    # Curriculum learning parameters
    cur_start_max_len = max(train_min_len + 2, train_max_len // 3)
    # Font size: start with a narrow mid-range, expand to full range over time
    full_font_min, full_font_max = 20, 40

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        # Curriculum: linearly ramp difficulty from easy to full over epochs
        if curriculum and epochs > 1:
            progress = (epoch - 1) / (epochs - 1)  # 0.0 -> 1.0
            cur_max_len = int(
                cur_start_max_len + progress * (train_max_len - cur_start_max_len)
            )
            cur_max_len = max(cur_max_len, train_min_len + 1)
            # Narrow font range early, full range later
            cur_font_min = int(full_font_min + (1 - progress) * 4)  # 24 -> 20
            cur_font_max = int(full_font_max - (1 - progress) * 6)  # 34 -> 40
            cur_font_range = (cur_font_min, max(cur_font_max, cur_font_min + 4))
        else:
            cur_max_len = train_max_len
            cur_font_range = (full_font_min, full_font_max)

        if curriculum:
            print(f"  [curriculum] max_len={cur_max_len}, font_range={cur_font_range}")

        for batch_idx in range(1, batches_per_epoch + 1):
            # Generate synthetic batch
            images, labels = generate_batch(
                batch_size,
                rng=rng,
                min_len=train_min_len,
                max_len=cur_max_len,
                font_size_range=cur_font_range,
            )

            # Collate
            batch_imgs, targets, input_lens, target_lens = collate_batch(images, labels)
            batch_imgs = batch_imgs.to(device)
            targets = targets.to(device)

            # Forward pass
            logits = model(batch_imgs)  # (T, B, C)
            log_probs = logits.log_softmax(dim=2)

            # CTC loss
            loss = ctc_loss(log_probs, targets, input_lens, target_lens)

            if entropy_weight > 0.0:
                # Optional entropy regularization
                probs = log_probs.exp()
                entropy = -(probs * log_probs).sum(dim=2).mean()
                loss = loss - entropy_weight * entropy

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

            if batch_idx % 50 == 0:
                avg = epoch_loss / batch_idx
                print(
                    f"  Epoch {epoch}/{epochs} "
                    f"[{batch_idx}/{batches_per_epoch}] "
                    f"loss={avg:.4f}"
                )

        avg_loss = epoch_loss / batches_per_epoch
        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch}/{epochs} — "
            f"avg_loss={avg_loss:.4f} — "
            f"time={elapsed:.1f}s — "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        should_validate = (epoch % val_every == 0) or (epoch == epochs)
        if should_validate:
            model.eval()
            val_start = time.time()
            if val_backend == "torch":
                val = _evaluate_model_torch(
                    model,
                    eval_images,
                    eval_labels,
                    device=device,
                    batch_size=val_batch_size,
                )
            else:
                val = _evaluate_model_numpy(model, eval_images, eval_labels)
            val_elapsed = time.time() - val_start
            val_cer = val["cer"]
            val_word_acc = val["word_acc"]
            print(
                f"  val_cer={val_cer:.4f}  val_word_acc={val_word_acc:.4f} "
                f"(val_time={val_elapsed:.1f}s, backend={val_backend})"
            )

            # Save best model
            if val_cer < best_val_cer:
                best_val_cer = val_cer
                best_val_word_acc = val_word_acc
                _save_model(model, out_path / "microocr_best.pth")
                _export_npz(model, out_path / "microocr.npz")
                print(
                    "  -> Saved best model "
                    f"(cer={best_val_cer:.4f}, word_acc={best_val_word_acc:.4f})"
                )
        else:
            print(f"  validation skipped (val_every={val_every})")

        # Save periodic checkpoint
        if epoch % 10 == 0:
            _save_model(model, out_path / f"microocr_epoch{epoch}.pth")

    # Save final model
    _save_model(model, out_path / "microocr_final.pth")
    _export_npz(model, out_path / "microocr_final.npz")
    if not (out_path / "microocr.npz").exists():
        _export_npz(model, out_path / "microocr.npz")
    print(f"\nTraining complete. Weights saved to {out_path}/")
    print(f"  PyTorch: microocr_final.pth")
    print(f"  NumPy:   microocr.npz (best validation checkpoint)")
    print(f"  NumPy:   microocr_final.npz (final epoch)")

    # Also copy to the package weights directory
    weights_dir = Path(__file__).parent.parent / "microocr" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_path / "microocr.npz", weights_dir / "microocr.npz")
    print(f"  Copied to: {weights_dir / 'microocr.npz'}")


def _evaluate_model_numpy(
    model: MicroOCRModel,
    images: list[np.ndarray],
    labels: list[str],
) -> dict[str, float]:
    """Evaluate current model weights with NumPy inference (BN-folded)."""
    weights = fold_bn_into_conv(model)
    return evaluate_arrays(weights, images, labels)


def _ctc_greedy_decode_path(path: np.ndarray) -> str:
    """Collapse repeats/blanks from a greedy CTC argmax path."""
    out: list[str] = []
    prev = BLANK_IDX
    for idx in path:
        cur = int(idx)
        if cur != BLANK_IDX and cur != prev:
            out.append(index_to_char(cur))
        prev = cur
    return "".join(out)


@torch.no_grad()
def _evaluate_model_torch(
    model: MicroOCRModel,
    images: list[np.ndarray],
    labels: list[str],
    *,
    device: torch.device,
    batch_size: int = 128,
) -> dict[str, float]:
    """Evaluate with batched PyTorch forward + greedy CTC decode."""
    if not images:
        raise ValueError("images must be non-empty")
    if len(images) != len(labels):
        raise ValueError("images and labels must have the same length")

    total_chars = 0
    char_errors = 0
    exact = 0

    for start in range(0, len(images), batch_size):
        end = min(start + batch_size, len(images))
        batch_images = images[start:end]
        batch_labels = labels[start:end]

        max_w = max(img.shape[1] for img in batch_images)
        # Use zero right-padding to better match single-sample convolution
        # boundary behavior from the NumPy evaluator.
        batch = np.zeros((len(batch_images), 1, TARGET_HEIGHT, max_w), dtype=np.float32)
        valid_steps = [img.shape[1] // 4 for img in batch_images]
        for i, img in enumerate(batch_images):
            w = img.shape[1]
            batch[i, 0, :, :w] = img

        batch_tensor = torch.from_numpy(batch).to(device)
        logits = model(batch_tensor)  # (T, B, C)
        paths = torch.argmax(logits, dim=2).permute(1, 0).cpu().numpy()  # (B, T)

        for path, label, t_steps in zip(
            paths,
            batch_labels,
            valid_steps,
            strict=True,
        ):
            pred = _ctc_greedy_decode_path(path[:t_steps])
            char_errors += edit_distance(label, pred)
            total_chars += len(label)
            if pred == label:
                exact += 1

    cer = char_errors / max(total_chars, 1)
    word_acc = exact / len(labels)
    return {"cer": float(cer), "word_acc": float(word_acc)}


def fold_bn_into_conv(model: MicroOCRModel) -> dict[str, np.ndarray]:
    """Fold BatchNorm parameters into conv weights and biases.

    For each conv+BN pair, computes:
        w_folded = (gamma / sqrt(var + eps)) * w
        b_folded = (gamma / sqrt(var + eps)) * (b - mean) + beta

    Returns a dict with keys like 'conv1.weight', 'conv1.bias', etc.
    that can be used directly by the NumPy inference engine.
    """
    model.eval()
    state = model.state_dict()
    result: dict[str, np.ndarray] = {}

    for i in range(1, 5):
        conv_w = state[f"conv{i}.weight"].cpu().numpy()
        conv_b = state[f"conv{i}.bias"].cpu().numpy()
        bn_gamma = state[f"bn{i}.weight"].cpu().numpy()
        bn_beta = state[f"bn{i}.bias"].cpu().numpy()
        bn_mean = state[f"bn{i}.running_mean"].cpu().numpy()
        bn_var = state[f"bn{i}.running_var"].cpu().numpy()
        eps = 1e-5  # PyTorch default BN epsilon

        scale = bn_gamma / np.sqrt(bn_var + eps)
        # Reshape scale for broadcasting: (C_out,) → (C_out, 1, 1, 1)
        w_folded = conv_w * scale.reshape(-1, 1, 1, 1)
        b_folded = scale * (conv_b - bn_mean) + bn_beta

        result[f"conv{i}.weight"] = w_folded.astype(np.float32)
        result[f"conv{i}.bias"] = b_folded.astype(np.float32)

    # Copy FC layers directly
    for key in ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"):
        result[key] = state[key].cpu().numpy().astype(np.float32)

    # Copy RNN (BiGRU) weights if present
    for key in state:
        if key.startswith("rnn."):
            result[key] = state[key].cpu().numpy().astype(np.float32)

    return result


def _save_model(model: MicroOCRModel, path: Path) -> None:
    """Save PyTorch model state dict."""
    torch.save(model.state_dict(), str(path))


def _export_npz(
    model: MicroOCRModel,
    path: Path,
    *,
    quantize_int8: bool = False,
) -> None:
    """Export model weights as NumPy .npz for inference without PyTorch.

    BN is folded into conv weights so inference sees only conv.weight/bias.
    Includes arch_version sentinel to enable residual connection in inference.
    Optionally quantizes weights to INT8 for smaller file size.
    """
    arrays = fold_bn_into_conv(model)

    # Architecture version marker: v3 = residual + BiGRU
    arrays["arch_version"] = np.array([3], dtype=np.int32)

    if quantize_int8:
        arrays = _quantize_weights_int8(arrays)

    np.savez(str(path), **arrays)


def _quantize_weights_int8(
    weights: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Per-channel INT8 quantization of weight tensors.

    For each weight tensor, computes per-channel (axis 0) scale factors
    and stores int8 data + scale. Bias tensors are kept as float32.
    """
    result: dict[str, np.ndarray] = {}
    for key, arr in weights.items():
        if key.endswith(".weight") and arr.ndim >= 2:
            # Per-channel quantization along axis 0
            abs_max = np.abs(arr).reshape(arr.shape[0], -1).max(axis=1)
            abs_max = np.maximum(abs_max, 1e-8)  # avoid division by zero
            scale = abs_max / 127.0
            shape = [-1] + [1] * (arr.ndim - 1)
            q = np.round(arr / scale.reshape(shape)).astype(np.int8)
            result[key + "_q"] = q
            result[key + "_scale"] = scale.astype(np.float32)
        else:
            result[key] = arr
    return result


def main():
    parser = argparse.ArgumentParser(description="Train MicroOCR model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Peak learning rate for OneCycleLR",
    )
    parser.add_argument(
        "--batches-per-epoch",
        type=int,
        default=200,
        help="Batches per epoch",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for weights",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--val-samples",
        type=int,
        default=256,
        help="Number of synthetic validation samples",
    )
    parser.add_argument(
        "--val-seed",
        type=int,
        default=1337,
        help="Validation RNG seed",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Export INT8 quantized weights",
    )
    parser.add_argument(
        "--train-min-len", type=int, default=2, help="Training min label length"
    )
    parser.add_argument(
        "--train-max-len", type=int, default=112, help="Training max label length"
    )
    parser.add_argument(
        "--val-min-len", type=int, default=2, help="Validation min label length"
    )
    parser.add_argument(
        "--val-max-len", type=int, default=120, help="Validation max label length"
    )
    parser.add_argument(
        "--val-backend",
        type=str,
        choices=("torch", "numpy"),
        default="torch",
        help="Validation backend: torch (fast) or numpy (parity check).",
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=128,
        help="Batch size used by torch validation backend.",
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=50,
        help="Run validation every N epochs (always runs on final epoch).",
    )
    parser.add_argument(
        "--eval-each-epoch",
        "--eval_each_epoch",
        nargs="?",
        const="true",
        type=_parse_eval_each_epoch_cli,
        default=None,
        help="Compatibility alias for validation cadence. "
        "Accepts true/false or integer N (same as --val-every N).",
    )
    parser.add_argument(
        "--curriculum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable curriculum learning (easy-to-hard progression). "
        "Use --no-curriculum to disable.",
    )
    parser.add_argument(
        "--entropy-weight",
        type=float,
        default=0.0,
        help="Entropy regularization weight (0 disables it).",
    )
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        batches_per_epoch=args.batches_per_epoch,
        output_dir=args.output_dir,
        seed=args.seed,
        val_samples=args.val_samples,
        val_seed=args.val_seed,
        train_min_len=args.train_min_len,
        train_max_len=args.train_max_len,
        val_min_len=args.val_min_len,
        val_max_len=args.val_max_len,
        curriculum=args.curriculum,
        entropy_weight=args.entropy_weight,
        val_backend=args.val_backend,
        val_batch_size=args.val_batch_size,
        val_every=args.val_every,
        eval_each_epoch=args.eval_each_epoch,
    )


if __name__ == "__main__":
    main()
