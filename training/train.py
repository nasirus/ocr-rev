"""
Training script for MicroOCR.

Usage:
    python -m training.train [--epochs 50] [--batch-size 32] [--lr 0.001]

Trains the MicroOCR CNN with CTC loss on synthetically generated text
images. Exports trained weights as both ``.pth`` (PyTorch) and ``.npz``
(NumPy) for inference.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from microocr.model import BLANK_IDX, NUM_CLASSES, MicroOCRModel, char_to_index
from microocr.preprocess import TARGET_HEIGHT
from training.eval import build_eval_set, evaluate_arrays
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


def train(
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.001,
    batches_per_epoch: int = 200,
    output_dir: str = "output",
    seed: int = 42,
    val_samples: int = 256,
    val_seed: int = 1337,
) -> None:
    """Train the MicroOCR model.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        batches_per_epoch: Number of batches per epoch.
        output_dir: Directory to save model weights.
        seed: Random seed.
        val_samples: Number of synthetic eval samples.
        val_seed: RNG seed for deterministic eval set.
    """
    if val_samples < 1:
        raise ValueError("val_samples must be >= 1")

    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Create model
    model = MicroOCRModel(num_classes=NUM_CLASSES).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ctc_loss = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    eval_images, eval_labels = build_eval_set(num_samples=val_samples, seed=val_seed)
    print(f"Validation set: {val_samples} synthetic samples (seed={val_seed})")

    best_val_cer = float("inf")
    best_val_word_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx in range(1, batches_per_epoch + 1):
            # Generate synthetic batch
            images, labels = generate_batch(batch_size, rng=rng)

            # Collate
            batch_imgs, targets, input_lens, target_lens = collate_batch(images, labels)
            batch_imgs = batch_imgs.to(device)
            targets = targets.to(device)

            # Forward pass
            logits = model(batch_imgs)  # (T, B, C)
            log_probs = logits.log_softmax(dim=2)

            # CTC loss
            loss = ctc_loss(log_probs, targets, input_lens, target_lens)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
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

        model.eval()
        val = _evaluate_model(model, eval_images, eval_labels)
        val_cer = val["cer"]
        val_word_acc = val["word_acc"]
        print(f"  val_cer={val_cer:.4f}  val_word_acc={val_word_acc:.4f}")

        scheduler.step(val_cer)

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


def _evaluate_model(
    model: MicroOCRModel,
    images: list[np.ndarray],
    labels: list[str],
) -> dict[str, float]:
    """Evaluate current model weights with NumPy inference."""
    state = model.state_dict()
    weights = {k: v.detach().cpu().numpy() for k, v in state.items()}
    return evaluate_arrays(weights, images, labels)


def _save_model(model: MicroOCRModel, path: Path) -> None:
    """Save PyTorch model state dict."""
    torch.save(model.state_dict(), str(path))


def _export_npz(model: MicroOCRModel, path: Path) -> None:
    """Export model weights as NumPy .npz for inference without PyTorch."""
    state = model.state_dict()
    arrays = {k: v.cpu().numpy() for k, v in state.items()}
    np.savez(str(path), **arrays)


def main():
    parser = argparse.ArgumentParser(description="Train MicroOCR model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
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
    )


if __name__ == "__main__":
    main()
