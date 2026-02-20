"""
Train the MicroOCR model.

Run:
    python run_training.py

No arguments needed. Trains with a tuned synthetic-data setup and
saves weights to output/ and microocr/weights/.
"""

from training.train import train

if __name__ == "__main__":
    train(
        epochs=100,
        batch_size=24,
        lr=0.0006,
        batches_per_epoch=200,
        output_dir="output",
        seed=42,
        val_samples=384,
        val_seed=1337,
        train_min_len=2,
        train_max_len=64,
        val_min_len=2,
        val_max_len=72,
        curriculum=True,
    )
