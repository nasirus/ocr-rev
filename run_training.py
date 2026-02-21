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
        epochs=200,
        batch_size=32,
        lr=0.001,
        batches_per_epoch=300,
        output_dir="output",
        seed=42,
        val_samples=512,
        val_seed=1337,
        train_min_len=2,
        train_max_len=112,
        val_min_len=2,
        val_max_len=120,
        curriculum=True,
    )
