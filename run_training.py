"""
Train the MicroOCR model.

Run:
    python run_training.py

No arguments needed. Trains for 50 epochs on synthetic data and
saves weights to output/ and microocr/weights/.
"""

from training.train import train

if __name__ == "__main__":
    train(
        epochs=10,
        batch_size=32,
        lr=0.0005,
        batches_per_epoch=200,
        output_dir="output",
        seed=42,
    )
