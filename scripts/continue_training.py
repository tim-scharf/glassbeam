"""
continue_training.py
--------------------
Continue training an existing model for additional epochs.

Usage:
    python3 scripts/continue_training.py --model output/all_minilm_mixed --data output/triplet_data_mixed.csv --epochs 1
"""

import argparse
import csv
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

# Also log to stdout for visibility
import sys
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


class LossLoggerCallback:
    """Log loss every N steps to stdout."""

    def __init__(self, log_every=25):
        self.log_every = log_every
        self.step = 0

    def __call__(self, score, epoch, steps):
        self.step += 1
        if self.step % self.log_every == 0:
            logger.info(f"Epoch {epoch}, Step {self.step}: loss={score:.6f}")


def load_triplet_data(csv_path):
    """Load triplet data from CSV."""
    examples = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            examples.append(InputExample(
                texts=[row['query'], row['positive'], row['negative']]
            ))
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to existing model')
    parser.add_argument('--data', type=str, required=True, help='Path to triplet data CSV')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--margin', type=float, default=1.0)
    args = parser.parse_args()

    logger.info(f"Loading model from: {args.model}")
    model = SentenceTransformer(args.model)
    logger.info("Model loaded")

    logger.info(f"Loading triplet data from {args.data}")
    examples = load_triplet_data(args.data)
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    logger.info(f"Loaded {len(examples):,} triplets")
    logger.info(f"Batch size: {args.batch_size}, Steps per epoch: {len(train_dataloader)}")

    train_loss = losses.TripletLoss(model=model, triplet_margin=args.margin)
    logger.info(f"TripletLoss initialized with margin={args.margin}")

    logger.info(f"Starting training: {args.epochs} epochs, lr={args.lr}")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=50,
        show_progress_bar=True,
        output_path=str(args.model),
        save_best_model=False,
    )

    model.save(str(args.model))
    logger.info(f"Model saved to {args.model}")
    logger.info("✓ Training continuation complete!")


if __name__ == '__main__':
    main()
