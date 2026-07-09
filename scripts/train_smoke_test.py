"""
train_smoke_test.py
-------------------
Smoke test: fresh model on 5k hard triplets with parallelized data loading.

Usage:
    python3 scripts/train_smoke_test.py
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import random

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def load_triplet_data_sample(csv_path, sample_size=5000):
    """Load and sample triplet data from CSV."""
    logger.info(f"Loading triplets from {csv_path}")

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        all_triplets = list(reader)

    logger.info(f"Total triplets: {len(all_triplets)}")

    # Sample
    sampled = random.sample(all_triplets, min(sample_size, len(all_triplets)))
    logger.info(f"Sampled {len(sampled)} triplets for smoke test")

    examples = [
        InputExample(texts=[t['query'], t['positive'], t['negative']])
        for t in sampled
    ]

    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--sample-size', type=int, default=5000)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--data', type=str, default='output/triplet_data_hard.csv')
    parser.add_argument('--output', type=str, default='output/all_minilm_smoke')
    args = parser.parse_args()

    logger.info(f"Loading fresh model: sentence-transformers/all-MiniLM-L6-v2")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    logger.info("Fresh model loaded")

    # Load sample
    examples = load_triplet_data_sample(args.data, args.sample_size)

    # DataLoader with parallelization
    train_dataloader = DataLoader(
        examples,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    logger.info(f"DataLoader: batch_size={args.batch_size}, num_workers={args.num_workers}")
    logger.info(f"Steps per epoch: {len(train_dataloader)}")

    train_loss = losses.TripletLoss(model=model, triplet_margin=args.margin)
    logger.info(f"TripletLoss initialized with margin={args.margin}")

    logger.info(f"Starting training: {args.epochs} epochs, lr={args.lr}")

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=50,
        show_progress_bar=True,
        output_path=str(output_path),
        save_best_model=False,
    )

    model.save(str(output_path))
    logger.info(f"Smoke test model saved to {output_path}")
    logger.info("✓ Smoke test complete!")


if __name__ == '__main__':
    main()
