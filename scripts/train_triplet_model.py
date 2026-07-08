"""
train_triplet_model.py
----------------------
Fine-tune a SentenceTransformer model on triplet loss.

Default model: sentence-transformers/all-MiniLM-L6-v2
Data: output/triplet_data.csv (query, positive, negative)

Usage:
    python3 scripts/train_triplet_model.py [--model sentence-transformers/all-MiniLM-L6-v2] [--epochs 3] [--batch-size 32] [--lr 2e-5]
"""

import argparse
import csv
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def load_triplet_data(csv_path):
    """Load triplet data from CSV into InputExample format."""
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
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--model', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--data', type=str, default='output/triplet_data.csv')
    parser.add_argument('--output', type=str, default='output/all_minilm_tuned')
    args = parser.parse_args()

    logger.info(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)
    logger.info(f"Model loaded successfully")

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
        warmup_steps=100,
        show_progress_bar=True,
    )

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    logger.info(f"Model saved to {output_path}")


if __name__ == '__main__':
    main()
