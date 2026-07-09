"""
train_triplet_model.py
----------------------
Fine-tune a SentenceTransformer model on triplet loss.

Default model: sentence-transformers/all-MiniLM-L6-v2
Data: output/triplet_data.csv (query, positive, negative)

Uses SentenceTransformerTrainer directly (not the deprecated .fit() wrapper) so
that --logging-steps and the loss callback actually fire — .fit()'s `callback`
param is silently ignored unless an `evaluator` is passed, which we don't have.

Usage:
    python3 scripts/train_triplet_model.py [--model sentence-transformers/all-MiniLM-L6-v2] [--epochs 3] [--batch-size 32] [--lr 2e-5]
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from transformers import TrainerCallback

# Setup logging with more detail
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

# Also log to stdout for visibility
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


class LossLogger:
    """Log training losses to JSON file."""

    def __init__(self, output_path):
        self.output_path = Path(output_path)
        self.losses = []

    def log_step(self, epoch, step, loss, learning_rate):
        """Log a training step."""
        self.losses.append({
            'epoch': epoch,
            'step': step,
            'loss': float(loss),
            'learning_rate': float(learning_rate),
        })

    def save(self):
        """Save losses to JSON file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w') as f:
            json.dump(self.losses, f, indent=2)
        logger.info(f"Loss log saved to {self.output_path}")


class LossLoggerCallback(TrainerCallback):
    """Bridges HF Trainer's on_log events (fired every `logging_steps`) into LossLogger."""

    def __init__(self, loss_logger):
        self.loss_logger = loss_logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or 'loss' not in logs:
            return
        self.loss_logger.log_step(
            epoch=logs.get('epoch', state.epoch),
            step=state.global_step,
            loss=logs['loss'],
            learning_rate=logs.get('learning_rate', 0.0),
        )
        logger.info(f"step={state.global_step} epoch={logs.get('epoch', state.epoch):.2f} loss={logs['loss']:.4f}")


def load_triplet_dataset(csv_path):
    """Load triplet data from CSV into a datasets.Dataset with anchor/positive/negative columns."""
    anchors, positives, negatives = [], [], []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            anchors.append(row['query'])
            positives.append(row['positive'])
            negatives.append(row['negative'])
    return Dataset.from_dict({'anchor': anchors, 'positive': positives, 'negative': negatives})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--logging-steps', type=int, default=50, help='How often (in steps) to log training loss')
    parser.add_argument('--model', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--data', type=str, default='output/triplet_data.csv')
    parser.add_argument('--output', type=str, default='output/all_minilm_tuned')
    args = parser.parse_args()

    logger.info(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)
    logger.info(f"Model loaded successfully")

    logger.info(f"Loading triplet data from {args.data}")
    train_dataset = load_triplet_dataset(args.data)
    steps_per_epoch = len(train_dataset) // args.batch_size
    logger.info(f"Loaded {len(train_dataset):,} triplets")
    logger.info(f"Batch size: {args.batch_size}, Steps per epoch: {steps_per_epoch}")

    train_loss = losses.TripletLoss(model=model, triplet_margin=args.margin)
    logger.info(f"TripletLoss initialized with margin={args.margin}")

    # Setup loss logging
    loss_log_path = Path(args.output).parent / f"{Path(args.output).stem}_losses.json"
    loss_logger = LossLogger(loss_log_path)

    logger.info(f"Starting training: {args.epochs} epochs, lr={args.lr}")
    logger.info(f"Loss will be logged to {loss_log_path} every {args.logging_steps} steps")

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_path / 'checkpoints'),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=100,
        logging_steps=args.logging_steps,
        save_strategy='no',
        report_to='none',
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
        callbacks=[LossLoggerCallback(loss_logger)],
    )
    trainer.train()

    loss_logger.save()

    # Final model save
    model.save(str(output_path))
    logger.info(f"Model saved to {output_path}")


if __name__ == '__main__':
    main()
