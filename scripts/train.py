#!/usr/bin/env python
"""
Training script for transformer language models.

Usage:
    python scripts/train.py --config configs/small_model.yaml
    python scripts/train.py --config configs/small_model.yaml --max_steps 5000
"""

import argparse
import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import load_config, Config, set_seed, get_device, get_logger
from src.data import BPETokenizer, TextDataset, create_dataloader
from src.models import Transformer, TransformerConfig
from src.training import Trainer, TrainingConfig, CheckpointCallback


logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a transformer language model")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/small_model.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to training data (text file)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to tokenizer model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum training steps (overrides config)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    return parser.parse_args()


def load_or_create_tokenizer(tokenizer_path, texts, vocab_size, output_dir):
    """Load existing tokenizer or train a new one."""
    if tokenizer_path and Path(tokenizer_path).exists():
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        return BPETokenizer(tokenizer_path)

    logger.info(f"Training new tokenizer with vocab_size={vocab_size}")
    tokenizer = BPETokenizer()
    tokenizer.train(texts, vocab_size=vocab_size, output_dir=output_dir)
    tokenizer.save(Path(output_dir) / "tokenizer.model")
    return tokenizer


def create_sample_data():
    """Create sample training data for demonstration."""
    sample_texts = [
        "The quick brown fox jumps over the lazy dog. " * 100,
        "Machine learning is a fascinating field of study. " * 100,
        "Neural networks can learn complex patterns in data. " * 100,
        "Transformers have revolutionized natural language processing. " * 100,
        "Attention mechanisms allow models to focus on relevant information. " * 100,
        "Deep learning models require large amounts of training data. " * 100,
        "The self-attention mechanism computes relationships between tokens. " * 100,
        "Language models predict the next word given a context. " * 100,
    ]
    return sample_texts


def main():
    """Main training function."""
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Load configuration
    logger.info(f"Loading config from {args.config}")
    config = Config.from_yaml(args.config)

    # Override config with command line arguments
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size

    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or create training data
    if args.data and Path(args.data).exists():
        logger.info(f"Loading data from {args.data}")
        with open(args.data, "r", encoding="utf-8") as f:
            texts = [f.read()]
    else:
        logger.info("Using sample training data")
        texts = create_sample_data()

    # Load or create tokenizer
    tokenizer = load_or_create_tokenizer(
        args.tokenizer,
        texts,
        config.model.vocab_size,
        output_dir,
    )

    # Update vocab size to match tokenizer
    actual_vocab_size = tokenizer.vocab_size
    logger.info(f"Tokenizer vocab size: {actual_vocab_size}")

    # Tokenize data
    logger.info("Tokenizing data...")
    full_text = " ".join(texts)
    token_ids = tokenizer.encode(full_text)
    logger.info(f"Total tokens: {len(token_ids):,}")

    # Create dataset and dataloader
    dataset = TextDataset(
        token_ids,
        seq_length=config.data.seq_length,
        pad_id=tokenizer.pad_id,
    )
    logger.info(f"Dataset size: {len(dataset):,} sequences")

    train_dataloader = create_dataloader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 for debugging
        pad_id=tokenizer.pad_id,
    )

    # Create model
    logger.info("Creating model...")
    model_config = TransformerConfig(
        vocab_size=actual_vocab_size,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        d_ff=config.model.d_ff,
        max_seq_len=config.model.max_seq_len,
        dropout=config.model.dropout,
        padding_idx=tokenizer.pad_id,
    )
    model = Transformer(model_config)

    num_params = model.get_num_parameters()
    logger.info(f"Model parameters: {num_params:,}")

    # Create training config
    training_config = TrainingConfig(
        max_steps=config.training.max_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_steps=config.training.warmup_steps,
        gradient_clip=config.training.gradient_clip,
        log_every_n_steps=config.training.log_interval,
        eval_every_n_steps=config.training.eval_interval,
        save_every_n_steps=config.training.save_interval,
    )

    # Create callbacks
    callbacks = [
        CheckpointCallback(
            save_dir=output_dir,
            save_every_n_steps=config.training.save_interval,
        ),
    ]

    # Create trainer
    trainer = Trainer(
        model=model,
        config=training_config,
        train_dataloader=train_dataloader,
        callbacks=callbacks,
        device=device,
    )

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    logger.info("Starting training...")
    results = trainer.train()

    # Save final model
    final_path = output_dir / "final_model.pt"
    model.save_pretrained(final_path)
    logger.info(f"Saved final model to {final_path}")

    logger.info(f"Training complete! Final loss: {results['final_loss']:.4f}")


if __name__ == "__main__":
    main()
