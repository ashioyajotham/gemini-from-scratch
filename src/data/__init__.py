"""
Data processing utilities for the transformer workshop.

This module provides:
- BPE tokenization (wrapping SentencePiece)
- Dataset classes for causal language modeling
- DataLoader creation with proper collation
- Text preprocessing utilities
"""

from .tokenizer import BPETokenizer, load_tokenizer, save_tokenizer
from .dataset import TextDataset, create_dataset
from .dataloader import create_dataloader, collate_fn
from .preprocessing import clean_text, chunk_text, normalize_text

__all__ = [
    # Tokenizer
    "BPETokenizer",
    "load_tokenizer",
    "save_tokenizer",
    # Dataset
    "TextDataset",
    "create_dataset",
    # DataLoader
    "create_dataloader",
    "collate_fn",
    # Preprocessing
    "clean_text",
    "chunk_text",
    "normalize_text",
]
