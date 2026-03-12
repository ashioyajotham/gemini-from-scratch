"""Dataset classes for causal language modeling."""

from typing import List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from .tokenizer import BPETokenizer


class TextDataset(Dataset):
    """
    Dataset for causal language modeling.

    Takes tokenized sequences and returns (input_ids, target_ids) pairs
    where targets are inputs shifted by one position.

    Example:
        >>> dataset = TextDataset(token_ids, seq_length=512)
        >>> input_ids, target_ids = dataset[0]
        >>> # target_ids[i] = input_ids[i+1] for next token prediction
    """

    def __init__(
        self,
        token_ids: List[int],
        seq_length: int,
        stride: Optional[int] = None,
        pad_id: int = 0
    ):
        """
        Initialize the dataset.

        Args:
            token_ids: Flat list of token IDs (concatenated documents).
            seq_length: Length of each sequence.
            stride: Step size between sequences. If None, equals seq_length (no overlap).
            pad_id: Padding token ID.

        Example:
            >>> token_ids = tokenizer.encode(full_text)
            >>> dataset = TextDataset(token_ids, seq_length=512)
        """
        self.token_ids = token_ids
        self.seq_length = seq_length
        self.stride = stride if stride is not None else seq_length
        self.pad_id = pad_id

        # Calculate number of sequences
        if len(token_ids) <= seq_length:
            self.num_sequences = 1
        else:
            self.num_sequences = max(1, (len(token_ids) - seq_length) // self.stride + 1)

    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return self.num_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single (input, target) pair.

        Args:
            idx: Sequence index.

        Returns:
            Tuple of (input_ids, target_ids) tensors of shape (seq_length,).
            target_ids are input_ids shifted by 1 position.
        """
        start = idx * self.stride
        end = start + self.seq_length + 1  # +1 for target

        # Extract sequence (need seq_length + 1 tokens for input/target pair)
        sequence = self.token_ids[start:end]

        # Pad if necessary
        if len(sequence) < self.seq_length + 1:
            padding = [self.pad_id] * (self.seq_length + 1 - len(sequence))
            sequence = sequence + padding

        # Split into input and target
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        target_ids = torch.tensor(sequence[1:], dtype=torch.long)

        return input_ids, target_ids


class DocumentDataset(Dataset):
    """
    Dataset that keeps documents separate (no cross-document sequences).

    Useful when you don't want the model to learn across document boundaries.

    Example:
        >>> documents = ["First document text.", "Second document text."]
        >>> dataset = DocumentDataset(documents, tokenizer, seq_length=512)
    """

    def __init__(
        self,
        documents: List[str],
        tokenizer: BPETokenizer,
        seq_length: int,
        add_bos: bool = True,
        add_eos: bool = True
    ):
        """
        Initialize the dataset.

        Args:
            documents: List of document strings.
            tokenizer: Tokenizer instance.
            seq_length: Maximum sequence length.
            add_bos: Add beginning-of-sequence token.
            add_eos: Add end-of-sequence token.
        """
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_id

        # Tokenize all documents
        self.sequences = []
        for doc in documents:
            token_ids = tokenizer.encode(doc, add_bos=add_bos, add_eos=add_eos)

            # Split long documents into chunks
            for i in range(0, len(token_ids), seq_length):
                chunk = token_ids[i:i + seq_length + 1]
                if len(chunk) > 1:  # Need at least 2 tokens for input/target
                    self.sequences.append(chunk)

    def __len__(self) -> int:
        """Return the number of sequences."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single (input, target) pair.

        Args:
            idx: Sequence index.

        Returns:
            Tuple of (input_ids, target_ids) tensors.
        """
        sequence = self.sequences[idx]

        # Pad if necessary
        if len(sequence) < self.seq_length + 1:
            padding = [self.pad_id] * (self.seq_length + 1 - len(sequence))
            sequence = sequence + padding

        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        target_ids = torch.tensor(sequence[1:], dtype=torch.long)

        return input_ids, target_ids


def create_dataset(
    texts: Union[str, List[str]],
    tokenizer: BPETokenizer,
    seq_length: int,
    stride: Optional[int] = None,
    document_mode: bool = False
) -> Dataset:
    """
    Create a dataset from texts.

    Args:
        texts: Single text string or list of document strings.
        tokenizer: Tokenizer instance.
        seq_length: Sequence length for each sample.
        stride: Step size between sequences (TextDataset only).
        document_mode: If True, keep documents separate.

    Returns:
        Dataset instance.

    Example:
        >>> # Concatenated mode (default)
        >>> dataset = create_dataset(texts, tokenizer, seq_length=512)

        >>> # Document mode (respects document boundaries)
        >>> dataset = create_dataset(docs, tokenizer, seq_length=512, document_mode=True)
    """
    if document_mode:
        if isinstance(texts, str):
            texts = [texts]
        return DocumentDataset(texts, tokenizer, seq_length)
    else:
        # Concatenate all texts
        if isinstance(texts, list):
            full_text = " ".join(texts)
        else:
            full_text = texts

        # Tokenize
        token_ids = tokenizer.encode(full_text)

        return TextDataset(token_ids, seq_length, stride, pad_id=tokenizer.pad_id)


def train_val_split(
    dataset: Dataset,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """
    Split dataset into training and validation sets.

    Args:
        dataset: Dataset to split.
        val_ratio: Fraction of data for validation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_dataset, val_dataset).

    Example:
        >>> train_ds, val_ds = train_val_split(dataset, val_ratio=0.1)
    """
    from torch.utils.data import random_split

    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)
