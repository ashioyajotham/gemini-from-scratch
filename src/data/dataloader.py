"""DataLoader utilities with proper collation for language modeling."""

from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    pad_id: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for batching variable-length sequences.

    Pads sequences to the same length within a batch.

    Args:
        batch: List of (input_ids, target_ids) tuples.
        pad_id: Padding token ID.

    Returns:
        Tuple of batched (input_ids, target_ids) tensors.

    Example:
        >>> dataloader = DataLoader(dataset, collate_fn=lambda b: collate_fn(b, pad_id=0))
    """
    input_ids = [item[0] for item in batch]
    target_ids = [item[1] for item in batch]

    # Pad sequences
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    target_ids_padded = pad_sequence(target_ids, batch_first=True, padding_value=pad_id)

    return input_ids_padded, target_ids_padded


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    pad_id: int = 0
) -> DataLoader:
    """
    Create a DataLoader with proper collation for language modeling.

    Args:
        dataset: Dataset to load from.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle data.
        num_workers: Number of worker processes for loading.
        pin_memory: Pin memory for faster GPU transfer.
        drop_last: Drop the last incomplete batch.
        pad_id: Padding token ID for collation.

    Returns:
        Configured DataLoader instance.

    Example:
        >>> train_loader = create_dataloader(
        ...     dataset,
        ...     batch_size=32,
        ...     shuffle=True,
        ...     num_workers=4
        ... )
        >>> for input_ids, target_ids in train_loader:
        ...     output = model(input_ids)
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=lambda b: collate_fn(b, pad_id=pad_id)
    )


def create_train_val_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = True,
    pad_id: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        batch_size: Batch size for both loaders.
        num_workers: Number of worker processes.
        pin_memory: Pin memory for faster GPU transfer.
        pad_id: Padding token ID.

    Returns:
        Tuple of (train_loader, val_loader).

    Example:
        >>> train_loader, val_loader = create_train_val_loaders(
        ...     train_ds, val_ds, batch_size=32
        ... )
    """
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        pad_id=pad_id
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        pad_id=pad_id
    )

    return train_loader, val_loader


class InfiniteDataLoader:
    """
    DataLoader that loops infinitely over the dataset.

    Useful for training with a fixed number of steps instead of epochs.

    Example:
        >>> loader = InfiniteDataLoader(dataset, batch_size=32)
        >>> for step, (input_ids, target_ids) in enumerate(loader):
        ...     if step >= max_steps:
        ...         break
        ...     loss = train_step(input_ids, target_ids)
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        pad_id: int = 0
    ):
        """
        Initialize the infinite dataloader.

        Args:
            dataset: Dataset to load from.
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle data.
            num_workers: Number of worker processes.
            pin_memory: Pin memory for faster GPU transfer.
            pad_id: Padding token ID.
        """
        self.dataloader = create_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            pad_id=pad_id
        )
        self._iterator = None

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._iterator is None:
            self._iterator = iter(self.dataloader)

        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.dataloader)
            return next(self._iterator)

    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        return len(self.dataloader)
