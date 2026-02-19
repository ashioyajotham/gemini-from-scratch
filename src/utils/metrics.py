"""Metrics computation for model evaluation."""

import math
from typing import Optional

import torch
import torch.nn.functional as F


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss.

    Perplexity = exp(loss)

    Lower perplexity indicates better model performance.

    Args:
        loss: Cross-entropy loss value.

    Returns:
        Perplexity value.

    Example:
        >>> loss = 3.5
        >>> ppl = compute_perplexity(loss)
        >>> print(f"Perplexity: {ppl:.2f}")  # ~33.12
    """
    return math.exp(loss)


def compute_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100
) -> float:
    """
    Compute token-level accuracy.

    Args:
        logits: Model output logits of shape (batch, seq_len, vocab_size).
        targets: Target token IDs of shape (batch, seq_len).
        ignore_index: Token ID to ignore in accuracy calculation.

    Returns:
        Accuracy as a float between 0 and 1.

    Example:
        >>> logits = model(input_ids)
        >>> acc = compute_accuracy(logits, targets)
        >>> print(f"Accuracy: {acc:.2%}")
    """
    # Get predictions
    predictions = logits.argmax(dim=-1)

    # Create mask for valid positions
    mask = targets != ignore_index

    # Count correct predictions
    correct = (predictions == targets) & mask
    total = mask.sum().item()

    if total == 0:
        return 0.0

    return correct.sum().item() / total


def compute_top_k_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5,
    ignore_index: int = -100
) -> float:
    """
    Compute top-k accuracy.

    A prediction is correct if the target is in the top-k predictions.

    Args:
        logits: Model output logits of shape (batch, seq_len, vocab_size).
        targets: Target token IDs of shape (batch, seq_len).
        k: Number of top predictions to consider.
        ignore_index: Token ID to ignore in accuracy calculation.

    Returns:
        Top-k accuracy as a float between 0 and 1.

    Example:
        >>> acc_5 = compute_top_k_accuracy(logits, targets, k=5)
        >>> print(f"Top-5 Accuracy: {acc_5:.2%}")
    """
    # Get top-k predictions: (batch, seq_len, k)
    _, top_k_preds = logits.topk(k, dim=-1)

    # Expand targets for comparison: (batch, seq_len, 1)
    targets_expanded = targets.unsqueeze(-1)

    # Check if target is in top-k
    correct = (top_k_preds == targets_expanded).any(dim=-1)

    # Create mask for valid positions
    mask = targets != ignore_index

    # Apply mask
    correct = correct & mask
    total = mask.sum().item()

    if total == 0:
        return 0.0

    return correct.sum().item() / total


def compute_loss_with_mask(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100
) -> torch.Tensor:
    """
    Compute cross-entropy loss with masking.

    Args:
        logits: Model output logits of shape (batch, seq_len, vocab_size).
        targets: Target token IDs of shape (batch, seq_len).
        ignore_index: Token ID to ignore in loss calculation.

    Returns:
        Mean loss value.

    Example:
        >>> loss = compute_loss_with_mask(logits, targets)
        >>> loss.backward()
    """
    # Reshape for cross_entropy: (batch * seq_len, vocab_size) and (batch * seq_len)
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)

    return F.cross_entropy(logits_flat, targets_flat, ignore_index=ignore_index)


class MetricsTracker:
    """
    Track and aggregate metrics during training.

    Example:
        >>> tracker = MetricsTracker()
        >>> for batch in dataloader:
        ...     loss = train_step(batch)
        ...     tracker.update("loss", loss.item())
        >>> print(tracker.get_average("loss"))
    """

    def __init__(self):
        self.metrics = {}
        self.counts = {}

    def update(self, name: str, value: float, count: int = 1) -> None:
        """Add a metric value."""
        if name not in self.metrics:
            self.metrics[name] = 0.0
            self.counts[name] = 0
        self.metrics[name] += value * count
        self.counts[name] += count

    def get_average(self, name: str) -> Optional[float]:
        """Get the average value of a metric."""
        if name not in self.metrics or self.counts[name] == 0:
            return None
        return self.metrics[name] / self.counts[name]

    def get_all_averages(self) -> dict:
        """Get averages of all tracked metrics."""
        return {name: self.get_average(name) for name in self.metrics}

    def reset(self) -> None:
        """Reset all tracked metrics."""
        self.metrics.clear()
        self.counts.clear()
