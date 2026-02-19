"""
Loss functions for language model training.

This module implements:
- Cross-entropy loss for language modeling
- Label smoothing
- Perplexity computation
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss for language modeling.

    Computes the standard cross-entropy loss between logits and targets,
    with optional label smoothing.

    Args:
        ignore_index: Token ID to ignore in loss computation (e.g., padding).
        label_smoothing: Label smoothing factor (0 = no smoothing).
        reduction: Reduction method ('mean', 'sum', 'none').

    Example:
        >>> loss_fn = CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()

        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss.

        Args:
            logits: Model output of shape (batch, seq_len, vocab_size).
            targets: Target token IDs of shape (batch, seq_len).

        Returns:
            Loss value.
        """
        # Reshape for cross_entropy
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.view(-1, vocab_size)
        targets = targets.view(-1)

        return F.cross_entropy(
            logits,
            targets,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction,
        )


class LanguageModelingLoss(nn.Module):
    """
    Loss function for causal language modeling.

    Automatically shifts targets to align with predictions
    (predicting next token from current position).

    Args:
        ignore_index: Token ID to ignore (padding).
        label_smoothing: Label smoothing factor.

    Example:
        >>> loss_fn = LanguageModelingLoss(ignore_index=0)
        >>> # logits: (batch, seq_len, vocab_size)
        >>> # input_ids: (batch, seq_len)
        >>> loss = loss_fn(logits, input_ids)
    """

    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ):
        super().__init__()

        self.ignore_index = ignore_index
        self.loss_fn = CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        shift: bool = True,
    ) -> torch.Tensor:
        """
        Compute language modeling loss.

        Args:
            logits: Model output of shape (batch, seq_len, vocab_size).
            labels: Target token IDs of shape (batch, seq_len).
            shift: If True, shift labels for next-token prediction.

        Returns:
            Loss value.
        """
        if shift:
            # Shift so that tokens < n predict n
            # logits: [0, 1, 2, ..., n-1] predict labels: [1, 2, 3, ..., n]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
        else:
            shift_logits = logits
            shift_labels = labels

        return self.loss_fn(shift_logits, shift_labels)


def compute_loss_with_mask(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Compute loss with optional attention mask.

    Args:
        logits: Model output of shape (batch, seq_len, vocab_size).
        targets: Target token IDs of shape (batch, seq_len).
        mask: Optional mask of shape (batch, seq_len). 1 = compute loss, 0 = ignore.
        label_smoothing: Label smoothing factor.

    Returns:
        Mean loss over non-masked positions.
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Reshape
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)

    # Compute per-token loss
    loss = F.cross_entropy(
        logits_flat,
        targets_flat,
        reduction="none",
        label_smoothing=label_smoothing,
    )
    loss = loss.view(batch_size, seq_len)

    # Apply mask
    if mask is not None:
        loss = loss * mask
        return loss.sum() / mask.sum()
    else:
        return loss.mean()


def compute_perplexity(loss: torch.Tensor) -> torch.Tensor:
    """
    Compute perplexity from loss.

    Perplexity = exp(loss)

    Args:
        loss: Cross-entropy loss value.

    Returns:
        Perplexity value.
    """
    return torch.exp(loss)


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Reduces the relative loss for well-classified examples,
    focusing on hard examples.

    Args:
        gamma: Focusing parameter (higher = more focus on hard examples).
        alpha: Class balancing weight.
        ignore_index: Token ID to ignore.
        reduction: Reduction method.

    Example:
        >>> loss_fn = FocalLoss(gamma=2.0)
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 1.0,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss."""
        # Reshape
        if logits.dim() > 2:
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)

        # Compute cross entropy
        ce_loss = F.cross_entropy(
            logits, targets, reduction="none", ignore_index=self.ignore_index
        )

        # Compute probabilities
        probs = F.softmax(logits, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Compute focal weight
        focal_weight = self.alpha * (1 - pt) ** self.gamma

        # Apply weight
        loss = focal_weight * ce_loss

        # Handle ignore_index
        mask = targets != self.ignore_index

        if self.reduction == "mean":
            return loss[mask].mean()
        elif self.reduction == "sum":
            return loss[mask].sum()
        else:
            return loss
