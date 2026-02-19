"""
Optimizers and learning rate schedulers.

This module implements:
- AdamW optimizer configuration
- Cosine annealing with warmup
- Linear warmup scheduler
- Learning rate utilities
"""

import math
from typing import Optional, Iterable, Callable

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler


def configure_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.95),
    eps: float = 1e-8,
    no_decay_params: Optional[list] = None,
) -> AdamW:
    """
    Configure AdamW optimizer with weight decay separation.

    Applies weight decay only to weight matrices, not to biases,
    layer norms, or embeddings.

    Args:
        model: Model to optimize.
        learning_rate: Learning rate.
        weight_decay: Weight decay coefficient.
        betas: Adam beta parameters.
        eps: Adam epsilon for numerical stability.
        no_decay_params: List of parameter name patterns to exclude from decay.

    Returns:
        Configured AdamW optimizer.

    Example:
        >>> optimizer = configure_optimizer(model, learning_rate=3e-4)
    """
    if no_decay_params is None:
        no_decay_params = ["bias", "LayerNorm", "layernorm", "layer_norm", "ln_", "norm", "embedding"]

    # Separate parameters into decay and no-decay groups
    decay_params = []
    no_decay_params_list = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if parameter should not have weight decay
        if any(nd in name for nd in no_decay_params):
            no_decay_params_list.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params_list, "weight_decay": 0.0},
    ]

    optimizer = AdamW(param_groups, lr=learning_rate, betas=betas, eps=eps)

    return optimizer


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Create a schedule with linear warmup and cosine decay.

    Learning rate increases linearly during warmup, then decreases
    following a cosine curve.

    Args:
        optimizer: Optimizer to schedule.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.
        num_cycles: Number of cosine cycles (0.5 = half cycle).
        min_lr_ratio: Minimum LR as ratio of initial LR.

    Returns:
        Learning rate scheduler.

    Example:
        >>> scheduler = get_cosine_schedule_with_warmup(
        ...     optimizer, num_warmup_steps=100, num_training_steps=10000
        ... )
    """
    def lr_lambda(current_step: int) -> float:
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))

        # Scale to min_lr_ratio
        return max(min_lr_ratio, min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay)

    return LambdaLR(optimizer, lr_lambda)


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    """
    Create a schedule with linear warmup and linear decay.

    Args:
        optimizer: Optimizer to schedule.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.

    Returns:
        Learning rate scheduler.

    Example:
        >>> scheduler = get_linear_schedule_with_warmup(
        ...     optimizer, num_warmup_steps=100, num_training_steps=10000
        ... )
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda)


def get_constant_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
) -> LambdaLR:
    """
    Create a schedule with linear warmup and constant LR.

    Args:
        optimizer: Optimizer to schedule.
        num_warmup_steps: Number of warmup steps.

    Returns:
        Learning rate scheduler.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


class WarmupCosineScheduler(_LRScheduler):
    """
    Learning rate scheduler with warmup and cosine decay.

    More flexible than LambdaLR-based schedulers.

    Args:
        optimizer: Optimizer to schedule.
        warmup_steps: Number of warmup steps.
        total_steps: Total number of training steps.
        min_lr: Minimum learning rate.
        last_epoch: Last epoch (for resuming).

    Example:
        >>> scheduler = WarmupCosineScheduler(
        ...     optimizer, warmup_steps=100, total_steps=10000, min_lr=1e-6
        ... )
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self._step_count < self.warmup_steps:
            # Linear warmup
            warmup_factor = float(self._step_count) / float(max(1, self.warmup_steps))
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = float(self._step_count - self.warmup_steps) / float(
                max(1, self.total_steps - self.warmup_steps)
            )
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_factor
                for base_lr in self.base_lrs
            ]


def get_scheduler(
    name: str,
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    **kwargs,
) -> _LRScheduler:
    """
    Factory function to create learning rate schedulers.

    Args:
        name: Scheduler name ('cosine', 'linear', 'constant').
        optimizer: Optimizer to schedule.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total training steps.
        **kwargs: Additional scheduler arguments.

    Returns:
        Learning rate scheduler.

    Example:
        >>> scheduler = get_scheduler(
        ...     "cosine", optimizer,
        ...     num_warmup_steps=100, num_training_steps=10000
        ... )
    """
    if name == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps, **kwargs
        )
    elif name == "linear":
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    elif name == "constant":
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps)
    else:
        raise ValueError(f"Unknown scheduler: {name}")
