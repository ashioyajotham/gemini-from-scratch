"""
Training utilities and loop implementation.

This module contains:
- Training loop (Trainer class)
- Optimizers and schedulers
- Loss functions
- Training callbacks
"""

from .optimizer import (
    configure_optimizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_scheduler,
    WarmupCosineScheduler,
)

from .losses import (
    CrossEntropyLoss,
    LanguageModelingLoss,
    compute_loss_with_mask,
    compute_perplexity,
    FocalLoss,
)

from .callbacks import (
    Callback,
    CallbackList,
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggingCallback,
    LRSchedulerCallback,
    GradientClippingCallback,
)

from .trainer import (
    TrainingConfig,
    Trainer,
    train_model,
)

__all__ = [
    # Optimizer
    "configure_optimizer",
    "get_cosine_schedule_with_warmup",
    "get_linear_schedule_with_warmup",
    "get_constant_schedule_with_warmup",
    "get_scheduler",
    "WarmupCosineScheduler",
    # Losses
    "CrossEntropyLoss",
    "LanguageModelingLoss",
    "compute_loss_with_mask",
    "compute_perplexity",
    "FocalLoss",
    # Callbacks
    "Callback",
    "CallbackList",
    "CheckpointCallback",
    "EarlyStoppingCallback",
    "LoggingCallback",
    "LRSchedulerCallback",
    "GradientClippingCallback",
    # Trainer
    "TrainingConfig",
    "Trainer",
    "train_model",
]
