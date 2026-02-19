"""Helper utilities for reproducibility, logging, and general tasks."""

import random
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value.

    Example:
        >>> set_seed(42)
        >>> # Now all random operations will be reproducible
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model.

    Args:
        model: PyTorch model.
        trainable_only: If True, count only trainable parameters.

    Returns:
        Total number of parameters.

    Example:
        >>> model = MyTransformer()
        >>> print(f"Parameters: {count_parameters(model):,}")
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_number(num: int) -> str:
    """
    Format a large number with K/M/B suffixes.

    Args:
        num: Number to format.

    Returns:
        Formatted string (e.g., "1.5M", "256K").

    Example:
        >>> format_number(1_500_000)
        '1.50M'
    """
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return str(num)


def get_logger(
    name: str,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Create and configure a logger.

    Args:
        name: Logger name (typically __name__).
        level: Logging level.
        format_string: Custom format string for log messages.

    Returns:
        Configured logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Training started")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers if they already exist
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)

        if format_string is None:
            format_string = "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"

        formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def print_model_summary(model: nn.Module, input_shape: Optional[tuple] = None) -> None:
    """
    Print a summary of the model architecture.

    Args:
        model: PyTorch model.
        input_shape: Optional input shape for forward pass simulation.

    Example:
        >>> print_model_summary(model)
    """
    print("=" * 60)
    print(f"Model: {model.__class__.__name__}")
    print("=" * 60)

    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)

    print(f"Total parameters:     {total_params:>12,} ({format_number(total_params)})")
    print(f"Trainable parameters: {trainable_params:>12,} ({format_number(trainable_params)})")
    print(f"Non-trainable:        {total_params - trainable_params:>12,}")
    print("=" * 60)

    # Print layer-by-layer breakdown
    print("\nLayer breakdown:")
    print("-" * 60)
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                print(f"  {name:<40} {params:>12,}")
    print("-" * 60)
