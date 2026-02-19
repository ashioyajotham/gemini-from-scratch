"""Model checkpointing utilities for saving and loading models."""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn
from torch.optim import Optimizer


def save_checkpoint(
    model: nn.Module,
    path: Union[str, Path],
    optimizer: Optional[Optimizer] = None,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    loss: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any
) -> None:
    """
    Save a model checkpoint with optional training state.

    Args:
        model: PyTorch model to save.
        path: Path to save the checkpoint.
        optimizer: Optional optimizer to save.
        epoch: Current epoch number.
        step: Current training step.
        loss: Current loss value.
        config: Model/training configuration.
        **kwargs: Additional metadata to save.

    Example:
        >>> save_checkpoint(
        ...     model=model,
        ...     path="checkpoints/model_step_1000.pt",
        ...     optimizer=optimizer,
        ...     step=1000,
        ...     loss=2.5,
        ...     config=config.to_dict()
        ... )
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "config": config,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    # Add any additional metadata
    checkpoint.update(kwargs)

    torch.save(checkpoint, path)


def load_checkpoint(
    path: Union[str, Path],
    model: Optional[nn.Module] = None,
    optimizer: Optional[Optimizer] = None,
    device: Optional[torch.device] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Load a model checkpoint.

    Args:
        path: Path to the checkpoint file.
        model: Model to load weights into. If None, returns state dict.
        optimizer: Optimizer to load state into.
        device: Device to load checkpoint onto.
        strict: If True, raises error for missing/unexpected keys.

    Returns:
        Dictionary containing checkpoint data (epoch, step, loss, config, etc.)

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.

    Example:
        >>> checkpoint = load_checkpoint(
        ...     "checkpoints/model_step_1000.pt",
        ...     model=model,
        ...     optimizer=optimizer,
        ...     device=device
        ... )
        >>> print(f"Resuming from step {checkpoint['step']}")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Load checkpoint
    if device is not None:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    else:
        checkpoint = torch.load(path, weights_only=False)

    # Load model weights
    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Return metadata
    return {
        "epoch": checkpoint.get("epoch"),
        "step": checkpoint.get("step"),
        "loss": checkpoint.get("loss"),
        "config": checkpoint.get("config"),
    }


def get_latest_checkpoint(checkpoint_dir: Union[str, Path]) -> Optional[Path]:
    """
    Find the latest checkpoint in a directory.

    Assumes checkpoints are named with step numbers (e.g., model_step_1000.pt).

    Args:
        checkpoint_dir: Directory containing checkpoints.

    Returns:
        Path to the latest checkpoint, or None if no checkpoints found.

    Example:
        >>> latest = get_latest_checkpoint("checkpoints/")
        >>> if latest:
        ...     load_checkpoint(latest, model)
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        return None

    # Sort by modification time
    checkpoints.sort(key=lambda p: p.stat().st_mtime)
    return checkpoints[-1]
