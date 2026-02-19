"""
Training callbacks for monitoring and control.

This module implements:
- Base callback interface
- Checkpoint saving
- Early stopping
- Learning rate logging
- Progress tracking
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class Callback(ABC):
    """
    Base class for training callbacks.

    Callbacks can be used to perform actions at various points
    during training (start, end, each step, each epoch, etc.).
    """

    def on_train_begin(self, trainer: "Trainer") -> None:
        """Called at the start of training."""
        pass

    def on_train_end(self, trainer: "Trainer") -> None:
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, trainer: "Trainer", epoch: int) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, trainer: "Trainer", epoch: int, logs: Dict[str, Any]) -> None:
        """Called at the end of each epoch."""
        pass

    def on_step_begin(self, trainer: "Trainer", step: int) -> None:
        """Called at the start of each training step."""
        pass

    def on_step_end(self, trainer: "Trainer", step: int, logs: Dict[str, Any]) -> None:
        """Called at the end of each training step."""
        pass

    def on_evaluate_begin(self, trainer: "Trainer") -> None:
        """Called at the start of evaluation."""
        pass

    def on_evaluate_end(self, trainer: "Trainer", logs: Dict[str, Any]) -> None:
        """Called at the end of evaluation."""
        pass


class CallbackList:
    """
    Container for managing multiple callbacks.

    Example:
        >>> callbacks = CallbackList([
        ...     CheckpointCallback(save_dir="checkpoints"),
        ...     EarlyStoppingCallback(patience=5),
        ... ])
    """

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []

    def add(self, callback: Callback) -> None:
        """Add a callback."""
        self.callbacks.append(callback)

    def on_train_begin(self, trainer: "Trainer") -> None:
        for callback in self.callbacks:
            callback.on_train_begin(trainer)

    def on_train_end(self, trainer: "Trainer") -> None:
        for callback in self.callbacks:
            callback.on_train_end(trainer)

    def on_epoch_begin(self, trainer: "Trainer", epoch: int) -> None:
        for callback in self.callbacks:
            callback.on_epoch_begin(trainer, epoch)

    def on_epoch_end(self, trainer: "Trainer", epoch: int, logs: Dict[str, Any]) -> None:
        for callback in self.callbacks:
            callback.on_epoch_end(trainer, epoch, logs)

    def on_step_begin(self, trainer: "Trainer", step: int) -> None:
        for callback in self.callbacks:
            callback.on_step_begin(trainer, step)

    def on_step_end(self, trainer: "Trainer", step: int, logs: Dict[str, Any]) -> None:
        for callback in self.callbacks:
            callback.on_step_end(trainer, step, logs)

    def on_evaluate_begin(self, trainer: "Trainer") -> None:
        for callback in self.callbacks:
            callback.on_evaluate_begin(trainer)

    def on_evaluate_end(self, trainer: "Trainer", logs: Dict[str, Any]) -> None:
        for callback in self.callbacks:
            callback.on_evaluate_end(trainer, logs)


class CheckpointCallback(Callback):
    """
    Save model checkpoints during training.

    Args:
        save_dir: Directory to save checkpoints.
        save_every_n_steps: Save every N steps.
        save_best_only: If True, only save when validation improves.
        monitor: Metric to monitor ('loss', 'val_loss', etc.).
        mode: 'min' or 'max' for the monitored metric.
        max_checkpoints: Maximum number of checkpoints to keep.

    Example:
        >>> callback = CheckpointCallback(
        ...     save_dir="checkpoints",
        ...     save_every_n_steps=1000,
        ...     save_best_only=True,
        ...     monitor="val_loss",
        ... )
    """

    def __init__(
        self,
        save_dir: str,
        save_every_n_steps: int = 1000,
        save_best_only: bool = False,
        monitor: str = "val_loss",
        mode: str = "min",
        max_checkpoints: int = 5,
    ):
        self.save_dir = Path(save_dir)
        self.save_every_n_steps = save_every_n_steps
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.max_checkpoints = max_checkpoints

        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.saved_checkpoints = []

    def on_train_begin(self, trainer: "Trainer") -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_step_end(self, trainer: "Trainer", step: int, logs: Dict[str, Any]) -> None:
        if step > 0 and step % self.save_every_n_steps == 0:
            if not self.save_best_only:
                self._save_checkpoint(trainer, step, logs)

    def on_evaluate_end(self, trainer: "Trainer", logs: Dict[str, Any]) -> None:
        if self.save_best_only and self.monitor in logs:
            value = logs[self.monitor]
            is_better = (
                value < self.best_value
                if self.mode == "min"
                else value > self.best_value
            )

            if is_better:
                self.best_value = value
                self._save_checkpoint(trainer, trainer.global_step, logs, is_best=True)

    def _save_checkpoint(
        self,
        trainer: "Trainer",
        step: int,
        logs: Dict[str, Any],
        is_best: bool = False,
    ) -> None:
        """Save a checkpoint."""
        if is_best:
            filename = "best_model.pt"
        else:
            filename = f"checkpoint_step_{step}.pt"

        path = self.save_dir / filename

        checkpoint = {
            "step": step,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "logs": logs,
        }

        if hasattr(trainer, "scheduler") and trainer.scheduler is not None:
            checkpoint["scheduler_state_dict"] = trainer.scheduler.state_dict()

        if hasattr(trainer.model, "config"):
            checkpoint["config"] = trainer.model.config.to_dict()

        torch.save(checkpoint, path)

        # Track checkpoints for cleanup
        if not is_best:
            self.saved_checkpoints.append(path)
            self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints exceeding max_checkpoints."""
        while len(self.saved_checkpoints) > self.max_checkpoints:
            oldest = self.saved_checkpoints.pop(0)
            if oldest.exists():
                oldest.unlink()


class EarlyStoppingCallback(Callback):
    """
    Stop training when a metric stops improving.

    Args:
        monitor: Metric to monitor.
        patience: Number of evaluations with no improvement before stopping.
        mode: 'min' or 'max' for the monitored metric.
        min_delta: Minimum change to qualify as an improvement.

    Example:
        >>> callback = EarlyStoppingCallback(
        ...     monitor="val_loss",
        ...     patience=5,
        ...     mode="min",
        ... )
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 5,
        mode: str = "min",
        min_delta: float = 0.0,
    ):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.should_stop = False

    def on_evaluate_end(self, trainer: "Trainer", logs: Dict[str, Any]) -> None:
        if self.monitor not in logs:
            return

        value = logs[self.monitor]

        if self.mode == "min":
            is_better = value < (self.best_value - self.min_delta)
        else:
            is_better = value > (self.best_value + self.min_delta)

        if is_better:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                trainer.should_stop = True


class LoggingCallback(Callback):
    """
    Log training metrics.

    Args:
        log_every_n_steps: Log every N steps.
        logger: Logger instance (if None, prints to stdout).

    Example:
        >>> callback = LoggingCallback(log_every_n_steps=100)
    """

    def __init__(self, log_every_n_steps: int = 100, logger=None):
        self.log_every_n_steps = log_every_n_steps
        self.logger = logger

    def on_step_end(self, trainer: "Trainer", step: int, logs: Dict[str, Any]) -> None:
        if step > 0 and step % self.log_every_n_steps == 0:
            self._log(f"Step {step}: {self._format_logs(logs)}")

    def on_evaluate_end(self, trainer: "Trainer", logs: Dict[str, Any]) -> None:
        self._log(f"Evaluation: {self._format_logs(logs)}")

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def _format_logs(self, logs: Dict[str, Any]) -> str:
        items = []
        for key, value in logs.items():
            if isinstance(value, float):
                items.append(f"{key}={value:.4f}")
            else:
                items.append(f"{key}={value}")
        return ", ".join(items)


class LRSchedulerCallback(Callback):
    """
    Step the learning rate scheduler.

    Automatically steps the scheduler after each training step.
    """

    def on_step_end(self, trainer: "Trainer", step: int, logs: Dict[str, Any]) -> None:
        if hasattr(trainer, "scheduler") and trainer.scheduler is not None:
            trainer.scheduler.step()
            # Add current LR to logs
            logs["lr"] = trainer.scheduler.get_last_lr()[0]


class GradientClippingCallback(Callback):
    """
    Clip gradients during training.

    Args:
        max_norm: Maximum gradient norm.
        norm_type: Type of norm (default: 2.0 for L2 norm).
    """

    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type

    def on_step_begin(self, trainer: "Trainer", step: int) -> None:
        # Note: Gradient clipping is typically done after backward, before optimizer step
        # This callback is mainly for configuration; actual clipping is in trainer
        pass
