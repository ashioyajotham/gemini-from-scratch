"""
Training loop implementation.

This module implements:
- Trainer class for model training
- Training and evaluation loops
- Gradient accumulation
- Mixed precision training support
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from .optimizer import configure_optimizer, get_scheduler
from .losses import LanguageModelingLoss
from .callbacks import Callback, CallbackList, LoggingCallback, LRSchedulerCallback


@dataclass
class TrainingConfig:
    """
    Configuration for training.

    Args:
        max_steps: Maximum number of training steps.
        learning_rate: Peak learning rate.
        weight_decay: Weight decay coefficient.
        warmup_steps: Number of warmup steps.
        gradient_clip: Maximum gradient norm (None to disable).
        gradient_accumulation_steps: Accumulate gradients over N steps.
        eval_every_n_steps: Evaluate every N steps.
        log_every_n_steps: Log metrics every N steps.
        save_every_n_steps: Save checkpoint every N steps.
        mixed_precision: Use mixed precision training (fp16).
        scheduler_type: LR scheduler type ('cosine', 'linear', 'constant').
        label_smoothing: Label smoothing factor.
    """

    max_steps: int = 10000
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    gradient_clip: Optional[float] = 1.0
    gradient_accumulation_steps: int = 1
    eval_every_n_steps: int = 500
    log_every_n_steps: int = 100
    save_every_n_steps: int = 1000
    mixed_precision: bool = False
    scheduler_type: str = "cosine"
    label_smoothing: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "gradient_clip": self.gradient_clip,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "eval_every_n_steps": self.eval_every_n_steps,
            "log_every_n_steps": self.log_every_n_steps,
            "save_every_n_steps": self.save_every_n_steps,
            "mixed_precision": self.mixed_precision,
            "scheduler_type": self.scheduler_type,
            "label_smoothing": self.label_smoothing,
        }


class Trainer:
    """
    Trainer class for language model training.

    Handles the training loop, evaluation, checkpointing, and callbacks.

    Args:
        model: Model to train.
        config: Training configuration.
        train_dataloader: Training data loader.
        eval_dataloader: Optional evaluation data loader.
        callbacks: List of training callbacks.
        device: Device to train on.

    Example:
        >>> trainer = Trainer(
        ...     model=model,
        ...     config=TrainingConfig(max_steps=10000),
        ...     train_dataloader=train_loader,
        ...     eval_dataloader=val_loader,
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        callbacks: Optional[List[Callback]] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # Device setup
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(self.device)

        # Optimizer and scheduler
        self.optimizer = configure_optimizer(
            model,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scheduler = get_scheduler(
            config.scheduler_type,
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.max_steps,
        )

        # Loss function
        self.loss_fn = LanguageModelingLoss(label_smoothing=config.label_smoothing)

        # Mixed precision
        self.scaler = GradScaler() if config.mixed_precision else None

        # Callbacks
        default_callbacks = [
            LoggingCallback(log_every_n_steps=config.log_every_n_steps),
            LRSchedulerCallback(),
        ]
        all_callbacks = default_callbacks + (callbacks or [])
        self.callbacks = CallbackList(all_callbacks)

        # Training state
        self.global_step = 0
        self.should_stop = False
        self.train_loss_history = []
        self.eval_loss_history = []

    def train(self) -> Dict[str, Any]:
        """
        Run the training loop.

        Returns:
            Dictionary with training metrics.
        """
        self.callbacks.on_train_begin(self)

        self.model.train()
        data_iter = iter(self.train_dataloader)

        accumulated_loss = 0.0
        num_accumulated = 0

        progress_bar = tqdm(total=self.config.max_steps, desc="Training")

        while self.global_step < self.config.max_steps and not self.should_stop:
            self.callbacks.on_step_begin(self, self.global_step)

            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)

            # Training step
            step_loss = self._training_step(batch)
            accumulated_loss += step_loss
            num_accumulated += 1

            # Gradient accumulation
            if num_accumulated >= self.config.gradient_accumulation_steps:
                # Gradient clipping
                if self.config.gradient_clip is not None:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip
                    )

                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

                # Compute average loss
                avg_loss = accumulated_loss / num_accumulated
                self.train_loss_history.append(avg_loss)

                # Log
                logs = {"loss": avg_loss, "step": self.global_step}
                self.callbacks.on_step_end(self, self.global_step, logs)

                # Update progress bar
                progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
                progress_bar.update(1)

                # Reset accumulation
                accumulated_loss = 0.0
                num_accumulated = 0
                self.global_step += 1

                # Evaluation
                if (
                    self.eval_dataloader is not None
                    and self.global_step % self.config.eval_every_n_steps == 0
                ):
                    eval_logs = self.evaluate()
                    self.callbacks.on_evaluate_end(self, eval_logs)
                    self.model.train()

        progress_bar.close()
        self.callbacks.on_train_end(self)

        return {
            "final_loss": self.train_loss_history[-1] if self.train_loss_history else None,
            "train_loss_history": self.train_loss_history,
            "eval_loss_history": self.eval_loss_history,
            "steps_trained": self.global_step,
        }

    def _training_step(self, batch) -> float:
        """
        Execute a single training step.

        Args:
            batch: Batch of data (input_ids, target_ids).

        Returns:
            Loss value for this step.
        """
        input_ids, target_ids = batch
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)

        # Forward pass with optional mixed precision
        if self.config.mixed_precision:
            with autocast():
                logits, _ = self.model(input_ids)
                loss = self.loss_fn(logits, target_ids, shift=False)
                loss = loss / self.config.gradient_accumulation_steps
        else:
            logits, _ = self.model(input_ids)
            loss = self.loss_fn(logits, target_ids, shift=False)
            loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.item() * self.config.gradient_accumulation_steps

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on the evaluation dataset.

        Returns:
            Dictionary with evaluation metrics.
        """
        if self.eval_dataloader is None:
            return {}

        self.callbacks.on_evaluate_begin(self)
        self.model.eval()

        total_loss = 0.0
        total_tokens = 0

        for batch in tqdm(self.eval_dataloader, desc="Evaluating", leave=False):
            input_ids, target_ids = batch
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            logits, _ = self.model(input_ids)
            loss = self.loss_fn(logits, target_ids, shift=False)

            # Count non-padding tokens
            num_tokens = (target_ids != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        self.eval_loss_history.append(avg_loss)

        return {
            "val_loss": avg_loss,
            "val_perplexity": perplexity,
            "step": self.global_step,
        }

    def save_checkpoint(self, path: str) -> None:
        """
        Save a training checkpoint.

        Args:
            path: Path to save checkpoint.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_loss_history": self.train_loss_history,
            "eval_loss_history": self.eval_loss_history,
            "config": self.config.to_dict(),
        }

        if hasattr(self.model, "config"):
            checkpoint["model_config"] = self.model.config.to_dict()

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """
        Load a training checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.global_step = checkpoint["global_step"]
        self.train_loss_history = checkpoint.get("train_loss_history", [])
        self.eval_loss_history = checkpoint.get("eval_loss_history", [])

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])


def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    eval_dataloader: Optional[DataLoader] = None,
    max_steps: int = 10000,
    learning_rate: float = 3e-4,
    warmup_steps: int = 100,
    device: Optional[torch.device] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function to train a model.

    Args:
        model: Model to train.
        train_dataloader: Training data loader.
        eval_dataloader: Optional evaluation data loader.
        max_steps: Maximum training steps.
        learning_rate: Learning rate.
        warmup_steps: Warmup steps.
        device: Training device.
        **kwargs: Additional TrainingConfig arguments.

    Returns:
        Training results dictionary.

    Example:
        >>> results = train_model(
        ...     model, train_loader,
        ...     max_steps=5000, learning_rate=1e-4
        ... )
    """
    config = TrainingConfig(
        max_steps=max_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        **kwargs,
    )

    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        device=device,
    )

    return trainer.train()
