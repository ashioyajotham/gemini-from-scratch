"""
Utility functions for the transformer workshop.

This module provides:
- Device management (CPU/CUDA/MPS)
- Configuration loading
- Checkpointing
- Metrics computation
- Visualization tools
"""

from .device import get_device, to_device, get_device_info
from .config import load_config, merge_configs, Config
from .helpers import set_seed, count_parameters, get_logger, format_number
from .checkpointing import save_checkpoint, load_checkpoint
from .metrics import compute_perplexity, compute_accuracy, compute_top_k_accuracy
from .visualization import plot_attention_weights, plot_loss_curve, plot_embeddings_tsne

__all__ = [
    # Device
    "get_device",
    "to_device",
    "get_device_info",
    # Config
    "load_config",
    "merge_configs",
    "Config",
    # Helpers
    "set_seed",
    "count_parameters",
    "get_logger",
    "format_number",
    # Checkpointing
    "save_checkpoint",
    "load_checkpoint",
    # Metrics
    "compute_perplexity",
    "compute_accuracy",
    "compute_top_k_accuracy",
    # Visualization
    "plot_attention_weights",
    "plot_loss_curve",
    "plot_embeddings_tsne",
]
