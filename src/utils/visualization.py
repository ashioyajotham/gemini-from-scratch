"""Visualization utilities for attention, embeddings, and training metrics."""

from typing import List, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_attention_weights(
    attention: np.ndarray,
    tokens: Optional[List[str]] = None,
    title: str = "Attention Weights",
    figsize: tuple = (10, 8),
    cmap: str = "viridis",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize attention weights as a heatmap.

    Args:
        attention: Attention weights of shape (seq_len, seq_len) or (n_heads, seq_len, seq_len).
        tokens: Optional list of token strings for axis labels.
        title: Plot title.
        figsize: Figure size (width, height).
        cmap: Colormap for heatmap.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib figure object.

    Example:
        >>> # Single head attention
        >>> attn = model.get_attention_weights()  # (seq_len, seq_len)
        >>> fig = plot_attention_weights(attn, tokens=["The", "cat", "sat"])

        >>> # Multi-head attention
        >>> attn = model.get_attention_weights()  # (n_heads, seq_len, seq_len)
        >>> fig = plot_attention_weights(attn, tokens=tokens)
    """
    if attention.ndim == 3:
        # Multi-head attention: create subplots
        n_heads = attention.shape[0]
        n_cols = min(4, n_heads)
        n_rows = (n_heads + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = np.array(axes).flatten()

        for head_idx in range(n_heads):
            ax = axes[head_idx]
            sns.heatmap(
                attention[head_idx],
                ax=ax,
                cmap=cmap,
                xticklabels=tokens if tokens else False,
                yticklabels=tokens if tokens else False,
                square=True
            )
            ax.set_title(f"Head {head_idx + 1}")

        # Hide unused subplots
        for idx in range(n_heads, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(title, fontsize=14)
        plt.tight_layout()

    else:
        # Single attention matrix
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            attention,
            ax=ax,
            cmap=cmap,
            xticklabels=tokens if tokens else False,
            yticklabels=tokens if tokens else False,
            square=True,
            cbar_kws={"label": "Attention Weight"}
        )
        ax.set_title(title)
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
        plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_loss_curve(
    losses: List[float],
    title: str = "Training Loss",
    xlabel: str = "Step",
    ylabel: str = "Loss",
    figsize: tuple = (10, 6),
    smoothing: float = 0.0,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training loss curve.

    Args:
        losses: List of loss values.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Figure size.
        smoothing: Exponential smoothing factor (0 = no smoothing, 0.9 = heavy smoothing).
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib figure object.

    Example:
        >>> losses = [3.5, 3.2, 2.9, 2.7, 2.5]
        >>> fig = plot_loss_curve(losses, smoothing=0.6)
    """
    fig, ax = plt.subplots(figsize=figsize)

    steps = list(range(1, len(losses) + 1))

    # Plot raw losses with transparency
    ax.plot(steps, losses, alpha=0.3, color="blue", label="Raw")

    # Apply exponential smoothing if requested
    if smoothing > 0:
        smoothed = []
        current = losses[0]
        for loss in losses:
            current = smoothing * current + (1 - smoothing) * loss
            smoothed.append(current)
        ax.plot(steps, smoothed, color="blue", linewidth=2, label="Smoothed")
        ax.legend()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_embeddings_tsne(
    embeddings: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Token Embeddings (t-SNE)",
    figsize: tuple = (12, 10),
    perplexity: int = 30,
    n_iter: int = 1000,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize embeddings using t-SNE dimensionality reduction.

    Args:
        embeddings: Embedding matrix of shape (n_tokens, d_model).
        labels: Optional list of labels for each embedding.
        title: Plot title.
        figsize: Figure size.
        perplexity: t-SNE perplexity parameter.
        n_iter: Number of t-SNE iterations.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib figure object.

    Example:
        >>> embeddings = model.get_token_embeddings()  # (vocab_size, d_model)
        >>> tokens = tokenizer.get_vocab()
        >>> fig = plot_embeddings_tsne(embeddings[:100], labels=tokens[:100])
    """
    from sklearn.manifold import TSNE

    # Reduce dimensionality
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot points
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=20)

    # Add labels if provided
    if labels:
        for i, label in enumerate(labels):
            ax.annotate(
                label,
                (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                fontsize=8,
                alpha=0.7
            )

    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_training_progress(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    val_steps: Optional[List[int]] = None,
    title: str = "Training Progress",
    figsize: tuple = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training and validation loss curves side by side with perplexity.

    Args:
        train_losses: List of training loss values.
        val_losses: Optional list of validation loss values.
        val_steps: Steps at which validation was performed.
        title: Plot title.
        figsize: Figure size.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left plot: Loss
    ax1 = axes[0]
    train_steps = list(range(1, len(train_losses) + 1))
    ax1.plot(train_steps, train_losses, label="Train", alpha=0.7)

    if val_losses:
        if val_steps is None:
            val_steps = list(range(1, len(val_losses) + 1))
        ax1.plot(val_steps, val_losses, label="Validation", marker="o")

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right plot: Perplexity
    ax2 = axes[1]
    train_ppl = [np.exp(loss) for loss in train_losses]
    ax2.plot(train_steps, train_ppl, label="Train", alpha=0.7)

    if val_losses:
        val_ppl = [np.exp(loss) for loss in val_losses]
        ax2.plot(val_steps, val_ppl, label="Validation", marker="o")

    ax2.set_xlabel("Step")
    ax2.set_ylabel("Perplexity")
    ax2.set_title("Perplexity")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
