"""
Attention mechanisms for transformer models.

This module implements:
- Scaled dot-product attention
- Multi-head attention
- Causal (masked) attention for autoregressive models
- Grouped-query attention (GQA) for efficiency
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
    training: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        query: Query tensor of shape (batch, n_heads, seq_len, d_k)
        key: Key tensor of shape (batch, n_heads, seq_len, d_k)
        value: Value tensor of shape (batch, n_heads, seq_len, d_v)
        mask: Optional attention mask. Use -inf for masked positions.
              Shape: (batch, 1, seq_len, seq_len) or (1, 1, seq_len, seq_len)
        dropout: Dropout probability for attention weights.
        training: Whether model is in training mode.

    Returns:
        Tuple of:
        - Output tensor of shape (batch, n_heads, seq_len, d_v)
        - Attention weights of shape (batch, n_heads, seq_len, seq_len)

    Example:
        >>> q = torch.randn(2, 8, 10, 64)  # batch=2, heads=8, seq=10, d_k=64
        >>> k = torch.randn(2, 8, 10, 64)
        >>> v = torch.randn(2, 8, 10, 64)
        >>> output, attn_weights = scaled_dot_product_attention(q, k, v)
    """
    d_k = query.size(-1)

    # Compute attention scores: (batch, n_heads, seq_len, seq_len)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # Softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)

    # Apply dropout
    if dropout > 0.0 and training:
        attention_weights = F.dropout(attention_weights, p=dropout, training=training)

    # Compute output
    output = torch.matmul(attention_weights, value)

    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    Allows the model to jointly attend to information from different
    representation subspaces at different positions.

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        dropout: Dropout probability.
        bias: Whether to use bias in projections.

    Example:
        >>> mha = MultiHeadAttention(d_model=512, n_heads=8)
        >>> x = torch.randn(2, 10, 512)  # batch=2, seq=10, d_model=512
        >>> output, attn_weights = mha(x, x, x)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        self.dropout = dropout

        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)

        # Output projection
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

        # Store attention weights for visualization
        self._attention_weights = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of multi-head attention.

        Args:
            query: Query tensor of shape (batch, seq_len, d_model)
            key: Key tensor of shape (batch, seq_len, d_model)
            value: Value tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask.
            return_attention: If True, return attention weights.

        Returns:
            Tuple of:
            - Output tensor of shape (batch, seq_len, d_model)
            - Attention weights if return_attention=True, else None
        """
        batch_size = query.size(0)

        # Linear projections and reshape to (batch, n_heads, seq_len, d_k)
        q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Compute attention
        output, attention_weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.dropout, training=self.training
        )

        # Store for visualization
        self._attention_weights = attention_weights.detach()

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(output)

        if return_attention:
            return output, attention_weights
        return output, None

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return the last computed attention weights."""
        return self._attention_weights


class CausalSelfAttention(nn.Module):
    """
    Causal (masked) self-attention for autoregressive models.

    Automatically applies a causal mask to prevent attending to future tokens.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        max_seq_len: Maximum sequence length for causal mask.
        dropout: Dropout probability.
        bias: Whether to use bias in projections.

    Example:
        >>> attn = CausalSelfAttention(d_model=256, n_heads=4, max_seq_len=512)
        >>> x = torch.randn(2, 10, 256)
        >>> output, _ = attn(x)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        self.mha = MultiHeadAttention(d_model, n_heads, dropout, bias)
        self.max_seq_len = max_seq_len

        # Register causal mask as buffer (not a parameter)
        # Lower triangular matrix: 1s where we can attend, 0s where we mask
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, max_seq_len, max_seq_len))

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with automatic causal masking.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            return_attention: If True, return attention weights.

        Returns:
            Tuple of output and optional attention weights.
        """
        seq_len = x.size(1)

        # Get causal mask for current sequence length
        mask = self.causal_mask[:, :, :seq_len, :seq_len]

        return self.mha(x, x, x, mask=mask, return_attention=return_attention)

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return the last computed attention weights."""
        return self.mha.get_attention_weights()


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) for efficient inference.

    Uses fewer key-value heads than query heads, reducing memory bandwidth
    requirements during inference while maintaining most of the quality.

    From "GQA: Training Generalized Multi-Query Transformer Models from
    Multi-Head Checkpoints" (Ainslie et al., 2023)

    Args:
        d_model: Model dimension.
        n_heads: Number of query heads.
        n_kv_heads: Number of key-value heads (must divide n_heads evenly).
        dropout: Dropout probability.
        bias: Whether to use bias in projections.

    Example:
        >>> # 8 query heads, 2 KV heads (each KV head shared by 4 query heads)
        >>> gqa = GroupedQueryAttention(d_model=512, n_heads=8, n_kv_heads=2)
        >>> x = torch.randn(2, 10, 512)
        >>> output, _ = gqa(x, x, x)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # Number of times to repeat KV heads
        self.d_k = d_model // n_heads
        self.dropout = dropout

        # Query projection (full n_heads)
        self.w_q = nn.Linear(d_model, d_model, bias=bias)

        # Key-Value projections (reduced n_kv_heads)
        kv_dim = self.d_k * n_kv_heads
        self.w_k = nn.Linear(d_model, kv_dim, bias=bias)
        self.w_v = nn.Linear(d_model, kv_dim, bias=bias)

        # Output projection
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

        self._attention_weights = None

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match number of query heads."""
        batch, n_kv_heads, seq_len, d_k = x.shape
        if self.n_rep == 1:
            return x
        # (batch, n_kv_heads, seq_len, d_k) -> (batch, n_heads, seq_len, d_k)
        return (
            x[:, :, None, :, :]
            .expand(batch, n_kv_heads, self.n_rep, seq_len, d_k)
            .reshape(batch, self.n_heads, seq_len, d_k)
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of grouped-query attention.

        Args:
            query: Query tensor of shape (batch, seq_len, d_model)
            key: Key tensor of shape (batch, seq_len, d_model)
            value: Value tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask.
            return_attention: If True, return attention weights.

        Returns:
            Tuple of output and optional attention weights.
        """
        batch_size, seq_len, _ = query.shape

        # Project queries (full n_heads)
        q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Project keys and values (reduced n_kv_heads)
        k = self.w_k(key).view(batch_size, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)

        # Repeat KV heads to match query heads
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # Compute attention
        output, attention_weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.dropout, training=self.training
        )

        self._attention_weights = attention_weights.detach()

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(output)

        if return_attention:
            return output, attention_weights
        return output, None

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return the last computed attention weights."""
        return self._attention_weights


def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Create a causal attention mask.

    Args:
        seq_len: Sequence length.
        device: Device to create mask on.

    Returns:
        Causal mask of shape (1, 1, seq_len, seq_len).
        1 = can attend, 0 = masked.
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.view(1, 1, seq_len, seq_len)


def create_padding_mask(
    lengths: torch.Tensor, max_len: int, device: torch.device = None
) -> torch.Tensor:
    """
    Create a padding mask from sequence lengths.

    Args:
        lengths: Tensor of sequence lengths, shape (batch,)
        max_len: Maximum sequence length.
        device: Device to create mask on.

    Returns:
        Padding mask of shape (batch, 1, 1, max_len).
        1 = valid token, 0 = padding.
    """
    batch_size = lengths.size(0)
    # Create range tensor and compare with lengths
    range_tensor = torch.arange(max_len, device=device).expand(batch_size, max_len)
    mask = range_tensor < lengths.unsqueeze(1)
    return mask.view(batch_size, 1, 1, max_len).float()
