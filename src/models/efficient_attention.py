"""
Efficient attention mechanisms for long sequences.

This module implements:
- Sliding window attention (local attention)
- Sparse attention patterns
- Linear attention approximations
- FlashAttention-style memory-efficient attention
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlidingWindowAttention(nn.Module):
    """
    Sliding window (local) attention.

    Each position can only attend to a fixed window of neighboring positions,
    reducing complexity from O(n²) to O(n * window_size).

    Used in Longformer, BigBird, and Mistral models.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        window_size: Size of the attention window (positions on each side).
        dropout: Dropout probability.
        bias: Whether to use bias in projections.

    Example:
        >>> attn = SlidingWindowAttention(d_model=512, n_heads=8, window_size=256)
        >>> x = torch.randn(2, 1024, 512)  # Long sequence
        >>> output, _ = attn(x)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int = 256,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.window_size = window_size
        self.dropout = dropout

        # Projections
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

        self._attention_weights = None

    def _create_sliding_window_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create a sliding window attention mask."""
        # Create position indices
        row_indices = torch.arange(seq_len, device=device).unsqueeze(1)
        col_indices = torch.arange(seq_len, device=device).unsqueeze(0)

        # Compute distances
        distances = torch.abs(row_indices - col_indices)

        # Create mask: 1 for positions within window, 0 otherwise
        mask = (distances <= self.window_size).float()

        # Also apply causal mask (optional, for autoregressive)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        mask = mask * causal_mask

        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with sliding window attention.

        Args:
            query: Query tensor of shape (batch, seq_len, d_model).
            key: Key tensor of shape (batch, seq_len, d_model).
            value: Value tensor of shape (batch, seq_len, d_model).
            return_attention: If True, return attention weights.

        Returns:
            Tuple of output and optional attention weights.
        """
        batch_size, seq_len, _ = query.shape
        device = query.device

        # Project
        q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply sliding window mask
        mask = self._create_sliding_window_mask(seq_len, device)
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        if self.dropout > 0 and self.training:
            attention_weights = F.dropout(attention_weights, p=self.dropout)

        self._attention_weights = attention_weights.detach()

        # Compute output
        output = torch.matmul(attention_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(output)

        if return_attention:
            return output, attention_weights
        return output, None

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return the last computed attention weights."""
        return self._attention_weights


class SparseAttention(nn.Module):
    """
    Sparse attention with configurable patterns.

    Supports:
    - Local attention (sliding window)
    - Global attention (selected tokens attend to all)
    - Random attention (for approximation)

    Similar to BigBird and Longformer patterns.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        local_window: Size of local attention window.
        global_tokens: Number of global tokens (from start).
        random_tokens: Number of random attention connections.
        dropout: Dropout probability.

    Example:
        >>> attn = SparseAttention(
        ...     d_model=512, n_heads=8,
        ...     local_window=64, global_tokens=2
        ... )
        >>> x = torch.randn(2, 512, 512)
        >>> output, _ = attn(x)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        local_window: int = 64,
        global_tokens: int = 1,
        random_tokens: int = 0,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.local_window = local_window
        self.global_tokens = global_tokens
        self.random_tokens = random_tokens
        self.dropout = dropout

        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

    def _create_sparse_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create a sparse attention mask."""
        mask = torch.zeros(seq_len, seq_len, device=device)

        # Local window attention
        for i in range(seq_len):
            start = max(0, i - self.local_window)
            end = min(seq_len, i + self.local_window + 1)
            mask[i, start:end] = 1

        # Global tokens (first N tokens attend to/from all)
        if self.global_tokens > 0:
            mask[:self.global_tokens, :] = 1  # Global tokens attend to all
            mask[:, :self.global_tokens] = 1  # All attend to global tokens

        # Causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        mask = mask * causal_mask

        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with sparse attention."""
        batch_size, seq_len, _ = query.shape
        device = query.device

        q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        mask = self._create_sparse_mask(seq_len, device)
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)
        if self.dropout > 0 and self.training:
            attention_weights = F.dropout(attention_weights, p=self.dropout)

        output = torch.matmul(attention_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(output)

        if return_attention:
            return output, attention_weights
        return output, None


class LinearAttention(nn.Module):
    """
    Linear attention approximation.

    Uses kernel feature maps to approximate softmax attention,
    reducing complexity from O(n²) to O(n).

    Based on "Transformers are RNNs" (Katharopoulos et al., 2020).

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        feature_map: Feature map type ('elu', 'relu', 'softmax').
        dropout: Dropout probability.

    Example:
        >>> attn = LinearAttention(d_model=512, n_heads=8)
        >>> x = torch.randn(2, 2048, 512)  # Very long sequence
        >>> output, _ = attn(x)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        feature_map: str = "elu",
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.feature_map = feature_map
        self.dropout = dropout

        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feature map to queries/keys."""
        if self.feature_map == "elu":
            return F.elu(x) + 1
        elif self.feature_map == "relu":
            return F.relu(x)
        elif self.feature_map == "softmax":
            return F.softmax(x, dim=-1)
        else:
            raise ValueError(f"Unknown feature map: {self.feature_map}")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with linear attention.

        Note: Does not return attention weights as they're not explicitly computed.
        """
        batch_size, seq_len, _ = query.shape

        q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Apply feature maps
        q = self._feature_map(q)
        k = self._feature_map(k)

        # Linear attention: (Q * (K^T * V)) instead of ((Q * K^T) * V)
        # For causal: use cumulative sums
        kv = torch.einsum("bhnd,bhnm->bhdm", k, v)  # (batch, heads, d_k, d_v)
        qkv = torch.einsum("bhnd,bhdm->bhnm", q, kv)  # (batch, heads, seq, d_v)

        # Normalize
        k_sum = k.sum(dim=2, keepdim=True)  # (batch, heads, 1, d_k)
        normalizer = torch.einsum("bhnd,bhkd->bhnk", q, k_sum).squeeze(-1)  # (batch, heads, seq)
        normalizer = normalizer.unsqueeze(-1) + 1e-6

        output = qkv / normalizer
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(output)

        # Linear attention doesn't produce explicit attention weights
        return output, None


class FlashAttentionSimulator(nn.Module):
    """
    Simulated FlashAttention for educational purposes.

    Demonstrates the tiling/chunking strategy used in FlashAttention
    to reduce memory usage. Not as fast as the real CUDA implementation,
    but useful for understanding the algorithm.

    Real FlashAttention should be used via torch.nn.functional.scaled_dot_product_attention
    with enable_flash=True (PyTorch 2.0+).

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        chunk_size: Size of chunks for tiled computation.
        dropout: Dropout probability.

    Example:
        >>> attn = FlashAttentionSimulator(d_model=512, n_heads=8, chunk_size=64)
        >>> x = torch.randn(2, 256, 512)
        >>> output, _ = attn(x)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        chunk_size: int = 64,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.chunk_size = chunk_size
        self.dropout = dropout

        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

    def _chunked_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        """
        Compute attention using chunked/tiled approach.

        This simulates the memory-efficient computation strategy of FlashAttention.
        """
        batch_size, n_heads, seq_len, d_k = q.shape
        device = q.device

        # Output accumulator
        output = torch.zeros_like(q)

        # Process query chunks
        for q_start in range(0, seq_len, self.chunk_size):
            q_end = min(q_start + self.chunk_size, seq_len)
            q_chunk = q[:, :, q_start:q_end, :]

            # Accumulators for this query chunk
            chunk_output = torch.zeros(
                batch_size, n_heads, q_end - q_start, d_k,
                device=device, dtype=q.dtype
            )
            chunk_max = torch.full(
                (batch_size, n_heads, q_end - q_start, 1),
                float("-inf"), device=device, dtype=q.dtype
            )
            chunk_sum = torch.zeros(
                batch_size, n_heads, q_end - q_start, 1,
                device=device, dtype=q.dtype
            )

            # Process key-value chunks
            k_end_limit = q_end if causal else seq_len
            for k_start in range(0, k_end_limit, self.chunk_size):
                k_end = min(k_start + self.chunk_size, k_end_limit)
                k_chunk = k[:, :, k_start:k_end, :]
                v_chunk = v[:, :, k_start:k_end, :]

                # Compute attention scores for this block
                scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) / math.sqrt(d_k)

                # Apply causal mask within this block
                if causal:
                    q_positions = torch.arange(q_start, q_end, device=device).unsqueeze(1)
                    k_positions = torch.arange(k_start, k_end, device=device).unsqueeze(0)
                    causal_mask = q_positions >= k_positions
                    scores = scores.masked_fill(~causal_mask, float("-inf"))

                # Online softmax update
                block_max = scores.max(dim=-1, keepdim=True)[0]
                new_max = torch.maximum(chunk_max, block_max)

                # Rescale previous accumulator
                scale_old = torch.exp(chunk_max - new_max)
                chunk_output = chunk_output * scale_old
                chunk_sum = chunk_sum * scale_old

                # Add new block contribution
                scale_new = torch.exp(scores - new_max)
                chunk_output = chunk_output + torch.matmul(scale_new, v_chunk)
                chunk_sum = chunk_sum + scale_new.sum(dim=-1, keepdim=True)

                chunk_max = new_max

            # Normalize
            output[:, :, q_start:q_end, :] = chunk_output / (chunk_sum + 1e-6)

        return output

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with chunked attention computation."""
        batch_size, seq_len, _ = query.shape

        q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        output = self._chunked_attention(q, k, v, causal=True)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(output)

        # Chunked attention doesn't store full attention matrix
        return output, None


def use_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = True,
    dropout: float = 0.0,
    training: bool = True,
) -> torch.Tensor:
    """
    Use PyTorch's native FlashAttention when available.

    This is the recommended way to get FlashAttention benefits.
    Falls back to standard attention if FlashAttention is not available.

    Args:
        query: Query tensor of shape (batch, n_heads, seq_len, d_k).
        key: Key tensor of shape (batch, n_heads, seq_len, d_k).
        value: Value tensor of shape (batch, n_heads, seq_len, d_v).
        is_causal: Whether to use causal masking.
        dropout: Dropout probability.
        training: Whether in training mode.

    Returns:
        Output tensor of shape (batch, n_heads, seq_len, d_v).

    Example:
        >>> output = use_flash_attention(q, k, v, is_causal=True)
    """
    # PyTorch 2.0+ has scaled_dot_product_attention with flash attention
    return F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=None,
        dropout_p=dropout if training else 0.0,
        is_causal=is_causal,
    )


class EfficientMultiHeadAttention(nn.Module):
    """
    Multi-head attention that automatically uses the most efficient implementation.

    Tries to use FlashAttention when available, falls back to standard attention.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        dropout: Dropout probability.
        bias: Whether to use bias.
        use_flash: Whether to try using FlashAttention.

    Example:
        >>> attn = EfficientMultiHeadAttention(d_model=512, n_heads=8)
        >>> x = torch.randn(2, 100, 512)
        >>> output, _ = attn(x, x, x)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        use_flash: bool = True,
    ):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout
        self.use_flash = use_flash

        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool = True,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with automatic backend selection."""
        batch_size, seq_len, _ = query.shape

        q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        if self.use_flash and not return_attention:
            # Use FlashAttention
            output = use_flash_attention(
                q, k, v,
                is_causal=is_causal,
                dropout=self.dropout,
                training=self.training,
            )
            attention_weights = None
        else:
            # Standard attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

            if is_causal:
                mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device))
                scores = scores.masked_fill(mask == 0, float("-inf"))

            attention_weights = F.softmax(scores, dim=-1)
            if self.dropout > 0 and self.training:
                attention_weights = F.dropout(attention_weights, p=self.dropout)

            output = torch.matmul(attention_weights, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(output)

        if return_attention:
            return output, attention_weights
        return output, None
