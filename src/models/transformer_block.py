"""
Transformer block implementations.

This module implements:
- Standard transformer block (Pre-LN and Post-LN variants)
- Decoder-only transformer block (for causal language modeling)
- RMSNorm (Root Mean Square Layer Normalization)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .attention import MultiHeadAttention, CausalSelfAttention, GroupedQueryAttention
from .feedforward import FeedForward, SwiGLU, create_ffn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    From "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019).

    Simpler and faster than LayerNorm, used in LLaMA and other modern models.

    RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma

    Args:
        d_model: Model dimension.
        eps: Small constant for numerical stability.

    Example:
        >>> norm = RMSNorm(512)
        >>> x = torch.randn(2, 10, 512)
        >>> y = norm(x)
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Calculate RMS
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # Normalize and scale
        return x / rms * self.weight


class TransformerBlock(nn.Module):
    """
    Standard transformer block.

    Consists of:
    1. Multi-head self-attention with residual connection
    2. Feed-forward network with residual connection
    3. Layer normalization (Pre-LN or Post-LN)

    Pre-LN (default, more stable training):
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    Post-LN (original Transformer):
        x = LayerNorm(x + Attention(x))
        x = LayerNorm(x + FFN(x))

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward hidden dimension.
        dropout: Dropout probability.
        activation: FFN activation ('gelu', 'relu', 'silu').
        norm_type: Normalization type ('layernorm', 'rmsnorm').
        norm_first: If True, use Pre-LN (default). If False, use Post-LN.
        bias: Whether to use bias in linear layers.
        layer_norm_eps: Epsilon for layer normalization.

    Example:
        >>> block = TransformerBlock(d_model=512, n_heads=8)
        >>> x = torch.randn(2, 10, 512)
        >>> y, attn = block(x)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_type: str = "layernorm",
        norm_first: bool = True,
        bias: bool = True,
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()

        self.d_model = d_model
        self.norm_first = norm_first

        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, n_heads, dropout, bias)

        # Feed-forward network
        d_ff = d_ff if d_ff is not None else 4 * d_model
        self.ffn = FeedForward(d_model, d_ff, dropout, activation, bias)

        # Layer normalization
        if norm_type == "layernorm":
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        elif norm_type == "rmsnorm":
            self.norm1 = RMSNorm(d_model, eps=layer_norm_eps)
            self.norm2 = RMSNorm(d_model, eps=layer_norm_eps)
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")

        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask.
            return_attention: If True, return attention weights.

        Returns:
            Tuple of:
            - Output tensor of shape (batch, seq_len, d_model)
            - Attention weights if return_attention=True, else None
        """
        attention_weights = None

        if self.norm_first:
            # Pre-LN: LayerNorm -> Sublayer -> Residual
            # Attention sublayer
            normed = self.norm1(x)
            attn_out, attention_weights = self.attention(
                normed, normed, normed, mask=mask, return_attention=return_attention
            )
            x = x + self.dropout(attn_out)

            # FFN sublayer
            normed = self.norm2(x)
            ffn_out = self.ffn(normed)
            x = x + self.dropout(ffn_out)
        else:
            # Post-LN: Sublayer -> Residual -> LayerNorm
            # Attention sublayer
            attn_out, attention_weights = self.attention(
                x, x, x, mask=mask, return_attention=return_attention
            )
            x = self.norm1(x + self.dropout(attn_out))

            # FFN sublayer
            ffn_out = self.ffn(x)
            x = self.norm2(x + self.dropout(ffn_out))

        return x, attention_weights

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return the last computed attention weights."""
        return self.attention.get_attention_weights()


class DecoderBlock(nn.Module):
    """
    Decoder-only transformer block for causal language modeling.

    Like TransformerBlock but with automatic causal masking and optionally
    uses modern architecture choices (RMSNorm, SwiGLU, GQA).

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        n_kv_heads: Number of key-value heads (for GQA). If None, equals n_heads.
        d_ff: Feed-forward hidden dimension.
        max_seq_len: Maximum sequence length for causal mask.
        dropout: Dropout probability.
        ffn_type: FFN type ('standard', 'swiglu', 'geglu').
        norm_type: Normalization type ('layernorm', 'rmsnorm').
        bias: Whether to use bias.
        layer_norm_eps: Epsilon for layer normalization.

    Example:
        >>> block = DecoderBlock(
        ...     d_model=512, n_heads=8,
        ...     ffn_type='swiglu', norm_type='rmsnorm'
        ... )
        >>> x = torch.randn(2, 10, 512)
        >>> y, attn = block(x)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        d_ff: Optional[int] = None,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        ffn_type: str = "swiglu",
        norm_type: str = "rmsnorm",
        bias: bool = False,
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads

        # Attention (with optional GQA)
        if self.n_kv_heads == n_heads:
            self.attention = CausalSelfAttention(
                d_model, n_heads, max_seq_len, dropout, bias
            )
        else:
            # Use GQA with manual causal masking
            self.attention = GroupedQueryAttention(
                d_model, n_heads, self.n_kv_heads, dropout, bias
            )
            # Register causal mask
            causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
            self.register_buffer(
                "causal_mask", causal_mask.view(1, 1, max_seq_len, max_seq_len)
            )

        # Feed-forward network
        self.ffn = create_ffn(d_model, d_ff, ffn_type, dropout, bias=bias)

        # Layer normalization
        if norm_type == "layernorm":
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        elif norm_type == "rmsnorm":
            self.norm1 = RMSNorm(d_model, eps=layer_norm_eps)
            self.norm2 = RMSNorm(d_model, eps=layer_norm_eps)
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")

        self.dropout = nn.Dropout(dropout)

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
        # Pre-LN attention
        normed = self.norm1(x)

        if isinstance(self.attention, CausalSelfAttention):
            attn_out, attention_weights = self.attention(
                normed, return_attention=return_attention
            )
        else:
            # GQA with manual mask
            seq_len = x.size(1)
            mask = self.causal_mask[:, :, :seq_len, :seq_len]
            attn_out, attention_weights = self.attention(
                normed, normed, normed, mask=mask, return_attention=return_attention
            )

        x = x + self.dropout(attn_out)

        # Pre-LN FFN
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + self.dropout(ffn_out)

        return x, attention_weights

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return the last computed attention weights."""
        return self.attention.get_attention_weights()


class TransformerBlockStack(nn.Module):
    """
    Stack of transformer blocks.

    Convenience module for creating multiple transformer blocks with shared
    configuration.

    Args:
        n_layers: Number of transformer blocks.
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward hidden dimension.
        dropout: Dropout probability.
        **kwargs: Additional arguments passed to each block.

    Example:
        >>> stack = TransformerBlockStack(n_layers=6, d_model=512, n_heads=8)
        >>> x = torch.randn(2, 10, 512)
        >>> y = stack(x)
    """

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        block_type: str = "encoder",
        **kwargs,
    ):
        super().__init__()

        self.n_layers = n_layers

        if block_type == "encoder":
            self.blocks = nn.ModuleList([
                TransformerBlock(d_model, n_heads, d_ff, dropout, **kwargs)
                for _ in range(n_layers)
            ])
        elif block_type == "decoder":
            self.blocks = nn.ModuleList([
                DecoderBlock(d_model, n_heads, d_ff=d_ff, dropout=dropout, **kwargs)
                for _ in range(n_layers)
            ])
        else:
            raise ValueError(f"Unknown block type: {block_type}")

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_all_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Forward pass through all blocks.

        Args:
            x: Input tensor.
            mask: Optional attention mask (for encoder blocks).
            return_all_attention: If True, return attention from all layers.

        Returns:
            Tuple of output and optional list of attention weights per layer.
        """
        all_attention = [] if return_all_attention else None

        for block in self.blocks:
            if isinstance(block, TransformerBlock):
                x, attn = block(x, mask=mask, return_attention=return_all_attention)
            else:
                x, attn = block(x, return_attention=return_all_attention)

            if return_all_attention:
                all_attention.append(attn)

        return x, all_attention

    def get_attention_weights(self, layer: int = -1) -> Optional[torch.Tensor]:
        """Get attention weights from a specific layer."""
        return self.blocks[layer].get_attention_weights()
