"""
Embedding layers for transformer models.

This module implements:
- Token embeddings
- Sinusoidal positional encoding (original Transformer)
- Learned positional embeddings
- Rotary Position Embedding (RoPE)
- Combined token + positional embeddings
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Token embedding layer.

    Converts token IDs to dense vectors.

    Args:
        vocab_size: Size of the vocabulary.
        d_model: Embedding dimension.
        padding_idx: Index of padding token (embeddings will be zeros).

    Example:
        >>> embed = TokenEmbedding(vocab_size=10000, d_model=512)
        >>> tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> embeddings = embed(tokens)  # (2, 3, 512)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

        # Initialize embeddings
        self._init_weights()

    def _init_weights(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if self.embedding.padding_idx is not None:
            nn.init.zeros_(self.embedding.weight[self.embedding.padding_idx])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Token IDs of shape (batch, seq_len)

        Returns:
            Embeddings of shape (batch, seq_len, d_model)
        """
        return self.embedding(x)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from "Attention Is All You Need".

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        d_model: Model dimension.
        max_seq_len: Maximum sequence length.
        dropout: Dropout probability.

    Example:
        >>> pe = SinusoidalPositionalEncoding(d_model=512, max_seq_len=1000)
        >>> x = torch.randn(2, 10, 512)
        >>> x_with_pe = pe(x)  # Adds positional encoding
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 5000,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but saved with model)
        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Tensor with positional encoding added.
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

    def get_encoding(self, seq_len: int) -> torch.Tensor:
        """Get positional encoding for visualization."""
        return self.pe[:, :seq_len, :].squeeze(0)


class LearnedPositionalEmbedding(nn.Module):
    """
    Learned positional embeddings.

    Each position has a learned embedding vector that is added to token embeddings.

    Args:
        max_seq_len: Maximum sequence length.
        d_model: Embedding dimension.
        dropout: Dropout probability.

    Example:
        >>> pe = LearnedPositionalEmbedding(max_seq_len=512, d_model=256)
        >>> x = torch.randn(2, 10, 256)
        >>> x_with_pe = pe(x)
    """

    def __init__(
        self,
        max_seq_len: int,
        d_model: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

        # Initialize
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional embedding to input.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Tensor with positional embedding added.
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        pos_emb = self.embedding(positions)  # (seq_len, d_model)
        x = x + pos_emb.unsqueeze(0)
        return self.dropout(x)

    def get_encoding(self, seq_len: int) -> torch.Tensor:
        """Get positional embeddings for visualization."""
        positions = torch.arange(seq_len, device=self.embedding.weight.device)
        return self.embedding(positions)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    From "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    (Su et al., 2021)

    RoPE encodes position by rotating the query and key vectors, which allows
    the model to extrapolate to longer sequences than seen during training.

    Args:
        d_model: Model dimension (must be even).
        max_seq_len: Maximum sequence length.
        base: Base for the geometric progression (default: 10000).

    Example:
        >>> rope = RotaryPositionalEmbedding(d_model=64, max_seq_len=512)
        >>> q = torch.randn(2, 8, 10, 64)  # (batch, heads, seq, d_k)
        >>> k = torch.randn(2, 8, 10, 64)
        >>> q_rot, k_rot = rope(q, k)
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
    ):
        super().__init__()

        assert d_model % 2 == 0, "d_model must be even for RoPE"

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos and sin for all positions
        self._precompute_cache(max_seq_len)

    def _precompute_cache(self, seq_len: int):
        """Precompute cos and sin values."""
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, d_model/2)

        # Create cos and sin caches
        cos_cache = torch.cos(freqs)
        sin_cache = torch.sin(freqs)

        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: Optional[int] = None,
    ) -> tuple:
        """
        Apply rotary embeddings to query and key tensors.

        Args:
            q: Query tensor of shape (batch, n_heads, seq_len, d_k)
            k: Key tensor of shape (batch, n_heads, seq_len, d_k)
            seq_len: Optional sequence length override.

        Returns:
            Tuple of (rotated_q, rotated_k) with same shapes as inputs.
        """
        if seq_len is None:
            seq_len = q.size(2)

        # Extend cache if needed
        if seq_len > self.cos_cache.size(0):
            self._precompute_cache(seq_len)

        cos = self.cos_cache[:seq_len]  # (seq_len, d_model/2)
        sin = self.sin_cache[:seq_len]  # (seq_len, d_model/2)

        # Reshape for broadcasting: (1, 1, seq_len, d_model/2)
        cos = cos.view(1, 1, seq_len, -1)
        sin = sin.view(1, 1, seq_len, -1)

        # Repeat cos and sin for the full d_model dimension
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)

        # Apply rotation
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)

        return q_rot, k_rot


class TransformerEmbedding(nn.Module):
    """
    Combined token and positional embedding for transformers.

    Combines token embeddings with positional information using the specified
    positional encoding type.

    Args:
        vocab_size: Vocabulary size.
        d_model: Model dimension.
        max_seq_len: Maximum sequence length.
        dropout: Dropout probability.
        padding_idx: Padding token index.
        pos_encoding: Type of positional encoding ('sinusoidal', 'learned', or 'none').

    Example:
        >>> embed = TransformerEmbedding(
        ...     vocab_size=10000, d_model=512, max_seq_len=1024
        ... )
        >>> tokens = torch.tensor([[1, 2, 3, 4]])
        >>> embeddings = embed(tokens)  # (1, 4, 512)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        padding_idx: Optional[int] = None,
        pos_encoding: str = "sinusoidal",
    ):
        super().__init__()

        self.d_model = d_model
        self.scale = math.sqrt(d_model)

        # Token embedding
        self.token_embedding = TokenEmbedding(vocab_size, d_model, padding_idx)

        # Positional encoding
        self.pos_encoding_type = pos_encoding
        if pos_encoding == "sinusoidal":
            self.positional_encoding = SinusoidalPositionalEncoding(
                d_model, max_seq_len, dropout=0.0
            )
        elif pos_encoding == "learned":
            self.positional_encoding = LearnedPositionalEmbedding(
                max_seq_len, d_model, dropout=0.0
            )
        elif pos_encoding == "none":
            self.positional_encoding = None
        else:
            raise ValueError(f"Unknown positional encoding type: {pos_encoding}")

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Token IDs of shape (batch, seq_len)

        Returns:
            Embeddings of shape (batch, seq_len, d_model)
        """
        # Token embedding with scaling
        embeddings = self.token_embedding(x) * self.scale

        # Add positional encoding
        if self.positional_encoding is not None:
            embeddings = self.positional_encoding(embeddings)

        return self.dropout(embeddings)

    def get_token_embeddings(self) -> torch.Tensor:
        """Get the token embedding weight matrix."""
        return self.token_embedding.embedding.weight
