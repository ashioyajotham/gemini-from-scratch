"""
Feed-forward network layers for transformer models.

This module implements:
- Standard position-wise feed-forward network
- GELU activation variants
- SwiGLU (Gated Linear Unit with Swish activation)
- GeGLU and other GLU variants
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit activation.

    GELU(x) = x * Phi(x) where Phi is the standard Gaussian CDF.

    Two approximations are available:
    - 'tanh': GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    - 'none': Uses exact computation via erf

    Args:
        approximate: Approximation method ('tanh' or 'none').

    Example:
        >>> gelu = GELU()
        >>> x = torch.randn(2, 10, 512)
        >>> y = gelu(x)
    """

    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate=self.approximate)


class SiLU(nn.Module):
    """
    Sigmoid Linear Unit (SiLU) / Swish activation.

    SiLU(x) = x * sigmoid(x)

    This is the activation used in SwiGLU.

    Example:
        >>> silu = SiLU()
        >>> x = torch.randn(2, 10, 512)
        >>> y = silu(x)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)


class FeedForward(nn.Module):
    """
    Standard position-wise feed-forward network.

    FFN(x) = activation(x W1 + b1) W2 + b2

    The default architecture expands the dimension by a factor of 4,
    applies an activation, then projects back.

    Args:
        d_model: Model dimension.
        d_ff: Hidden dimension. If None, defaults to 4 * d_model.
        dropout: Dropout probability.
        activation: Activation function ('relu', 'gelu', 'silu').
        bias: Whether to use bias in linear layers.

    Example:
        >>> ffn = FeedForward(d_model=512, d_ff=2048)
        >>> x = torch.randn(2, 10, 512)
        >>> y = ffn(x)  # (2, 10, 512)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        activation: str = "gelu",
        bias: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else 4 * d_model

        # Linear layers
        self.w1 = nn.Linear(d_model, self.d_ff, bias=bias)
        self.w2 = nn.Linear(self.d_ff, d_model, bias=bias)

        # Activation
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = GELU()
        elif activation == "silu":
            self.activation = SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Glorot."""
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        if self.w1.bias is not None:
            nn.init.zeros_(self.w1.bias)
        if self.w2.bias is not None:
            nn.init.zeros_(self.w2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        x = self.w1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.w2(x)
        return x


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network.

    From "GLU Variants Improve Transformer" (Shazeer, 2020).

    SwiGLU(x) = (Swish(x W1) ⊙ (x V)) W2

    Uses gated linear units with Swish (SiLU) activation for better
    performance than standard FFN.

    Args:
        d_model: Model dimension.
        d_ff: Hidden dimension. If None, defaults to 4 * d_model * 2/3
              (adjusted to maintain parameter count).
        dropout: Dropout probability.
        bias: Whether to use bias in linear layers.

    Example:
        >>> swiglu = SwiGLU(d_model=512)
        >>> x = torch.randn(2, 10, 512)
        >>> y = swiglu(x)  # (2, 10, 512)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        self.d_model = d_model
        # Adjust d_ff to maintain similar parameter count as standard FFN
        # Standard: 2 * d_model * d_ff
        # SwiGLU: 3 * d_model * d_ff (gate, up, down)
        # So we use 2/3 * 4 * d_model ≈ 2.67 * d_model
        if d_ff is None:
            self.d_ff = int(4 * d_model * 2 / 3)
            # Round to multiple of 256 for efficiency
            self.d_ff = ((self.d_ff + 255) // 256) * 256
        else:
            self.d_ff = d_ff

        # Gate and up projections (can be fused)
        self.w_gate = nn.Linear(d_model, self.d_ff, bias=bias)
        self.w_up = nn.Linear(d_model, self.d_ff, bias=bias)

        # Down projection
        self.w_down = nn.Linear(self.d_ff, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.w_gate.weight)
        nn.init.xavier_uniform_(self.w_up.weight)
        nn.init.xavier_uniform_(self.w_down.weight)
        if self.w_gate.bias is not None:
            nn.init.zeros_(self.w_gate.bias)
        if self.w_up.bias is not None:
            nn.init.zeros_(self.w_up.bias)
        if self.w_down.bias is not None:
            nn.init.zeros_(self.w_down.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Gate with SiLU activation
        gate = F.silu(self.w_gate(x))
        # Up projection (no activation)
        up = self.w_up(x)
        # Element-wise multiplication
        x = gate * up
        x = self.dropout(x)
        # Down projection
        x = self.w_down(x)
        return x


class GeGLU(nn.Module):
    """
    GeGLU feed-forward network.

    GeGLU(x) = (GELU(x W1) ⊙ (x V)) W2

    Similar to SwiGLU but uses GELU instead of Swish activation.

    Args:
        d_model: Model dimension.
        d_ff: Hidden dimension.
        dropout: Dropout probability.
        bias: Whether to use bias.

    Example:
        >>> geglu = GeGLU(d_model=512)
        >>> x = torch.randn(2, 10, 512)
        >>> y = geglu(x)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        self.d_model = d_model
        if d_ff is None:
            self.d_ff = int(4 * d_model * 2 / 3)
            self.d_ff = ((self.d_ff + 255) // 256) * 256
        else:
            self.d_ff = d_ff

        self.w_gate = nn.Linear(d_model, self.d_ff, bias=bias)
        self.w_up = nn.Linear(d_model, self.d_ff, bias=bias)
        self.w_down = nn.Linear(self.d_ff, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        gate = F.gelu(self.w_gate(x))
        up = self.w_up(x)
        x = gate * up
        x = self.dropout(x)
        x = self.w_down(x)
        return x


class FeedForwardExpert(nn.Module):
    """
    Feed-forward expert for Mixture of Experts.

    A lightweight FFN that can be used as an expert in MoE layers.

    Args:
        d_model: Model dimension.
        d_ff: Hidden dimension.
        dropout: Dropout probability.
        activation: Activation type ('gelu', 'silu', 'relu').

    Example:
        >>> expert = FeedForwardExpert(d_model=512, d_ff=2048)
        >>> x = torch.randn(100, 512)  # Selected tokens
        >>> y = expert(x)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        activation: str = "silu",
    ):
        super().__init__()

        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "silu":
            self.activation = F.silu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.activation(self.w1(x))
        x = self.dropout(x)
        x = self.w2(x)
        return x


def create_ffn(
    d_model: int,
    d_ff: Optional[int] = None,
    ffn_type: str = "standard",
    dropout: float = 0.0,
    activation: str = "gelu",
    bias: bool = True,
) -> nn.Module:
    """
    Factory function to create feed-forward networks.

    Args:
        d_model: Model dimension.
        d_ff: Hidden dimension.
        ffn_type: Type of FFN ('standard', 'swiglu', 'geglu').
        dropout: Dropout probability.
        activation: Activation for standard FFN.
        bias: Whether to use bias.

    Returns:
        Feed-forward network module.

    Example:
        >>> ffn = create_ffn(d_model=512, ffn_type='swiglu')
    """
    if ffn_type == "standard":
        return FeedForward(d_model, d_ff, dropout, activation, bias)
    elif ffn_type == "swiglu":
        return SwiGLU(d_model, d_ff, dropout, bias=False)
    elif ffn_type == "geglu":
        return GeGLU(d_model, d_ff, dropout, bias=False)
    else:
        raise ValueError(f"Unknown FFN type: {ffn_type}")
