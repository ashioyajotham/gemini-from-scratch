"""
Mixture of Experts (MoE) implementation.

This module implements:
- Expert feed-forward networks
- Top-k routing with load balancing
- MoE transformer block
- Auxiliary losses for balanced routing
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feedforward import SwiGLU, FeedForward


class Router(nn.Module):
    """
    Router network for Mixture of Experts.

    Routes each token to the top-k experts based on learned routing weights.

    Args:
        d_model: Model dimension.
        num_experts: Total number of experts.
        top_k: Number of experts to route each token to.
        noise_std: Standard deviation of noise added during training (for exploration).

    Example:
        >>> router = Router(d_model=512, num_experts=8, top_k=2)
        >>> x = torch.randn(2, 10, 512)
        >>> weights, indices = router(x)
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        noise_std: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std

        # Routing projection
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute routing weights and expert assignments.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Tuple of:
            - routing_weights: Weights for top-k experts (batch, seq_len, top_k)
            - expert_indices: Indices of top-k experts (batch, seq_len, top_k)
            - router_logits: Full router logits for auxiliary loss (batch, seq_len, num_experts)
        """
        # Compute router logits
        router_logits = self.gate(x)  # (batch, seq_len, num_experts)

        # Add noise during training for exploration
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(router_logits) * self.noise_std
            router_logits = router_logits + noise

        # Get top-k experts
        routing_weights, expert_indices = torch.topk(
            router_logits, self.top_k, dim=-1
        )

        # Normalize weights with softmax
        routing_weights = F.softmax(routing_weights, dim=-1)

        return routing_weights, expert_indices, router_logits


class Expert(nn.Module):
    """
    A single expert (feed-forward network).

    Args:
        d_model: Model dimension.
        d_ff: Hidden dimension.
        dropout: Dropout probability.
        ffn_type: Type of FFN ('standard', 'swiglu').

    Example:
        >>> expert = Expert(d_model=512, d_ff=2048)
        >>> x = torch.randn(100, 512)  # Tokens routed to this expert
        >>> y = expert(x)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        ffn_type: str = "swiglu",
    ):
        super().__init__()

        if ffn_type == "swiglu":
            self.ffn = SwiGLU(d_model, d_ff, dropout)
        else:
            self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the expert."""
        return self.ffn(x)


class MoELayer(nn.Module):
    """
    Mixture of Experts layer.

    Replaces a standard FFN with multiple expert FFNs, where each token
    is routed to a subset of experts.

    Args:
        d_model: Model dimension.
        d_ff: Expert hidden dimension.
        num_experts: Number of expert networks.
        top_k: Number of experts per token.
        dropout: Dropout probability.
        ffn_type: Type of FFN for experts.
        capacity_factor: Capacity factor for load balancing (tokens per expert).
        noise_std: Noise for router exploration.

    Example:
        >>> moe = MoELayer(d_model=512, num_experts=8, top_k=2)
        >>> x = torch.randn(2, 10, 512)
        >>> output, aux_loss = moe(x)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.0,
        ffn_type: str = "swiglu",
        capacity_factor: float = 1.25,
        noise_std: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        # Router
        self.router = Router(d_model, num_experts, top_k, noise_std)

        # Experts
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout, ffn_type)
            for _ in range(num_experts)
        ])

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MoE layer.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Tuple of:
            - output: Output tensor of shape (batch, seq_len, d_model)
            - aux_loss: Auxiliary load balancing loss
        """
        batch_size, seq_len, d_model = x.shape

        # Get routing weights and expert assignments
        routing_weights, expert_indices, router_logits = self.router(x)

        # Flatten batch and sequence dimensions
        x_flat = x.view(-1, d_model)  # (batch * seq_len, d_model)
        routing_weights_flat = routing_weights.view(-1, self.top_k)
        expert_indices_flat = expert_indices.view(-1, self.top_k)

        # Initialize output
        output_flat = torch.zeros_like(x_flat)

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens assigned to this expert
            expert_mask = (expert_indices_flat == expert_idx).any(dim=-1)

            if not expert_mask.any():
                continue

            # Get tokens for this expert
            expert_tokens = x_flat[expert_mask]

            # Get routing weights for this expert
            # Find which top-k slot this expert is in for each token
            expert_weights = torch.zeros(expert_mask.sum(), device=x.device)
            for k in range(self.top_k):
                k_mask = expert_indices_flat[expert_mask, k] == expert_idx
                expert_weights[k_mask] = routing_weights_flat[expert_mask][k_mask, k]

            # Process through expert
            expert_output = self.experts[expert_idx](expert_tokens)

            # Weighted contribution
            output_flat[expert_mask] += expert_weights.unsqueeze(-1) * expert_output

        # Reshape output
        output = output_flat.view(batch_size, seq_len, d_model)

        # Compute auxiliary loss for load balancing
        aux_loss = self._compute_aux_loss(router_logits, expert_indices)

        return output, aux_loss

    def _compute_aux_loss(
        self,
        router_logits: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute auxiliary load balancing loss.

        Encourages balanced routing across experts.
        """
        batch_size, seq_len, num_experts = router_logits.shape

        # Compute router probabilities
        router_probs = F.softmax(router_logits, dim=-1)

        # Fraction of tokens routed to each expert
        # (average probability assigned to each expert)
        expert_probs = router_probs.mean(dim=[0, 1])  # (num_experts,)

        # Fraction of tokens with each expert in top-k
        expert_indices_flat = expert_indices.view(-1, self.top_k)
        expert_counts = torch.zeros(num_experts, device=router_logits.device)
        for k in range(self.top_k):
            expert_counts.scatter_add_(
                0,
                expert_indices_flat[:, k],
                torch.ones(expert_indices_flat.size(0), device=router_logits.device)
            )
        expert_fraction = expert_counts / (batch_size * seq_len * self.top_k)

        # Load balancing loss: dot product of fractions
        # Minimized when both distributions are uniform
        aux_loss = (expert_probs * expert_fraction).sum() * num_experts

        return aux_loss


class MoETransformerBlock(nn.Module):
    """
    Transformer block with Mixture of Experts FFN.

    Replaces the standard FFN with an MoE layer while keeping
    the attention mechanism unchanged.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_ff: Expert hidden dimension.
        num_experts: Number of experts.
        top_k: Experts per token.
        max_seq_len: Maximum sequence length.
        dropout: Dropout probability.
        norm_type: Normalization type ('layernorm', 'rmsnorm').

    Example:
        >>> block = MoETransformerBlock(
        ...     d_model=512, n_heads=8, num_experts=8, top_k=2
        ... )
        >>> x = torch.randn(2, 10, 512)
        >>> output, aux_loss = block(x)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        num_experts: int = 8,
        top_k: int = 2,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        norm_type: str = "rmsnorm",
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()

        from .transformer_block import RMSNorm, DecoderBlock
        from .attention import CausalSelfAttention

        self.d_model = d_model

        # Attention (same as standard block)
        self.attention = CausalSelfAttention(
            d_model, n_heads, max_seq_len, dropout, bias=False
        )

        # MoE FFN
        self.moe = MoELayer(
            d_model, d_ff, num_experts, top_k, dropout
        )

        # Normalization
        if norm_type == "rmsnorm":
            self.norm1 = RMSNorm(d_model, eps=layer_norm_eps)
            self.norm2 = RMSNorm(d_model, eps=layer_norm_eps)
        else:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            return_attention: If True, return attention weights.

        Returns:
            Tuple of:
            - output: Output tensor
            - aux_loss: MoE auxiliary loss
            - attention_weights: If requested
        """
        # Pre-LN attention
        normed = self.norm1(x)
        attn_out, attn_weights = self.attention(normed, return_attention=return_attention)
        x = x + self.dropout(attn_out)

        # Pre-LN MoE
        normed = self.norm2(x)
        moe_out, aux_loss = self.moe(normed)
        x = x + self.dropout(moe_out)

        return x, aux_loss, attn_weights


class SparseMoE(nn.Module):
    """
    Sparse Mixture of Experts with capacity-based routing.

    Implements token dropping when experts exceed capacity,
    similar to Switch Transformer and GShard.

    Args:
        d_model: Model dimension.
        d_ff: Expert hidden dimension.
        num_experts: Number of experts.
        top_k: Experts per token.
        capacity_factor: Expert capacity as factor of average load.
        dropout: Dropout probability.

    Example:
        >>> moe = SparseMoE(d_model=512, num_experts=8, capacity_factor=1.25)
        >>> x = torch.randn(2, 10, 512)
        >>> output, aux_loss = moe(x)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        num_experts: int = 8,
        top_k: int = 1,
        capacity_factor: float = 1.25,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        self.router = Router(d_model, num_experts, top_k, noise_std=0.0)
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout, "swiglu")
            for _ in range(num_experts)
        ])

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with capacity-based routing."""
        batch_size, seq_len, d_model = x.shape
        total_tokens = batch_size * seq_len

        # Compute expert capacity
        capacity = int(self.capacity_factor * total_tokens * self.top_k / self.num_experts)

        # Get routing
        routing_weights, expert_indices, router_logits = self.router(x)

        # Flatten
        x_flat = x.view(-1, d_model)
        routing_weights_flat = routing_weights.view(-1, self.top_k)
        expert_indices_flat = expert_indices.view(-1, self.top_k)

        # Initialize output
        output_flat = torch.zeros_like(x_flat)

        # Track tokens processed per expert
        expert_counts = torch.zeros(self.num_experts, dtype=torch.long, device=x.device)

        # Process tokens
        for token_idx in range(total_tokens):
            for k in range(self.top_k):
                expert_idx = expert_indices_flat[token_idx, k].item()

                # Check capacity
                if expert_counts[expert_idx] >= capacity:
                    continue  # Drop token for this expert

                expert_counts[expert_idx] += 1

                # Process token
                token = x_flat[token_idx:token_idx+1]
                weight = routing_weights_flat[token_idx, k]
                expert_output = self.experts[expert_idx](token)
                output_flat[token_idx] += weight * expert_output.squeeze(0)

        output = output_flat.view(batch_size, seq_len, d_model)

        # Auxiliary loss
        aux_loss = self._compute_aux_loss(router_logits, expert_indices)

        return output, aux_loss

    def _compute_aux_loss(
        self,
        router_logits: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute load balancing auxiliary loss."""
        router_probs = F.softmax(router_logits, dim=-1)
        expert_probs = router_probs.mean(dim=[0, 1])

        # Encourage uniform distribution
        target = torch.ones_like(expert_probs) / self.num_experts
        aux_loss = F.mse_loss(expert_probs, target) * self.num_experts

        return aux_loss
