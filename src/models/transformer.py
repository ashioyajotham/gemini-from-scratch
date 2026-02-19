"""
Complete transformer model implementation.

This module implements:
- Decoder-only transformer (GPT-style) for causal language modeling
- Configurable architecture (attention, FFN, normalization)
- Weight initialization strategies
- Weight tying between embeddings and output projection
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import TransformerEmbedding, RotaryPositionalEmbedding
from .transformer_block import DecoderBlock, RMSNorm


@dataclass
class TransformerConfig:
    """
    Configuration for transformer model.

    Args:
        vocab_size: Size of vocabulary.
        d_model: Model dimension.
        n_heads: Number of attention heads.
        n_kv_heads: Number of KV heads for GQA. If None, equals n_heads.
        n_layers: Number of transformer layers.
        d_ff: Feed-forward hidden dimension. If None, defaults based on ffn_type.
        max_seq_len: Maximum sequence length.
        dropout: Dropout probability.
        bias: Whether to use bias in linear layers.
        ffn_type: Type of FFN ('standard', 'swiglu', 'geglu').
        norm_type: Type of normalization ('layernorm', 'rmsnorm').
        pos_encoding: Type of positional encoding ('sinusoidal', 'learned', 'rope', 'none').
        tie_weights: Whether to tie embedding and output weights.
        layer_norm_eps: Epsilon for layer normalization.
        initializer_range: Standard deviation for weight initialization.
        padding_idx: Padding token index.
    """

    vocab_size: int = 32000
    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: Optional[int] = None
    n_layers: int = 6
    d_ff: Optional[int] = None
    max_seq_len: int = 2048
    dropout: float = 0.1
    bias: bool = False
    ffn_type: str = "swiglu"
    norm_type: str = "rmsnorm"
    pos_encoding: str = "rope"
    tie_weights: bool = True
    layer_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    padding_idx: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout,
            "bias": self.bias,
            "ffn_type": self.ffn_type,
            "norm_type": self.norm_type,
            "pos_encoding": self.pos_encoding,
            "tie_weights": self.tie_weights,
            "layer_norm_eps": self.layer_norm_eps,
            "initializer_range": self.initializer_range,
            "padding_idx": self.padding_idx,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TransformerConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class Transformer(nn.Module):
    """
    Decoder-only transformer for causal language modeling.

    Architecture:
    1. Token embedding (+ positional encoding if not RoPE)
    2. Stack of decoder blocks
    3. Final layer normalization
    4. Output projection to vocabulary

    Args:
        config: TransformerConfig with model hyperparameters.

    Example:
        >>> config = TransformerConfig(vocab_size=10000, d_model=256, n_layers=4)
        >>> model = Transformer(config)
        >>> tokens = torch.randint(0, 10000, (2, 10))
        >>> logits = model(tokens)  # (2, 10, 10000)
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config

        # Token embedding
        if config.pos_encoding == "rope":
            # For RoPE, we only need token embeddings (no positional encoding in embedding layer)
            self.embedding = nn.Embedding(
                config.vocab_size, config.d_model, padding_idx=config.padding_idx
            )
            self.rope = RotaryPositionalEmbedding(
                config.d_model // config.n_heads, config.max_seq_len
            )
        else:
            self.embedding = TransformerEmbedding(
                vocab_size=config.vocab_size,
                d_model=config.d_model,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout,
                padding_idx=config.padding_idx,
                pos_encoding=config.pos_encoding,
            )
            self.rope = None

        self.embed_dropout = nn.Dropout(config.dropout)

        # Transformer layers
        self.layers = nn.ModuleList([
            DecoderBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                d_ff=config.d_ff,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout,
                ffn_type=config.ffn_type,
                norm_type=config.norm_type,
                bias=config.bias,
                layer_norm_eps=config.layer_norm_eps,
            )
            for _ in range(config.n_layers)
        ])

        # Final layer norm
        if config.norm_type == "rmsnorm":
            self.final_norm = RMSNorm(config.d_model, eps=config.layer_norm_eps)
        else:
            self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        # Output projection
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        if config.tie_weights:
            if config.pos_encoding == "rope":
                self.lm_head.weight = self.embedding.weight
            else:
                self.lm_head.weight = self.embedding.token_embedding.embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Report model size
        self._num_parameters = sum(p.numel() for p in self.parameters())

    def _init_weights(self, module: nn.Module):
        """Initialize weights using scaled initialization."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).
            attention_mask: Optional attention mask (not typically used for causal LM).
            return_hidden_states: If True, return hidden states from all layers.

        Returns:
            Tuple of:
            - Logits of shape (batch, seq_len, vocab_size)
            - Hidden states list if return_hidden_states=True, else None
        """
        # Get embeddings
        if self.config.pos_encoding == "rope":
            hidden_states = self.embedding(input_ids)
            hidden_states = self.embed_dropout(hidden_states)
        else:
            hidden_states = self.embedding(input_ids)

        all_hidden_states = [hidden_states] if return_hidden_states else None

        # Pass through transformer layers
        for layer in self.layers:
            hidden_states, _ = layer(hidden_states)
            if return_hidden_states:
                all_hidden_states.append(hidden_states)

        # Final norm
        hidden_states = self.final_norm(hidden_states)

        # Output projection
        logits = self.lm_head(hidden_states)

        return logits, all_hidden_states

    def get_num_parameters(self, non_embedding: bool = False) -> int:
        """
        Get number of parameters.

        Args:
            non_embedding: If True, exclude embedding parameters.

        Returns:
            Number of parameters.
        """
        if non_embedding:
            if self.config.pos_encoding == "rope":
                embedding_params = self.embedding.weight.numel()
            else:
                embedding_params = self.embedding.token_embedding.embedding.weight.numel()
            return self._num_parameters - embedding_params
        return self._num_parameters

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: Starting token IDs of shape (batch, seq_len).
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k filtering.
            top_p: Top-p (nucleus) filtering.
            do_sample: If False, use greedy decoding.
            eos_token_id: End of sequence token ID.
            pad_token_id: Padding token ID.

        Returns:
            Generated token IDs of shape (batch, seq_len + max_new_tokens).
        """
        self.eval()
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Track which sequences are done
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            # Get logits for the last position
            logits, _ = self.forward(input_ids)
            next_token_logits = logits[:, -1, :]  # (batch, vocab_size)

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float("-inf")

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float("-inf")

            # Sample or greedy
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Replace with pad for done sequences
            if pad_token_id is not None:
                next_token = torch.where(done.unsqueeze(1), pad_token_id, next_token)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Check for EOS
            if eos_token_id is not None:
                done = done | (next_token.squeeze(-1) == eos_token_id)
                if done.all():
                    break

        return input_ids

    @classmethod
    def from_config(cls, config: TransformerConfig) -> "Transformer":
        """Create model from config."""
        return cls(config)

    @classmethod
    def from_pretrained(cls, path: str, device: Optional[torch.device] = None) -> "Transformer":
        """
        Load a pretrained model.

        Args:
            path: Path to checkpoint file.
            device: Device to load model on.

        Returns:
            Loaded model.
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = TransformerConfig.from_dict(checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def save_pretrained(self, path: str):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint.
        """
        torch.save({
            "config": self.config.to_dict(),
            "model_state_dict": self.state_dict(),
        }, path)


def create_model(
    vocab_size: int,
    d_model: int = 256,
    n_heads: int = 4,
    n_layers: int = 4,
    **kwargs,
) -> Transformer:
    """
    Factory function to create a transformer model.

    Args:
        vocab_size: Vocabulary size.
        d_model: Model dimension.
        n_heads: Number of attention heads.
        n_layers: Number of layers.
        **kwargs: Additional config arguments.

    Returns:
        Transformer model.

    Example:
        >>> model = create_model(vocab_size=10000, d_model=512, n_layers=6)
    """
    config = TransformerConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        **kwargs,
    )
    return Transformer(config)
