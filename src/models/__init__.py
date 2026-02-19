"""
Transformer model components.

This module contains the building blocks for transformer architectures:
- Attention mechanisms (scaled dot-product, multi-head, GQA)
- Embeddings (token, positional, RoPE)
- Feed-forward networks (standard, SwiGLU, GeGLU)
- Transformer blocks (encoder, decoder)
- Efficient attention (sliding window, sparse, linear)
- Mixture of Experts (MoE)
- Multimodal components (vision encoder, cross-modal attention)
"""

from .attention import (
    scaled_dot_product_attention,
    MultiHeadAttention,
    CausalSelfAttention,
    GroupedQueryAttention,
    create_causal_mask,
    create_padding_mask,
)

from .embeddings import (
    TokenEmbedding,
    SinusoidalPositionalEncoding,
    LearnedPositionalEmbedding,
    RotaryPositionalEmbedding,
    TransformerEmbedding,
)

from .feedforward import (
    GELU,
    SiLU,
    FeedForward,
    SwiGLU,
    GeGLU,
    FeedForwardExpert,
    create_ffn,
)

from .transformer_block import (
    RMSNorm,
    TransformerBlock,
    DecoderBlock,
    TransformerBlockStack,
)

from .transformer import (
    TransformerConfig,
    Transformer,
    create_model,
)

from .efficient_attention import (
    SlidingWindowAttention,
    SparseAttention,
    LinearAttention,
    FlashAttentionSimulator,
    EfficientMultiHeadAttention,
    use_flash_attention,
)

from .moe import (
    Router,
    Expert,
    MoELayer,
    MoETransformerBlock,
    SparseMoE,
)

from .multimodal import (
    PatchEmbedding,
    VisionEncoder,
    CrossModalAttention,
    MultimodalFusion,
    VisionLanguageProjector,
    MultimodalTransformerBlock,
)

__all__ = [
    # Attention
    "scaled_dot_product_attention",
    "MultiHeadAttention",
    "CausalSelfAttention",
    "GroupedQueryAttention",
    "create_causal_mask",
    "create_padding_mask",
    # Embeddings
    "TokenEmbedding",
    "SinusoidalPositionalEncoding",
    "LearnedPositionalEmbedding",
    "RotaryPositionalEmbedding",
    "TransformerEmbedding",
    # Feed-forward
    "GELU",
    "SiLU",
    "FeedForward",
    "SwiGLU",
    "GeGLU",
    "FeedForwardExpert",
    "create_ffn",
    # Transformer blocks
    "RMSNorm",
    "TransformerBlock",
    "DecoderBlock",
    "TransformerBlockStack",
    # Complete model
    "TransformerConfig",
    "Transformer",
    "create_model",
    # Efficient attention
    "SlidingWindowAttention",
    "SparseAttention",
    "LinearAttention",
    "FlashAttentionSimulator",
    "EfficientMultiHeadAttention",
    "use_flash_attention",
    # Mixture of Experts
    "Router",
    "Expert",
    "MoELayer",
    "MoETransformerBlock",
    "SparseMoE",
    # Multimodal
    "PatchEmbedding",
    "VisionEncoder",
    "CrossModalAttention",
    "MultimodalFusion",
    "VisionLanguageProjector",
    "MultimodalTransformerBlock",
]
