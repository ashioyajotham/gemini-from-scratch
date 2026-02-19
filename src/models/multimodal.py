"""
Multimodal components for vision-language models.

This module implements:
- Vision encoder (ViT-style patch embedding)
- Cross-modal attention
- Multimodal fusion strategies
- Vision-language projection
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention
from .transformer_block import TransformerBlock, RMSNorm
from .embeddings import SinusoidalPositionalEncoding


class PatchEmbedding(nn.Module):
    """
    Convert image to patch embeddings (ViT-style).

    Splits image into fixed-size patches and projects each patch
    to the model dimension.

    Args:
        image_size: Input image size (assumes square).
        patch_size: Size of each patch.
        in_channels: Number of input channels (3 for RGB).
        d_model: Output embedding dimension.

    Example:
        >>> patch_embed = PatchEmbedding(image_size=224, patch_size=16, d_model=768)
        >>> image = torch.randn(2, 3, 224, 224)
        >>> patches = patch_embed(image)  # (2, 196, 768)
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        d_model: int = 768,
    ):
        super().__init__()

        assert image_size % patch_size == 0, "Image size must be divisible by patch size"

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.d_model = d_model

        # Patch projection using convolution
        self.projection = nn.Conv2d(
            in_channels, d_model,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert image to patch embeddings.

        Args:
            x: Image tensor of shape (batch, channels, height, width).

        Returns:
            Patch embeddings of shape (batch, num_patches, d_model).
        """
        # x: (batch, channels, H, W)
        x = self.projection(x)  # (batch, d_model, H/patch, W/patch)
        x = x.flatten(2)  # (batch, d_model, num_patches)
        x = x.transpose(1, 2)  # (batch, num_patches, d_model)
        return x


class VisionEncoder(nn.Module):
    """
    Vision Transformer (ViT) encoder for image understanding.

    Converts images to a sequence of embeddings that can be processed
    by a transformer.

    Args:
        image_size: Input image size.
        patch_size: Patch size.
        in_channels: Number of input channels.
        d_model: Model dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        d_ff: Feed-forward hidden dimension.
        dropout: Dropout probability.
        use_cls_token: Whether to use a [CLS] token.

    Example:
        >>> encoder = VisionEncoder(image_size=224, d_model=768, n_layers=12)
        >>> image = torch.randn(2, 3, 224, 224)
        >>> features = encoder(image)  # (2, 197, 768) with CLS token
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        use_cls_token: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.use_cls_token = use_cls_token

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            image_size, patch_size, in_channels, d_model
        )
        num_patches = self.patch_embed.num_patches

        # Class token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            num_positions = num_patches + 1
        else:
            self.cls_token = None
            num_positions = num_patches

        # Positional embedding (learned)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, d_model))

        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                norm_type="layernorm",
                norm_first=True,
            )
            for _ in range(n_layers)
        ])

        # Final norm
        self.norm = nn.LayerNorm(d_model)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        return_all_tokens: bool = True,
    ) -> torch.Tensor:
        """
        Encode image to features.

        Args:
            x: Image tensor of shape (batch, channels, height, width).
            return_all_tokens: If False and use_cls_token, return only CLS token.

        Returns:
            Features of shape (batch, num_tokens, d_model).
        """
        batch_size = x.size(0)

        # Patch embedding
        x = self.patch_embed(x)  # (batch, num_patches, d_model)

        # Add CLS token
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.dropout(x)

        # Transformer blocks
        for block in self.blocks:
            x, _ = block(x)

        # Final norm
        x = self.norm(x)

        if not return_all_tokens and self.use_cls_token:
            return x[:, 0]  # Return only CLS token

        return x


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for fusing different modalities.

    Allows one modality to attend to another (e.g., text attending to image).

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        dropout: Dropout probability.

    Example:
        >>> cross_attn = CrossModalAttention(d_model=768, n_heads=12)
        >>> text_features = torch.randn(2, 10, 768)
        >>> image_features = torch.randn(2, 197, 768)
        >>> fused = cross_attn(text_features, image_features)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_features: torch.Tensor,
        context_features: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply cross-modal attention.

        Args:
            query_features: Query modality features (batch, query_len, d_model).
            context_features: Context modality features (batch, context_len, d_model).
            return_attention: If True, return attention weights.

        Returns:
            Tuple of fused features and optional attention weights.
        """
        # Pre-norm
        normed_query = self.norm(query_features)

        # Cross-attention: query attends to context
        attended, attn_weights = self.attention(
            normed_query, context_features, context_features,
            return_attention=return_attention
        )

        # Residual connection
        output = query_features + self.dropout(attended)

        return output, attn_weights


class MultimodalFusion(nn.Module):
    """
    Multimodal fusion module for combining vision and language.

    Supports different fusion strategies:
    - 'concat': Concatenate modalities
    - 'cross_attention': Cross-modal attention
    - 'gated': Gated fusion

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads for cross-attention.
        fusion_type: Fusion strategy ('concat', 'cross_attention', 'gated').
        dropout: Dropout probability.

    Example:
        >>> fusion = MultimodalFusion(d_model=768, fusion_type='cross_attention')
        >>> text = torch.randn(2, 10, 768)
        >>> image = torch.randn(2, 197, 768)
        >>> fused = fusion(text, image)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        fusion_type: str = "cross_attention",
        dropout: float = 0.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.fusion_type = fusion_type

        if fusion_type == "cross_attention":
            # Bidirectional cross-attention
            self.text_to_image = CrossModalAttention(d_model, n_heads, dropout)
            self.image_to_text = CrossModalAttention(d_model, n_heads, dropout)
        elif fusion_type == "gated":
            # Gated fusion
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid(),
            )
            self.proj = nn.Linear(d_model * 2, d_model)
        elif fusion_type == "concat":
            # Simple concatenation (no learnable parameters)
            pass
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse text and image features.

        Args:
            text_features: Text features (batch, text_len, d_model).
            image_features: Image features (batch, image_len, d_model).

        Returns:
            Tuple of fused (text_features, image_features).
        """
        if self.fusion_type == "cross_attention":
            # Text attends to image
            fused_text, _ = self.text_to_image(text_features, image_features)
            # Image attends to text
            fused_image, _ = self.image_to_text(image_features, text_features)
            return fused_text, fused_image

        elif self.fusion_type == "gated":
            # Pool image features
            image_pooled = image_features.mean(dim=1, keepdim=True)
            image_pooled = image_pooled.expand(-1, text_features.size(1), -1)

            # Concatenate and gate
            concat = torch.cat([text_features, image_pooled], dim=-1)
            gate = self.gate(concat)
            fused = self.proj(concat)
            fused_text = text_features + gate * fused

            return fused_text, image_features

        elif self.fusion_type == "concat":
            # Return concatenated sequence
            return text_features, image_features

        return text_features, image_features


class VisionLanguageProjector(nn.Module):
    """
    Project vision features to language model space.

    Used to align vision encoder outputs with the language model's
    embedding space.

    Args:
        vision_dim: Vision encoder output dimension.
        language_dim: Language model dimension.
        num_tokens: Number of visual tokens to produce (for resampling).
        projector_type: Type of projector ('linear', 'mlp', 'resampler').

    Example:
        >>> projector = VisionLanguageProjector(
        ...     vision_dim=1024, language_dim=4096, projector_type='mlp'
        ... )
        >>> image_features = torch.randn(2, 197, 1024)
        >>> projected = projector(image_features)  # (2, 197, 4096)
    """

    def __init__(
        self,
        vision_dim: int,
        language_dim: int,
        num_tokens: Optional[int] = None,
        projector_type: str = "mlp",
    ):
        super().__init__()

        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.num_tokens = num_tokens
        self.projector_type = projector_type

        if projector_type == "linear":
            self.projector = nn.Linear(vision_dim, language_dim)
        elif projector_type == "mlp":
            self.projector = nn.Sequential(
                nn.Linear(vision_dim, language_dim),
                nn.GELU(),
                nn.Linear(language_dim, language_dim),
            )
        elif projector_type == "resampler":
            # Learnable queries for resampling
            assert num_tokens is not None, "num_tokens required for resampler"
            self.queries = nn.Parameter(torch.randn(1, num_tokens, language_dim))
            self.cross_attn = MultiHeadAttention(language_dim, n_heads=8)
            self.proj_kv = nn.Linear(vision_dim, language_dim)
            self.norm = nn.LayerNorm(language_dim)
        else:
            raise ValueError(f"Unknown projector type: {projector_type}")

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Project vision features to language space.

        Args:
            image_features: Vision features (batch, num_patches, vision_dim).

        Returns:
            Projected features (batch, num_tokens, language_dim).
        """
        batch_size = image_features.size(0)

        if self.projector_type in ["linear", "mlp"]:
            return self.projector(image_features)

        elif self.projector_type == "resampler":
            # Project KV to language dim
            kv = self.proj_kv(image_features)

            # Expand queries
            queries = self.queries.expand(batch_size, -1, -1)

            # Cross-attention: queries attend to image features
            output, _ = self.cross_attn(queries, kv, kv)
            output = self.norm(output)

            return output

        return image_features


class MultimodalTransformerBlock(nn.Module):
    """
    Transformer block with interleaved cross-modal attention.

    Processes text while attending to visual context.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward dimension.
        dropout: Dropout probability.
        cross_attention_freq: Apply cross-attention every N layers.

    Example:
        >>> block = MultimodalTransformerBlock(d_model=768, n_heads=12)
        >>> text = torch.randn(2, 10, 768)
        >>> image = torch.randn(2, 197, 768)
        >>> output = block(text, image)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
    ):
        super().__init__()

        from .attention import CausalSelfAttention
        from .feedforward import SwiGLU

        # Self-attention for text
        self.self_attn = CausalSelfAttention(
            d_model, n_heads, max_seq_len, dropout
        )

        # Cross-attention to image
        self.cross_attn = CrossModalAttention(d_model, n_heads, dropout)

        # Feed-forward
        self.ffn = SwiGLU(d_model, d_ff, dropout)

        # Norms
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process text with visual context.

        Args:
            text_features: Text features (batch, text_len, d_model).
            image_features: Image features (batch, image_len, d_model).

        Returns:
            Processed text features.
        """
        # Self-attention on text
        normed = self.norm1(text_features)
        attn_out, _ = self.self_attn(normed)
        text_features = text_features + self.dropout(attn_out)

        # Cross-attention to image
        normed = self.norm2(text_features)
        cross_out, _ = self.cross_attn(normed, image_features)
        text_features = text_features + self.dropout(cross_out)

        # FFN
        normed = self.norm3(text_features)
        ffn_out = self.ffn(normed)
        text_features = text_features + self.dropout(ffn_out)

        return text_features
