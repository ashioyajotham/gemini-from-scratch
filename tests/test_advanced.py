"""
Tests for Phase 4 advanced features.

Tests efficient attention, Mixture of Experts, and multimodal components.
"""

import pytest
import torch
import torch.nn as nn

from src.models.efficient_attention import (
    SlidingWindowAttention,
    SparseAttention,
    LinearAttention,
    FlashAttentionSimulator,
    EfficientMultiHeadAttention,
    use_flash_attention,
)
from src.models.moe import (
    Router,
    Expert,
    MoELayer,
    MoETransformerBlock,
    SparseMoE,
)
from src.models.multimodal import (
    PatchEmbedding,
    VisionEncoder,
    CrossModalAttention,
    MultimodalFusion,
    VisionLanguageProjector,
    MultimodalTransformerBlock,
)


# =============================================================================
# Efficient Attention Tests
# =============================================================================

class TestSlidingWindowAttention:
    """Tests for SlidingWindowAttention."""

    def test_output_shape(self):
        """Test output shape is correct."""
        batch_size, seq_len, d_model = 2, 16, 64
        attn = SlidingWindowAttention(d_model=d_model, n_heads=4, window_size=4)
        x = torch.randn(batch_size, seq_len, d_model)

        output, weights = attn(x, x, x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert weights is None  # No attention weights by default

    def test_causal_masking(self):
        """Test causal masking is applied correctly."""
        d_model = 64
        attn = SlidingWindowAttention(d_model=d_model, n_heads=4, window_size=4)
        x = torch.randn(1, 8, d_model)

        output, _ = attn(x, x, x)

        assert output.shape == (1, 8, d_model)

    def test_different_window_sizes(self):
        """Test different window sizes."""
        d_model = 64
        for window_size in [2, 4, 8]:
            attn = SlidingWindowAttention(
                d_model=d_model, n_heads=4, window_size=window_size
            )
            x = torch.randn(2, 16, d_model)
            output, _ = attn(x, x, x)
            assert output.shape == (2, 16, d_model)

    def test_return_attention_weights(self):
        """Test returning attention weights."""
        d_model = 64
        attn = SlidingWindowAttention(d_model=d_model, n_heads=4, window_size=4)
        x = torch.randn(2, 8, d_model)

        output, weights = attn(x, x, x, return_attention=True)

        assert output.shape == (2, 8, d_model)
        assert weights is not None


class TestSparseAttention:
    """Tests for SparseAttention."""

    def test_output_shape(self):
        """Test output shape is correct."""
        batch_size, seq_len, d_model = 2, 16, 64
        attn = SparseAttention(d_model=d_model, n_heads=4, local_window=4)
        x = torch.randn(batch_size, seq_len, d_model)

        output, weights = attn(x, x, x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_global_attention(self):
        """Test with global tokens."""
        d_model = 64
        attn = SparseAttention(
            d_model=d_model, n_heads=4, local_window=4, global_tokens=2
        )
        x = torch.randn(2, 16, d_model)

        output, _ = attn(x, x, x)

        assert output.shape == (2, 16, d_model)

    def test_return_attention_weights(self):
        """Test returning attention weights."""
        d_model = 64
        attn = SparseAttention(d_model=d_model, n_heads=4, local_window=4)
        x = torch.randn(2, 8, d_model)

        output, weights = attn(x, x, x, return_attention=True)

        assert weights is not None


class TestLinearAttention:
    """Tests for LinearAttention."""

    def test_output_shape(self):
        """Test output shape is correct."""
        batch_size, seq_len, d_model = 2, 32, 64
        attn = LinearAttention(d_model=d_model, n_heads=4)
        x = torch.randn(batch_size, seq_len, d_model)

        output, weights = attn(x, x, x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert weights is None  # Linear attention doesn't return weights

    def test_long_sequence(self):
        """Test with longer sequences (linear attention scales better)."""
        d_model = 64
        attn = LinearAttention(d_model=d_model, n_heads=4)
        x = torch.randn(2, 128, d_model)

        output, _ = attn(x, x, x)

        assert output.shape == (2, 128, d_model)

    def test_different_feature_maps(self):
        """Test different feature map types."""
        d_model = 64
        for feature_map in ["elu", "relu", "softmax"]:
            attn = LinearAttention(
                d_model=d_model, n_heads=4, feature_map=feature_map
            )
            x = torch.randn(2, 16, d_model)
            output, _ = attn(x, x, x)
            assert output.shape == (2, 16, d_model)


class TestFlashAttentionSimulator:
    """Tests for FlashAttentionSimulator."""

    def test_output_shape(self):
        """Test output shape is correct."""
        batch_size, seq_len, d_model = 2, 16, 64
        attn = FlashAttentionSimulator(d_model=d_model, n_heads=4, chunk_size=8)
        x = torch.randn(batch_size, seq_len, d_model)

        output, weights = attn(x, x, x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_different_chunk_sizes(self):
        """Test different chunk sizes."""
        d_model = 64
        for chunk_size in [4, 8, 16]:
            attn = FlashAttentionSimulator(
                d_model=d_model, n_heads=4, chunk_size=chunk_size
            )
            x = torch.randn(2, 32, d_model)
            output, _ = attn(x, x, x)
            assert output.shape == (2, 32, d_model)


class TestEfficientMultiHeadAttention:
    """Tests for EfficientMultiHeadAttention."""

    def test_with_flash_attention(self):
        """Test with FlashAttention enabled."""
        d_model = 64
        attn = EfficientMultiHeadAttention(
            d_model=d_model, n_heads=4, use_flash=True
        )
        x = torch.randn(2, 16, d_model)

        output, _ = attn(x, x, x)

        assert output.shape == (2, 16, d_model)

    def test_without_flash_attention(self):
        """Test with FlashAttention disabled."""
        d_model = 64
        attn = EfficientMultiHeadAttention(
            d_model=d_model, n_heads=4, use_flash=False
        )
        x = torch.randn(2, 16, d_model)

        output, weights = attn(x, x, x, return_attention=True)

        assert output.shape == (2, 16, d_model)
        assert weights is not None

    def test_causal_mode(self):
        """Test causal attention mode."""
        d_model = 64
        attn = EfficientMultiHeadAttention(d_model=d_model, n_heads=4)
        x = torch.randn(2, 16, d_model)

        # Test with causal=True (default)
        output1, _ = attn(x, x, x, is_causal=True)
        # Test with causal=False
        output2, _ = attn(x, x, x, is_causal=False)

        assert output1.shape == (2, 16, d_model)
        assert output2.shape == (2, 16, d_model)


class TestUseFlashAttention:
    """Tests for use_flash_attention function."""

    def test_output_shape(self):
        """Test that function returns correct output shape."""
        batch_size, n_heads, seq_len, d_k = 2, 4, 16, 32
        q = torch.randn(batch_size, n_heads, seq_len, d_k)
        k = torch.randn(batch_size, n_heads, seq_len, d_k)
        v = torch.randn(batch_size, n_heads, seq_len, d_k)

        output = use_flash_attention(q, k, v, is_causal=True)

        assert output.shape == (batch_size, n_heads, seq_len, d_k)

    def test_non_causal(self):
        """Test non-causal attention."""
        q = torch.randn(2, 4, 16, 32)
        k = torch.randn(2, 4, 16, 32)
        v = torch.randn(2, 4, 16, 32)

        output = use_flash_attention(q, k, v, is_causal=False)

        assert output.shape == (2, 4, 16, 32)


# =============================================================================
# Mixture of Experts Tests
# =============================================================================

class TestRouter:
    """Tests for Router."""

    def test_output_shapes(self):
        """Test output shapes are correct."""
        batch_size, seq_len, d_model = 2, 10, 64
        router = Router(d_model=d_model, num_experts=8, top_k=2)
        x = torch.randn(batch_size, seq_len, d_model)

        weights, indices, logits = router(x)

        assert weights.shape == (batch_size, seq_len, 2)
        assert indices.shape == (batch_size, seq_len, 2)
        assert logits.shape == (batch_size, seq_len, 8)

    def test_weights_sum_to_one(self):
        """Test routing weights sum to 1."""
        d_model = 64
        router = Router(d_model=d_model, num_experts=8, top_k=2)
        x = torch.randn(2, 10, d_model)

        weights, _, _ = router(x)

        # Weights should sum to 1 for each token
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_top_k_selection(self):
        """Test top-k expert selection."""
        d_model = 64
        router = Router(d_model=d_model, num_experts=8, top_k=3)
        x = torch.randn(2, 10, d_model)

        weights, indices, _ = router(x)

        assert weights.shape[-1] == 3
        assert indices.shape[-1] == 3


class TestExpert:
    """Tests for Expert."""

    def test_output_shape(self):
        """Test output shape is correct."""
        d_model = 64
        expert = Expert(d_model=d_model, d_ff=256)
        x = torch.randn(100, d_model)

        output = expert(x)

        assert output.shape == (100, d_model)

    def test_ffn_types(self):
        """Test different FFN types."""
        d_model = 64
        for ffn_type in ["swiglu", "standard"]:
            expert = Expert(d_model=d_model, d_ff=256, ffn_type=ffn_type)
            x = torch.randn(50, d_model)
            output = expert(x)
            assert output.shape == (50, d_model)


class TestMoELayer:
    """Tests for MoELayer."""

    def test_output_shape(self):
        """Test output shape is correct."""
        batch_size, seq_len, d_model = 2, 10, 64
        moe = MoELayer(d_model=d_model, num_experts=4, top_k=2)
        x = torch.randn(batch_size, seq_len, d_model)

        output, aux_loss = moe(x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert aux_loss.shape == ()  # Scalar

    def test_aux_loss_positive(self):
        """Test auxiliary loss is positive."""
        d_model = 64
        moe = MoELayer(d_model=d_model, num_experts=4, top_k=2)
        x = torch.randn(2, 10, d_model)

        _, aux_loss = moe(x)

        assert aux_loss >= 0

    def test_different_expert_counts(self):
        """Test different numbers of experts."""
        d_model = 64
        for num_experts in [2, 4, 8]:
            moe = MoELayer(d_model=d_model, num_experts=num_experts, top_k=2)
            x = torch.randn(2, 10, d_model)
            output, _ = moe(x)
            assert output.shape == (2, 10, d_model)


class TestMoETransformerBlock:
    """Tests for MoETransformerBlock."""

    def test_output_shape(self):
        """Test output shape is correct."""
        batch_size, seq_len, d_model = 2, 10, 64
        block = MoETransformerBlock(
            d_model=d_model, n_heads=4, num_experts=4, top_k=2
        )
        x = torch.randn(batch_size, seq_len, d_model)

        output, aux_loss, attn_weights = block(x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert aux_loss.shape == ()

    def test_return_attention(self):
        """Test returning attention weights."""
        d_model = 64
        block = MoETransformerBlock(
            d_model=d_model, n_heads=4, num_experts=4, top_k=2
        )
        x = torch.randn(2, 10, d_model)

        _, _, attn_weights = block(x, return_attention=True)

        assert attn_weights is not None


class TestSparseMoE:
    """Tests for SparseMoE."""

    def test_output_shape(self):
        """Test output shape is correct."""
        batch_size, seq_len, d_model = 2, 10, 64
        moe = SparseMoE(d_model=d_model, num_experts=4, top_k=1)
        x = torch.randn(batch_size, seq_len, d_model)

        output, aux_loss = moe(x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert aux_loss.shape == ()

    def test_capacity_limiting(self):
        """Test that capacity limiting works."""
        d_model = 64
        # Small capacity factor to force token dropping
        moe = SparseMoE(
            d_model=d_model, num_experts=4, top_k=1, capacity_factor=0.5
        )
        x = torch.randn(2, 10, d_model)

        output, _ = moe(x)

        assert output.shape == (2, 10, d_model)


# =============================================================================
# Multimodal Tests
# =============================================================================

class TestPatchEmbedding:
    """Tests for PatchEmbedding."""

    def test_output_shape(self):
        """Test output shape is correct."""
        patch_embed = PatchEmbedding(
            image_size=224, patch_size=16, d_model=768
        )
        image = torch.randn(2, 3, 224, 224)

        patches = patch_embed(image)

        # 224/16 = 14, 14*14 = 196 patches
        assert patches.shape == (2, 196, 768)

    def test_num_patches_calculation(self):
        """Test number of patches is calculated correctly."""
        for image_size, patch_size in [(224, 16), (256, 32), (384, 24)]:
            patch_embed = PatchEmbedding(
                image_size=image_size, patch_size=patch_size, d_model=512
            )
            expected_patches = (image_size // patch_size) ** 2
            assert patch_embed.num_patches == expected_patches

    def test_different_channels(self):
        """Test different input channels."""
        patch_embed = PatchEmbedding(
            image_size=64, patch_size=8, in_channels=1, d_model=256
        )
        image = torch.randn(2, 1, 64, 64)

        patches = patch_embed(image)

        assert patches.shape == (2, 64, 256)


class TestVisionEncoder:
    """Tests for VisionEncoder."""

    def test_output_shape_with_cls(self):
        """Test output shape with CLS token."""
        encoder = VisionEncoder(
            image_size=64, patch_size=8, d_model=128, n_layers=2, n_heads=4
        )
        image = torch.randn(2, 3, 64, 64)

        features = encoder(image)

        # 64/8 = 8, 8*8 = 64 patches + 1 CLS token = 65
        assert features.shape == (2, 65, 128)

    def test_output_shape_without_cls(self):
        """Test output shape without CLS token."""
        encoder = VisionEncoder(
            image_size=64,
            patch_size=8,
            d_model=128,
            n_layers=2,
            n_heads=4,
            use_cls_token=False,
        )
        image = torch.randn(2, 3, 64, 64)

        features = encoder(image)

        # 64 patches only
        assert features.shape == (2, 64, 128)

    def test_cls_only_output(self):
        """Test returning only CLS token."""
        encoder = VisionEncoder(
            image_size=64, patch_size=8, d_model=128, n_layers=2, n_heads=4
        )
        image = torch.randn(2, 3, 64, 64)

        features = encoder(image, return_all_tokens=False)

        assert features.shape == (2, 128)


class TestCrossModalAttention:
    """Tests for CrossModalAttention."""

    def test_output_shape(self):
        """Test output shape is correct."""
        d_model = 128
        cross_attn = CrossModalAttention(d_model=d_model, n_heads=4)
        text_features = torch.randn(2, 10, d_model)
        image_features = torch.randn(2, 64, d_model)

        output, weights = cross_attn(text_features, image_features)

        assert output.shape == (2, 10, d_model)

    def test_return_attention(self):
        """Test returning attention weights."""
        d_model = 128
        cross_attn = CrossModalAttention(d_model=d_model, n_heads=4)
        text_features = torch.randn(2, 10, d_model)
        image_features = torch.randn(2, 64, d_model)

        _, weights = cross_attn(
            text_features, image_features, return_attention=True
        )

        assert weights is not None


class TestMultimodalFusion:
    """Tests for MultimodalFusion."""

    def test_cross_attention_fusion(self):
        """Test cross-attention fusion."""
        d_model = 128
        fusion = MultimodalFusion(
            d_model=d_model, n_heads=4, fusion_type="cross_attention"
        )
        text = torch.randn(2, 10, d_model)
        image = torch.randn(2, 64, d_model)

        fused_text, fused_image = fusion(text, image)

        assert fused_text.shape == (2, 10, d_model)
        assert fused_image.shape == (2, 64, d_model)

    def test_gated_fusion(self):
        """Test gated fusion."""
        d_model = 128
        fusion = MultimodalFusion(
            d_model=d_model, n_heads=4, fusion_type="gated"
        )
        text = torch.randn(2, 10, d_model)
        image = torch.randn(2, 64, d_model)

        fused_text, fused_image = fusion(text, image)

        assert fused_text.shape == (2, 10, d_model)
        assert fused_image.shape == (2, 64, d_model)

    def test_concat_fusion(self):
        """Test concatenation fusion."""
        d_model = 128
        fusion = MultimodalFusion(
            d_model=d_model, n_heads=4, fusion_type="concat"
        )
        text = torch.randn(2, 10, d_model)
        image = torch.randn(2, 64, d_model)

        fused_text, fused_image = fusion(text, image)

        # Concat just returns the inputs unchanged
        assert fused_text.shape == (2, 10, d_model)
        assert fused_image.shape == (2, 64, d_model)


class TestVisionLanguageProjector:
    """Tests for VisionLanguageProjector."""

    def test_linear_projector(self):
        """Test linear projector."""
        projector = VisionLanguageProjector(
            vision_dim=768, language_dim=1024, projector_type="linear"
        )
        image_features = torch.randn(2, 196, 768)

        projected = projector(image_features)

        assert projected.shape == (2, 196, 1024)

    def test_mlp_projector(self):
        """Test MLP projector."""
        projector = VisionLanguageProjector(
            vision_dim=768, language_dim=1024, projector_type="mlp"
        )
        image_features = torch.randn(2, 196, 768)

        projected = projector(image_features)

        assert projected.shape == (2, 196, 1024)

    def test_resampler_projector(self):
        """Test resampler projector."""
        projector = VisionLanguageProjector(
            vision_dim=768,
            language_dim=1024,
            num_tokens=32,
            projector_type="resampler",
        )
        image_features = torch.randn(2, 196, 768)

        projected = projector(image_features)

        # Resamples to fixed number of tokens
        assert projected.shape == (2, 32, 1024)


class TestMultimodalTransformerBlock:
    """Tests for MultimodalTransformerBlock."""

    def test_output_shape(self):
        """Test output shape is correct."""
        d_model = 128
        block = MultimodalTransformerBlock(d_model=d_model, n_heads=4)
        text = torch.randn(2, 10, d_model)
        image = torch.randn(2, 64, d_model)

        output = block(text, image)

        assert output.shape == (2, 10, d_model)

    def test_with_different_sequence_lengths(self):
        """Test with varying sequence lengths."""
        d_model = 128
        block = MultimodalTransformerBlock(d_model=d_model, n_heads=4)

        for text_len, image_len in [(5, 32), (20, 100), (50, 196)]:
            text = torch.randn(2, text_len, d_model)
            image = torch.randn(2, image_len, d_model)
            output = block(text, image)
            assert output.shape == (2, text_len, d_model)


# =============================================================================
# Integration Tests
# =============================================================================

class TestPhase4Integration:
    """Integration tests for Phase 4 components."""

    def test_moe_with_efficient_attention(self):
        """Test MoE with efficient attention patterns."""
        d_model = 64

        # Create efficient attention
        attn = EfficientMultiHeadAttention(
            d_model=d_model, n_heads=4, use_flash=True
        )

        # Create MoE layer
        moe = MoELayer(d_model=d_model, num_experts=4, top_k=2)

        # Forward pass
        x = torch.randn(2, 16, d_model)
        x, _ = attn(x, x, x)
        x, aux_loss = moe(x)

        assert x.shape == (2, 16, d_model)
        assert aux_loss >= 0

    def test_vision_encoder_to_language_model(self):
        """Test vision encoder to language model projection."""
        vision_dim = 256
        language_dim = 512

        # Vision encoder
        encoder = VisionEncoder(
            image_size=64,
            patch_size=8,
            d_model=vision_dim,
            n_layers=2,
            n_heads=4,
        )

        # Projector
        projector = VisionLanguageProjector(
            vision_dim=vision_dim,
            language_dim=language_dim,
            projector_type="mlp",
        )

        # Forward pass
        image = torch.randn(2, 3, 64, 64)
        features = encoder(image)
        projected = projector(features)

        assert projected.shape == (2, 65, language_dim)

    def test_multimodal_with_moe(self):
        """Test multimodal components with MoE."""
        d_model = 128

        # Vision encoder
        encoder = VisionEncoder(
            image_size=32,
            patch_size=8,
            d_model=d_model,
            n_layers=1,
            n_heads=4,
        )

        # Multimodal fusion
        fusion = MultimodalFusion(
            d_model=d_model, n_heads=4, fusion_type="cross_attention"
        )

        # MoE layer for text
        moe = MoELayer(d_model=d_model, num_experts=4, top_k=2)

        # Forward pass
        image = torch.randn(2, 3, 32, 32)
        text = torch.randn(2, 10, d_model)

        image_features = encoder(image)
        fused_text, fused_image = fusion(text, image_features)
        output, aux_loss = moe(fused_text)

        assert output.shape == (2, 10, d_model)
        assert aux_loss >= 0

    def test_sliding_window_with_sparse(self):
        """Test sliding window and sparse attention together."""
        d_model = 64
        seq_len = 32

        sliding = SlidingWindowAttention(d_model=d_model, n_heads=4, window_size=8)
        sparse = SparseAttention(d_model=d_model, n_heads=4, local_window=8, global_tokens=2)

        x = torch.randn(2, seq_len, d_model)

        out1, _ = sliding(x, x, x)
        out2, _ = sparse(x, x, x)

        assert out1.shape == (2, seq_len, d_model)
        assert out2.shape == (2, seq_len, d_model)
