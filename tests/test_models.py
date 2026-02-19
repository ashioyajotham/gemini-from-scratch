"""Tests for transformer model components."""

import math
import pytest
import torch
import torch.nn as nn

from src.models.attention import (
    scaled_dot_product_attention,
    MultiHeadAttention,
    CausalSelfAttention,
    GroupedQueryAttention,
    create_causal_mask,
    create_padding_mask,
)
from src.models.embeddings import (
    TokenEmbedding,
    SinusoidalPositionalEncoding,
    LearnedPositionalEmbedding,
    RotaryPositionalEmbedding,
    TransformerEmbedding,
)
from src.models.feedforward import (
    FeedForward,
    SwiGLU,
    GeGLU,
    create_ffn,
)
from src.models.transformer_block import (
    RMSNorm,
    TransformerBlock,
    DecoderBlock,
    TransformerBlockStack,
)


class TestScaledDotProductAttention:
    """Tests for scaled dot-product attention."""

    def test_output_shape(self):
        """Output should have correct shape."""
        batch, heads, seq_len, d_k = 2, 8, 10, 64
        q = torch.randn(batch, heads, seq_len, d_k)
        k = torch.randn(batch, heads, seq_len, d_k)
        v = torch.randn(batch, heads, seq_len, d_k)

        output, weights = scaled_dot_product_attention(q, k, v)

        assert output.shape == (batch, heads, seq_len, d_k)
        assert weights.shape == (batch, heads, seq_len, seq_len)

    def test_attention_weights_sum_to_one(self):
        """Attention weights should sum to 1 along last dimension."""
        q = torch.randn(2, 4, 5, 32)
        k = torch.randn(2, 4, 5, 32)
        v = torch.randn(2, 4, 5, 32)

        _, weights = scaled_dot_product_attention(q, k, v)

        # Sum along last dimension should be ~1
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_masking(self):
        """Masked positions should have zero attention weight."""
        q = torch.randn(1, 1, 3, 4)
        k = torch.randn(1, 1, 3, 4)
        v = torch.randn(1, 1, 3, 4)

        # Mask: only attend to first position
        mask = torch.tensor([[[[1, 0, 0], [1, 1, 0], [1, 1, 1]]]])

        _, weights = scaled_dot_product_attention(q, k, v, mask=mask)

        # Masked positions should have ~0 weight
        assert weights[0, 0, 0, 1] < 1e-6
        assert weights[0, 0, 0, 2] < 1e-6


class TestMultiHeadAttention:
    """Tests for multi-head attention."""

    def test_output_shape(self):
        """Output should match input shape."""
        d_model, n_heads = 512, 8
        mha = MultiHeadAttention(d_model, n_heads)

        x = torch.randn(2, 10, d_model)
        output, _ = mha(x, x, x)

        assert output.shape == x.shape

    def test_return_attention(self):
        """Should return attention weights when requested."""
        mha = MultiHeadAttention(256, 4)
        x = torch.randn(2, 5, 256)

        output, weights = mha(x, x, x, return_attention=True)

        assert weights is not None
        assert weights.shape == (2, 4, 5, 5)

    def test_get_attention_weights(self):
        """Should store and return attention weights."""
        mha = MultiHeadAttention(128, 2)
        x = torch.randn(1, 4, 128)

        mha(x, x, x)
        weights = mha.get_attention_weights()

        assert weights is not None
        assert weights.shape == (1, 2, 4, 4)

    def test_d_model_divisibility(self):
        """Should raise error if d_model not divisible by n_heads."""
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=100, n_heads=3)


class TestCausalSelfAttention:
    """Tests for causal self-attention."""

    def test_output_shape(self):
        """Output should match input shape."""
        attn = CausalSelfAttention(d_model=256, n_heads=4, max_seq_len=100)
        x = torch.randn(2, 10, 256)

        output, _ = attn(x)

        assert output.shape == x.shape

    def test_causal_masking(self):
        """Future positions should not be attended to."""
        attn = CausalSelfAttention(d_model=64, n_heads=2, max_seq_len=10)
        x = torch.randn(1, 5, 64)

        attn(x)
        weights = attn.get_attention_weights()

        # Upper triangular (future positions) should be ~0
        for i in range(5):
            for j in range(i + 1, 5):
                assert weights[0, 0, i, j] < 1e-6


class TestGroupedQueryAttention:
    """Tests for grouped-query attention."""

    def test_output_shape(self):
        """Output should match input shape."""
        gqa = GroupedQueryAttention(d_model=512, n_heads=8, n_kv_heads=2)
        x = torch.randn(2, 10, 512)

        output, _ = gqa(x, x, x)

        assert output.shape == x.shape

    def test_fewer_kv_heads(self):
        """Should work with fewer KV heads than query heads."""
        gqa = GroupedQueryAttention(d_model=256, n_heads=8, n_kv_heads=2)

        # Check projection dimensions
        assert gqa.w_q.out_features == 256  # Full d_model
        assert gqa.w_k.out_features == 64   # d_k * n_kv_heads = 32 * 2

    def test_invalid_head_ratio(self):
        """Should raise error if n_heads not divisible by n_kv_heads."""
        with pytest.raises(AssertionError):
            GroupedQueryAttention(d_model=256, n_heads=8, n_kv_heads=3)


class TestMaskCreation:
    """Tests for mask creation utilities."""

    def test_causal_mask_shape(self):
        """Causal mask should have correct shape."""
        mask = create_causal_mask(10)
        assert mask.shape == (1, 1, 10, 10)

    def test_causal_mask_lower_triangular(self):
        """Causal mask should be lower triangular."""
        mask = create_causal_mask(5).squeeze()
        expected = torch.tril(torch.ones(5, 5))
        assert torch.equal(mask, expected)

    def test_padding_mask(self):
        """Padding mask should mask positions beyond length."""
        lengths = torch.tensor([3, 5, 2])
        mask = create_padding_mask(lengths, max_len=5)

        assert mask.shape == (3, 1, 1, 5)
        # First sequence: valid for positions 0, 1, 2
        assert mask[0, 0, 0, 2] == 1
        assert mask[0, 0, 0, 3] == 0


class TestTokenEmbedding:
    """Tests for token embedding."""

    def test_output_shape(self):
        """Output should have correct shape."""
        embed = TokenEmbedding(vocab_size=1000, d_model=256)
        tokens = torch.tensor([[1, 2, 3, 4]])

        output = embed(tokens)

        assert output.shape == (1, 4, 256)

    def test_padding_idx(self):
        """Padding tokens should have zero embeddings."""
        embed = TokenEmbedding(vocab_size=100, d_model=64, padding_idx=0)

        # Padding embedding should be zeros
        assert torch.allclose(
            embed.embedding.weight[0],
            torch.zeros(64)
        )


class TestSinusoidalPositionalEncoding:
    """Tests for sinusoidal positional encoding."""

    def test_output_shape(self):
        """Output should match input shape."""
        pe = SinusoidalPositionalEncoding(d_model=512, max_seq_len=100)
        x = torch.randn(2, 10, 512)

        output = pe(x)

        assert output.shape == x.shape

    def test_encoding_deterministic(self):
        """Encoding should be deterministic."""
        pe = SinusoidalPositionalEncoding(d_model=64, max_seq_len=50)

        enc1 = pe.get_encoding(10)
        enc2 = pe.get_encoding(10)

        assert torch.equal(enc1, enc2)

    def test_different_positions_different_encodings(self):
        """Different positions should have different encodings."""
        pe = SinusoidalPositionalEncoding(d_model=64, max_seq_len=50)
        enc = pe.get_encoding(10)

        # Each position should be unique
        for i in range(10):
            for j in range(i + 1, 10):
                assert not torch.allclose(enc[i], enc[j])


class TestLearnedPositionalEmbedding:
    """Tests for learned positional embedding."""

    def test_output_shape(self):
        """Output should match input shape."""
        pe = LearnedPositionalEmbedding(max_seq_len=100, d_model=256)
        x = torch.randn(2, 10, 256)

        output = pe(x)

        assert output.shape == x.shape


class TestRotaryPositionalEmbedding:
    """Tests for RoPE."""

    def test_output_shape(self):
        """Output should match input shape."""
        rope = RotaryPositionalEmbedding(d_model=64, max_seq_len=100)
        q = torch.randn(2, 4, 10, 64)
        k = torch.randn(2, 4, 10, 64)

        q_rot, k_rot = rope(q, k)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_even_dimension_required(self):
        """Should raise error for odd d_model."""
        with pytest.raises(AssertionError):
            RotaryPositionalEmbedding(d_model=63)


class TestTransformerEmbedding:
    """Tests for combined transformer embedding."""

    def test_sinusoidal(self):
        """Should work with sinusoidal encoding."""
        embed = TransformerEmbedding(
            vocab_size=1000, d_model=256, pos_encoding="sinusoidal"
        )
        tokens = torch.tensor([[1, 2, 3]])
        output = embed(tokens)

        assert output.shape == (1, 3, 256)

    def test_learned(self):
        """Should work with learned encoding."""
        embed = TransformerEmbedding(
            vocab_size=1000, d_model=256, pos_encoding="learned"
        )
        tokens = torch.tensor([[1, 2, 3]])
        output = embed(tokens)

        assert output.shape == (1, 3, 256)


class TestFeedForward:
    """Tests for feed-forward networks."""

    def test_output_shape(self):
        """Output should match input shape."""
        ffn = FeedForward(d_model=512, d_ff=2048)
        x = torch.randn(2, 10, 512)

        output = ffn(x)

        assert output.shape == x.shape

    def test_default_d_ff(self):
        """Default d_ff should be 4 * d_model."""
        ffn = FeedForward(d_model=256)
        assert ffn.d_ff == 1024

    def test_activations(self):
        """Should support different activations."""
        for activation in ["relu", "gelu", "silu"]:
            ffn = FeedForward(d_model=64, activation=activation)
            x = torch.randn(1, 5, 64)
            output = ffn(x)
            assert output.shape == x.shape


class TestSwiGLU:
    """Tests for SwiGLU feed-forward."""

    def test_output_shape(self):
        """Output should match input shape."""
        swiglu = SwiGLU(d_model=512)
        x = torch.randn(2, 10, 512)

        output = swiglu(x)

        assert output.shape == x.shape

    def test_gating_mechanism(self):
        """SwiGLU should use gating."""
        swiglu = SwiGLU(d_model=64)

        # Should have gate, up, and down projections
        assert hasattr(swiglu, 'w_gate')
        assert hasattr(swiglu, 'w_up')
        assert hasattr(swiglu, 'w_down')


class TestGeGLU:
    """Tests for GeGLU feed-forward."""

    def test_output_shape(self):
        """Output should match input shape."""
        geglu = GeGLU(d_model=256)
        x = torch.randn(2, 10, 256)

        output = geglu(x)

        assert output.shape == x.shape


class TestCreateFFN:
    """Tests for FFN factory function."""

    def test_standard(self):
        """Should create standard FFN."""
        ffn = create_ffn(d_model=256, ffn_type="standard")
        assert isinstance(ffn, FeedForward)

    def test_swiglu(self):
        """Should create SwiGLU."""
        ffn = create_ffn(d_model=256, ffn_type="swiglu")
        assert isinstance(ffn, SwiGLU)

    def test_geglu(self):
        """Should create GeGLU."""
        ffn = create_ffn(d_model=256, ffn_type="geglu")
        assert isinstance(ffn, GeGLU)


class TestRMSNorm:
    """Tests for RMSNorm."""

    def test_output_shape(self):
        """Output should match input shape."""
        norm = RMSNorm(512)
        x = torch.randn(2, 10, 512)

        output = norm(x)

        assert output.shape == x.shape

    def test_normalization(self):
        """Output should be approximately normalized."""
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64) * 10  # Large values

        output = norm(x)

        # RMS of output should be close to 1
        rms = torch.sqrt(output.pow(2).mean(-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)


class TestTransformerBlock:
    """Tests for transformer block."""

    def test_output_shape(self):
        """Output should match input shape."""
        block = TransformerBlock(d_model=512, n_heads=8)
        x = torch.randn(2, 10, 512)

        output, _ = block(x)

        assert output.shape == x.shape

    def test_pre_ln(self):
        """Pre-LN should be default."""
        block = TransformerBlock(d_model=256, n_heads=4)
        assert block.norm_first is True

    def test_post_ln(self):
        """Should support Post-LN."""
        block = TransformerBlock(d_model=256, n_heads=4, norm_first=False)
        x = torch.randn(2, 5, 256)

        output, _ = block(x)

        assert output.shape == x.shape

    def test_return_attention(self):
        """Should return attention weights when requested."""
        block = TransformerBlock(d_model=128, n_heads=2)
        x = torch.randn(2, 5, 128)

        output, weights = block(x, return_attention=True)

        assert weights is not None
        assert weights.shape == (2, 2, 5, 5)

    def test_rmsnorm(self):
        """Should support RMSNorm."""
        block = TransformerBlock(d_model=256, n_heads=4, norm_type="rmsnorm")
        assert isinstance(block.norm1, RMSNorm)


class TestDecoderBlock:
    """Tests for decoder-only block."""

    def test_output_shape(self):
        """Output should match input shape."""
        block = DecoderBlock(d_model=512, n_heads=8)
        x = torch.randn(2, 10, 512)

        output, _ = block(x)

        assert output.shape == x.shape

    def test_causal_masking(self):
        """Should automatically apply causal masking."""
        block = DecoderBlock(d_model=64, n_heads=2, max_seq_len=20)
        x = torch.randn(1, 5, 64)

        block(x)
        weights = block.get_attention_weights()

        # Check causal masking
        for i in range(5):
            for j in range(i + 1, 5):
                assert weights[0, 0, i, j] < 1e-6

    def test_swiglu_default(self):
        """Should use SwiGLU by default."""
        block = DecoderBlock(d_model=256, n_heads=4)
        assert isinstance(block.ffn, SwiGLU)

    def test_gqa(self):
        """Should support grouped-query attention."""
        block = DecoderBlock(d_model=256, n_heads=8, n_kv_heads=2)
        x = torch.randn(2, 5, 256)

        output, _ = block(x)

        assert output.shape == x.shape


class TestTransformerBlockStack:
    """Tests for transformer block stack."""

    def test_output_shape(self):
        """Output should match input shape."""
        stack = TransformerBlockStack(n_layers=4, d_model=256, n_heads=4)
        x = torch.randn(2, 10, 256)

        output, _ = stack(x)

        assert output.shape == x.shape

    def test_n_layers(self):
        """Should have correct number of layers."""
        stack = TransformerBlockStack(n_layers=6, d_model=128, n_heads=2)
        assert len(stack.blocks) == 6

    def test_decoder_stack(self):
        """Should support decoder blocks."""
        stack = TransformerBlockStack(
            n_layers=3, d_model=256, n_heads=4, block_type="decoder"
        )
        x = torch.randn(2, 5, 256)

        output, _ = stack(x)

        assert output.shape == x.shape

    def test_return_all_attention(self):
        """Should return attention from all layers."""
        stack = TransformerBlockStack(n_layers=3, d_model=128, n_heads=2)
        x = torch.randn(2, 5, 128)

        output, all_attn = stack(x, return_all_attention=True)

        assert len(all_attn) == 3
        for attn in all_attn:
            assert attn.shape == (2, 2, 5, 5)


class TestGradientFlow:
    """Tests for gradient flow through components."""

    def test_attention_gradients(self):
        """Gradients should flow through attention."""
        mha = MultiHeadAttention(64, 2)
        x = torch.randn(1, 5, 64, requires_grad=True)

        output, _ = mha(x, x, x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_transformer_block_gradients(self):
        """Gradients should flow through transformer block."""
        block = TransformerBlock(d_model=64, n_heads=2)
        x = torch.randn(1, 5, 64, requires_grad=True)

        output, _ = block(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_decoder_block_gradients(self):
        """Gradients should flow through decoder block."""
        block = DecoderBlock(d_model=64, n_heads=2)
        x = torch.randn(1, 5, 64, requires_grad=True)

        output, _ = block(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
