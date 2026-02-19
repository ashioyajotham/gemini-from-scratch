"""
KV-Cache for efficient autoregressive generation.

This module implements:
- Key-Value cache for attention layers
- Cache management for multi-layer transformers
"""

from typing import Optional, Tuple, List

import torch


class KVCache:
    """
    Key-Value cache for a single attention layer.

    Stores key and value tensors from previous forward passes to avoid
    recomputation during autoregressive generation.

    Args:
        batch_size: Batch size.
        max_seq_len: Maximum sequence length.
        n_heads: Number of attention heads (or KV heads for GQA).
        head_dim: Dimension per head.
        device: Device to store cache on.
        dtype: Data type for cache tensors.

    Example:
        >>> cache = KVCache(batch_size=2, max_seq_len=512, n_heads=8, head_dim=64)
        >>> cache.update(new_keys, new_values)
        >>> keys, values = cache.get()
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        n_heads: int,
        head_dim: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # Initialize empty cache
        self.cache_k = torch.zeros(
            batch_size, n_heads, max_seq_len, head_dim,
            device=device, dtype=dtype
        )
        self.cache_v = torch.zeros(
            batch_size, n_heads, max_seq_len, head_dim,
            device=device, dtype=dtype
        )

        # Current sequence length in cache
        self.seq_len = 0

    def update(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key-value pairs.

        Args:
            keys: New keys of shape (batch, n_heads, new_seq_len, head_dim).
            values: New values of shape (batch, n_heads, new_seq_len, head_dim).

        Returns:
            Tuple of (all_keys, all_values) including cached values.
        """
        new_seq_len = keys.size(2)
        end_pos = self.seq_len + new_seq_len

        # Store new keys and values
        self.cache_k[:, :, self.seq_len:end_pos, :] = keys
        self.cache_v[:, :, self.seq_len:end_pos, :] = values

        self.seq_len = end_pos

        # Return all cached keys and values
        return self.cache_k[:, :, :end_pos, :], self.cache_v[:, :, :end_pos, :]

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get all cached keys and values.

        Returns:
            Tuple of (keys, values) tensors.
        """
        return (
            self.cache_k[:, :, :self.seq_len, :],
            self.cache_v[:, :, :self.seq_len, :],
        )

    def reset(self) -> None:
        """Reset the cache."""
        self.seq_len = 0
        self.cache_k.zero_()
        self.cache_v.zero_()

    def get_seq_len(self) -> int:
        """Get current sequence length in cache."""
        return self.seq_len


class LayerCache:
    """
    Cache manager for all attention layers in a transformer.

    Args:
        n_layers: Number of transformer layers.
        batch_size: Batch size.
        max_seq_len: Maximum sequence length.
        n_heads: Number of attention heads per layer.
        head_dim: Dimension per head.
        device: Device to store cache on.
        dtype: Data type for cache tensors.

    Example:
        >>> cache = LayerCache(
        ...     n_layers=6, batch_size=2, max_seq_len=512,
        ...     n_heads=8, head_dim=64
        ... )
        >>> # During generation:
        >>> keys, values = cache.update(layer_idx, new_k, new_v)
    """

    def __init__(
        self,
        n_layers: int,
        batch_size: int,
        max_seq_len: int,
        n_heads: int,
        head_dim: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.n_layers = n_layers
        self.caches = [
            KVCache(batch_size, max_seq_len, n_heads, head_dim, device, dtype)
            for _ in range(n_layers)
        ]

    def update(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache for a specific layer.

        Args:
            layer_idx: Layer index.
            keys: New keys.
            values: New values.

        Returns:
            Tuple of all cached keys and values for this layer.
        """
        return self.caches[layer_idx].update(keys, values)

    def get(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached keys and values for a specific layer."""
        return self.caches[layer_idx].get()

    def reset(self) -> None:
        """Reset all layer caches."""
        for cache in self.caches:
            cache.reset()

    def get_seq_len(self) -> int:
        """Get current sequence length (same for all layers)."""
        return self.caches[0].get_seq_len() if self.caches else 0


class DynamicCache:
    """
    Dynamic KV-cache that grows as needed.

    More memory efficient than pre-allocated cache for variable-length generation.

    Args:
        n_layers: Number of transformer layers.

    Example:
        >>> cache = DynamicCache(n_layers=6)
        >>> cache.update(layer_idx, keys, values)
    """

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self.key_cache: List[Optional[torch.Tensor]] = [None] * n_layers
        self.value_cache: List[Optional[torch.Tensor]] = [None] * n_layers

    def update(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache for a specific layer.

        Args:
            layer_idx: Layer index.
            keys: New keys of shape (batch, n_heads, new_seq_len, head_dim).
            values: New values.

        Returns:
            Tuple of all cached keys and values for this layer.
        """
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = keys
            self.value_cache[layer_idx] = values
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], keys], dim=2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], values], dim=2
            )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get(self, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get cached keys and values for a specific layer."""
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reset(self) -> None:
        """Reset all caches."""
        self.key_cache = [None] * self.n_layers
        self.value_cache = [None] * self.n_layers

    def get_seq_len(self) -> int:
        """Get current sequence length."""
        if self.key_cache[0] is not None:
            return self.key_cache[0].size(2)
        return 0

    @property
    def seen_tokens(self) -> int:
        """Number of tokens seen (alias for get_seq_len)."""
        return self.get_seq_len()


def create_cache(
    n_layers: int,
    batch_size: int,
    max_seq_len: int,
    n_heads: int,
    head_dim: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    dynamic: bool = False,
) -> LayerCache:
    """
    Factory function to create a KV-cache.

    Args:
        n_layers: Number of transformer layers.
        batch_size: Batch size.
        max_seq_len: Maximum sequence length.
        n_heads: Number of attention heads.
        head_dim: Dimension per head.
        device: Device for cache tensors.
        dtype: Data type for cache tensors.
        dynamic: If True, use dynamic cache (grows as needed).

    Returns:
        Cache object.

    Example:
        >>> cache = create_cache(
        ...     n_layers=6, batch_size=2, max_seq_len=512,
        ...     n_heads=8, head_dim=64
        ... )
    """
    if dynamic:
        return DynamicCache(n_layers)
    else:
        return LayerCache(
            n_layers, batch_size, max_seq_len, n_heads, head_dim, device, dtype
        )
