"""
Text generation utilities.

This module contains:
- Sampling strategies (greedy, top-k, top-p, temperature)
- Beam search
- KV-cache for efficient generation
"""

from .sampling import (
    apply_temperature,
    top_k_filtering,
    top_p_filtering,
    sample_from_logits,
    repetition_penalty,
    SamplingConfig,
    generate,
)

from .cache import (
    KVCache,
    LayerCache,
    DynamicCache,
    create_cache,
)

from .beam_search import (
    BeamHypothesis,
    BeamSearchScorer,
    beam_search,
)

__all__ = [
    # Sampling
    "apply_temperature",
    "top_k_filtering",
    "top_p_filtering",
    "sample_from_logits",
    "repetition_penalty",
    "SamplingConfig",
    "generate",
    # Cache
    "KVCache",
    "LayerCache",
    "DynamicCache",
    "create_cache",
    # Beam search
    "BeamHypothesis",
    "BeamSearchScorer",
    "beam_search",
]
