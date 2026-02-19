"""
Sampling strategies for text generation.

This module implements:
- Greedy decoding
- Temperature sampling
- Top-k sampling
- Top-p (nucleus) sampling
- Combined sampling strategies
"""

from typing import Optional, Callable

import torch
import torch.nn.functional as F


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Apply temperature scaling to logits.

    Higher temperature = more random, lower = more deterministic.

    Args:
        logits: Logits of shape (..., vocab_size).
        temperature: Temperature value (> 0).

    Returns:
        Scaled logits.

    Example:
        >>> logits = model(input_ids)[:, -1, :]
        >>> scaled = apply_temperature(logits, temperature=0.8)
    """
    if temperature == 1.0:
        return logits
    return logits / temperature


def top_k_filtering(
    logits: torch.Tensor,
    k: int,
    filter_value: float = float("-inf"),
) -> torch.Tensor:
    """
    Filter logits to keep only top-k values.

    Args:
        logits: Logits of shape (..., vocab_size).
        k: Number of top values to keep.
        filter_value: Value to assign to filtered positions.

    Returns:
        Filtered logits.

    Example:
        >>> logits = model(input_ids)[:, -1, :]
        >>> filtered = top_k_filtering(logits, k=50)
    """
    if k <= 0:
        return logits

    # Get the k-th largest value
    top_k_values, _ = torch.topk(logits, k, dim=-1)
    threshold = top_k_values[..., -1, None]

    # Filter values below threshold
    return torch.where(logits < threshold, filter_value, logits)


def top_p_filtering(
    logits: torch.Tensor,
    p: float,
    filter_value: float = float("-inf"),
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """
    Filter logits using nucleus (top-p) sampling.

    Keeps the smallest set of tokens whose cumulative probability
    exceeds p.

    Args:
        logits: Logits of shape (..., vocab_size).
        p: Cumulative probability threshold (0 < p <= 1).
        filter_value: Value to assign to filtered positions.
        min_tokens_to_keep: Minimum number of tokens to keep.

    Returns:
        Filtered logits.

    Example:
        >>> logits = model(input_ids)[:, -1, :]
        >>> filtered = top_p_filtering(logits, p=0.9)
    """
    if p >= 1.0:
        return logits

    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Find cutoff point
    sorted_indices_to_remove = cumulative_probs > p

    # Shift to keep at least one token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., :min_tokens_to_keep] = False

    # Scatter back to original order
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )

    return torch.where(indices_to_remove, filter_value, logits)


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    do_sample: bool = True,
) -> torch.Tensor:
    """
    Sample tokens from logits with various strategies.

    Args:
        logits: Logits of shape (batch, vocab_size).
        temperature: Sampling temperature.
        top_k: Top-k filtering (None to disable).
        top_p: Top-p filtering (None to disable).
        do_sample: If False, use greedy decoding.

    Returns:
        Sampled token IDs of shape (batch, 1).

    Example:
        >>> logits = model(input_ids)[:, -1, :]
        >>> next_token = sample_from_logits(
        ...     logits, temperature=0.8, top_p=0.9
        ... )
    """
    # Apply temperature
    logits = apply_temperature(logits, temperature)

    # Apply top-k filtering
    if top_k is not None and top_k > 0:
        logits = top_k_filtering(logits, top_k)

    # Apply top-p filtering
    if top_p is not None and top_p < 1.0:
        logits = top_p_filtering(logits, top_p)

    # Sample or greedy
    if do_sample:
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
    else:
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

    return next_token


def repetition_penalty(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    penalty: float = 1.0,
) -> torch.Tensor:
    """
    Apply repetition penalty to logits.

    Reduces probability of tokens that have already been generated.

    Args:
        logits: Logits of shape (batch, vocab_size).
        generated_ids: Previously generated token IDs of shape (batch, seq_len).
        penalty: Repetition penalty factor (> 1 = penalize repetition).

    Returns:
        Penalized logits.

    Example:
        >>> logits = apply_repetition_penalty(logits, generated_ids, penalty=1.2)
    """
    if penalty == 1.0:
        return logits

    # Gather logits for generated tokens
    batch_size = logits.size(0)

    for i in range(batch_size):
        unique_tokens = generated_ids[i].unique()
        for token in unique_tokens:
            if logits[i, token] > 0:
                logits[i, token] /= penalty
            else:
                logits[i, token] *= penalty

    return logits


class SamplingConfig:
    """
    Configuration for text generation sampling.

    Args:
        temperature: Sampling temperature.
        top_k: Top-k filtering (0 to disable).
        top_p: Top-p filtering (1.0 to disable).
        repetition_penalty: Repetition penalty (1.0 to disable).
        do_sample: If False, use greedy decoding.
        max_new_tokens: Maximum number of new tokens to generate.
        min_new_tokens: Minimum number of new tokens to generate.
        eos_token_id: End of sequence token ID.
        pad_token_id: Padding token ID.

    Example:
        >>> config = SamplingConfig(temperature=0.8, top_p=0.9)
    """

    def __init__(
        self,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        max_new_tokens: int = 100,
        min_new_tokens: int = 0,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    config: Optional[SamplingConfig] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Generate text using the model.

    Args:
        model: Language model with forward method.
        input_ids: Starting token IDs of shape (batch, seq_len).
        config: Sampling configuration.
        **kwargs: Override config parameters.

    Returns:
        Generated token IDs of shape (batch, seq_len + new_tokens).

    Example:
        >>> generated = generate(
        ...     model, input_ids,
        ...     config=SamplingConfig(temperature=0.8, top_p=0.9)
        ... )
    """
    if config is None:
        config = SamplingConfig(**kwargs)
    else:
        # Override config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

    model.eval()
    device = input_ids.device
    batch_size = input_ids.size(0)

    # Track which sequences are done
    done = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # Generate tokens
    for step in range(config.max_new_tokens):
        # Get logits for the last position
        outputs = model(input_ids)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        next_token_logits = logits[:, -1, :]

        # Apply repetition penalty
        if config.repetition_penalty != 1.0:
            next_token_logits = repetition_penalty(
                next_token_logits, input_ids, config.repetition_penalty
            )

        # Prevent EOS before min_new_tokens
        if step < config.min_new_tokens and config.eos_token_id is not None:
            next_token_logits[:, config.eos_token_id] = float("-inf")

        # Sample next token
        next_token = sample_from_logits(
            next_token_logits,
            temperature=config.temperature,
            top_k=config.top_k if config.top_k > 0 else None,
            top_p=config.top_p if config.top_p < 1.0 else None,
            do_sample=config.do_sample,
        )

        # Replace with pad for done sequences
        if config.pad_token_id is not None:
            next_token = torch.where(
                done.unsqueeze(1),
                torch.full_like(next_token, config.pad_token_id),
                next_token,
            )

        # Append to sequence
        input_ids = torch.cat([input_ids, next_token], dim=1)

        # Check for EOS
        if config.eos_token_id is not None:
            done = done | (next_token.squeeze(-1) == config.eos_token_id)
            if done.all():
                break

    return input_ids
