"""
Beam search implementation for text generation.

This module implements:
- Beam search decoding
- Length normalization
- N-gram blocking
"""

from typing import Optional, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class BeamHypothesis:
    """A single beam hypothesis."""

    tokens: torch.Tensor  # Token IDs
    score: float  # Log probability score
    normalized_score: float  # Length-normalized score


class BeamSearchScorer:
    """
    Scorer for beam search that tracks hypotheses and handles early stopping.

    Args:
        batch_size: Batch size.
        num_beams: Number of beams.
        max_length: Maximum generation length.
        length_penalty: Length normalization factor (> 1 = longer, < 1 = shorter).
        early_stopping: Stop when num_beams hypotheses are complete.
        num_beam_hyps_to_keep: Number of hypotheses to return per batch item.
    """

    def __init__(
        self,
        batch_size: int,
        num_beams: int,
        max_length: int,
        length_penalty: float = 1.0,
        early_stopping: bool = False,
        num_beam_hyps_to_keep: int = 1,
    ):
        self.batch_size = batch_size
        self.num_beams = num_beams
        self.max_length = max_length
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep

        # Track hypotheses for each batch item
        self.hypotheses = [[] for _ in range(batch_size)]
        self.done = [False] * batch_size

    def add(
        self,
        batch_idx: int,
        tokens: torch.Tensor,
        score: float,
    ) -> None:
        """Add a completed hypothesis."""
        length = tokens.size(0)
        normalized_score = score / (length ** self.length_penalty)

        hyp = BeamHypothesis(
            tokens=tokens.clone(),
            score=score,
            normalized_score=normalized_score,
        )

        self.hypotheses[batch_idx].append(hyp)

        # Sort by normalized score
        self.hypotheses[batch_idx].sort(
            key=lambda x: x.normalized_score, reverse=True
        )

        # Keep only top hypotheses
        if len(self.hypotheses[batch_idx]) > self.num_beam_hyps_to_keep:
            self.hypotheses[batch_idx].pop()

    def is_done(self, batch_idx: int, best_score: float, cur_len: int) -> bool:
        """Check if beam search should stop for this batch item."""
        if len(self.hypotheses[batch_idx]) < self.num_beam_hyps_to_keep:
            return False

        if self.early_stopping:
            return True

        # Check if the best possible score can beat current best
        worst_score = self.hypotheses[batch_idx][-1].normalized_score
        best_possible = best_score / (self.max_length ** self.length_penalty)

        return worst_score >= best_possible

    def finalize(
        self,
        input_ids: torch.Tensor,
        scores: torch.Tensor,
        pad_token_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Finalize beam search and return best sequences.

        Returns:
            Tuple of (sequences, scores).
        """
        # Add any remaining beams
        batch_size = input_ids.size(0) // self.num_beams

        for batch_idx in range(batch_size):
            if self.done[batch_idx]:
                continue

            for beam_idx in range(self.num_beams):
                idx = batch_idx * self.num_beams + beam_idx
                tokens = input_ids[idx]
                score = scores[idx].item()
                self.add(batch_idx, tokens, score)

        # Collect best hypotheses
        max_len = max(
            hyp.tokens.size(0)
            for hyps in self.hypotheses
            for hyp in hyps
        )

        sequences = torch.full(
            (batch_size * self.num_beam_hyps_to_keep, max_len),
            pad_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        sequence_scores = torch.zeros(
            batch_size * self.num_beam_hyps_to_keep,
            device=input_ids.device,
        )

        for batch_idx, hyps in enumerate(self.hypotheses):
            for hyp_idx, hyp in enumerate(hyps[:self.num_beam_hyps_to_keep]):
                idx = batch_idx * self.num_beam_hyps_to_keep + hyp_idx
                seq_len = hyp.tokens.size(0)
                sequences[idx, :seq_len] = hyp.tokens
                sequence_scores[idx] = hyp.normalized_score

        return sequences, sequence_scores


@torch.no_grad()
def beam_search(
    model,
    input_ids: torch.Tensor,
    num_beams: int = 4,
    max_new_tokens: int = 50,
    length_penalty: float = 1.0,
    early_stopping: bool = True,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    num_return_sequences: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate text using beam search.

    Args:
        model: Language model.
        input_ids: Starting token IDs of shape (batch, seq_len).
        num_beams: Number of beams.
        max_new_tokens: Maximum number of new tokens.
        length_penalty: Length normalization factor.
        early_stopping: Stop when all beams have finished.
        eos_token_id: End of sequence token ID.
        pad_token_id: Padding token ID.
        num_return_sequences: Number of sequences to return per input.

    Returns:
        Tuple of (generated_ids, scores).

    Example:
        >>> sequences, scores = beam_search(
        ...     model, input_ids, num_beams=4, max_new_tokens=50
        ... )
    """
    model.eval()
    device = input_ids.device
    batch_size = input_ids.size(0)

    if pad_token_id is None:
        pad_token_id = 0

    # Expand input for beam search
    input_ids = input_ids.repeat_interleave(num_beams, dim=0)
    beam_scores = torch.zeros(batch_size * num_beams, device=device)

    # Initialize scorer
    scorer = BeamSearchScorer(
        batch_size=batch_size,
        num_beams=num_beams,
        max_length=input_ids.size(1) + max_new_tokens,
        length_penalty=length_penalty,
        early_stopping=early_stopping,
        num_beam_hyps_to_keep=num_return_sequences,
    )

    # Generate tokens
    for step in range(max_new_tokens):
        # Get model predictions
        outputs = model(input_ids)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        next_token_logits = logits[:, -1, :]
        vocab_size = next_token_logits.size(-1)

        # Compute scores
        next_token_scores = F.log_softmax(next_token_logits, dim=-1)

        # Add beam scores
        next_token_scores = next_token_scores + beam_scores.unsqueeze(-1)

        # Reshape for beam selection: (batch, num_beams * vocab)
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

        # Select top-k tokens
        next_scores, next_tokens = torch.topk(
            next_token_scores, 2 * num_beams, dim=-1, largest=True, sorted=True
        )

        # Compute beam and token indices
        next_beam_indices = next_tokens // vocab_size
        next_tokens = next_tokens % vocab_size

        # Build next beams
        next_input_ids = []
        next_beam_scores = []

        for batch_idx in range(batch_size):
            if scorer.done[batch_idx]:
                # Pad with dummy tokens
                next_input_ids.extend([input_ids[batch_idx * num_beams]] * num_beams)
                next_beam_scores.extend([0.0] * num_beams)
                continue

            beam_idx_offset = batch_idx * num_beams
            beams_added = 0

            for beam_rank, (score, token, beam_idx) in enumerate(
                zip(
                    next_scores[batch_idx],
                    next_tokens[batch_idx],
                    next_beam_indices[batch_idx],
                )
            ):
                global_beam_idx = beam_idx_offset + beam_idx.item()

                # Check if this is an EOS token
                if eos_token_id is not None and token.item() == eos_token_id:
                    # Add completed hypothesis
                    scorer.add(
                        batch_idx,
                        torch.cat([
                            input_ids[global_beam_idx],
                            token.unsqueeze(0),
                        ]),
                        score.item(),
                    )
                else:
                    # Add to next beams
                    next_input_ids.append(
                        torch.cat([
                            input_ids[global_beam_idx],
                            token.unsqueeze(0),
                        ])
                    )
                    next_beam_scores.append(score.item())
                    beams_added += 1

                if beams_added >= num_beams:
                    break

            # Check if done
            best_score = next_scores[batch_idx][0].item()
            scorer.done[batch_idx] = scorer.is_done(
                batch_idx, best_score, input_ids.size(1)
            )

        # Check if all done
        if all(scorer.done):
            break

        # Update for next iteration
        input_ids = torch.stack(next_input_ids)
        beam_scores = torch.tensor(next_beam_scores, device=device)

    # Finalize
    return scorer.finalize(input_ids, beam_scores, pad_token_id)
