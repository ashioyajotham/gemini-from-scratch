#!/usr/bin/env python
"""
Evaluate a trained language model checkpoint.

Computes perplexity on a validation set and optionally runs generation samples.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/pretrained/model.pt \\
                               --data data/raw/tinystories_small.txt
    python scripts/evaluate.py --checkpoint checkpoints/pretrained/model.pt \\
                               --prompts "Once upon a time" "The princess"
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.transformer import TransformerLM, TransformerConfig
from src.generation.sampling import greedy_decode, sample_decode
from src.utils.helpers import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained language model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--data", type=str, default=None, help="Path to evaluation text file")
    parser.add_argument("--tokenizer", type=str, default=None, help="Path to tokenizer model")
    parser.add_argument("--prompts", nargs="+", default=None, help="Text prompts for generation")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Tokens to generate per prompt")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling p")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length for evaluation")
    parser.add_argument("--device", type=str, default=None, help="Device override (cuda/cpu/mps)")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    return parser.parse_args()


def load_checkpoint(path: str, device: torch.device):
    """Load model from checkpoint file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)

    # Support multiple checkpoint formats
    if "model_config" in ckpt:
        cfg_dict = ckpt["model_config"]
        cfg = TransformerConfig(**cfg_dict)
    elif "config" in ckpt:
        cfg = ckpt["config"]
        if isinstance(cfg, dict):
            cfg = TransformerConfig(**cfg)
    else:
        raise ValueError("Checkpoint does not contain model config. Cannot reconstruct model.")

    model = TransformerLM(cfg).to(device)

    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {n_params:,} parameters")
    print(f"  Config: d_model={cfg.d_model}, n_layers={cfg.n_layers}, n_heads={cfg.n_heads}")

    return model, cfg, ckpt


def build_char_tokenizer(text: str):
    """Build a simple character tokenizer from text."""
    chars = sorted(set(text))
    c2i = {c: i for i, c in enumerate(chars)}
    i2c = {i: c for i, c in enumerate(chars)}
    encode = lambda s: [c2i.get(c, 0) for c in s]
    decode = lambda ids: "".join(i2c.get(i, "?") for i in ids)
    return encode, decode, len(chars)


def compute_perplexity(model, token_ids, seq_len, batch_size, device):
    """Compute perplexity on a sequence of token IDs."""
    token_ids = torch.tensor(token_ids, dtype=torch.long)

    # Build non-overlapping sequences
    n = (len(token_ids) - 1) // seq_len
    if n == 0:
        print("  WARNING: Not enough data for even one sequence — skipping perplexity.")
        return float("nan")

    seqs = token_ids[: n * seq_len + 1]
    x = seqs[:-1].view(n, seq_len)
    y = seqs[1:].view(n, seq_len)

    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                yb.view(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += yb.numel()

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


def run_generation(model, encode, decode, prompts, cfg, args, device):
    """Generate text from prompts and print results."""
    print(f"\n{'─'*60}")
    print("Text Generation Samples")
    print(f"{'─'*60}")
    results = []
    for prompt in prompts:
        prompt_ids = torch.tensor([encode(prompt)], device=device)
        print(f"\nPrompt: \"{prompt}\"")

        # Greedy
        greedy_out = greedy_decode(model, prompt_ids, max_new_tokens=args.max_new_tokens)
        greedy_text = decode(greedy_out[0, len(encode(prompt)):].tolist())
        print(f"  Greedy  : {greedy_text}")

        # Sampling
        sample_out = sample_decode(
            model, prompt_ids, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        )
        sample_text = decode(sample_out[0, len(encode(prompt)):].tolist())
        print(f"  Sampled : {sample_text}")

        results.append({
            "prompt": prompt,
            "greedy": greedy_text,
            "sampled": sample_text,
        })
    return results


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else get_device()
    print(f"Device: {device}")

    # Load model
    model, cfg, ckpt = load_checkpoint(args.checkpoint, device)

    results = {
        "checkpoint": args.checkpoint,
        "model_params": sum(p.numel() for p in model.parameters()),
        "config": cfg.to_dict() if hasattr(cfg, "to_dict") else {},
        "step": ckpt.get("step", ckpt.get("global_step", "unknown")),
    }

    # Perplexity evaluation
    if args.data:
        data_path = Path(args.data)
        if not data_path.exists():
            print(f"WARNING: Data file not found: {data_path}")
        else:
            print(f"\nEvaluating on: {data_path}")
            text = data_path.read_text(encoding="utf-8")
            encode, decode, vocab_size = build_char_tokenizer(text)
            token_ids = encode(text)
            print(f"  Text length: {len(text):,} chars → {len(token_ids):,} tokens")

            ppl = compute_perplexity(
                model, token_ids, min(args.seq_len, cfg.max_seq_len),
                args.batch_size, device
            )
            print(f"\n  Perplexity : {ppl:.2f}")
            print(f"  (Random baseline: vocab_size = {vocab_size})")
            results["perplexity"] = ppl

    # Generation
    if args.prompts:
        try:
            if args.data:
                gen_results = run_generation(model, encode, decode, args.prompts, cfg, args, device)
            else:
                # Fallback: character tokenizer from a tiny vocabulary
                all_chars = "abcdefghijklmnopqrstuvwxyz .,!?'\"-:;()\n"
                chars = sorted(set(all_chars))
                c2i = {c: i % cfg.vocab_size for i, c in enumerate(chars)}
                i2c = {i: c for c, i in c2i.items()}
                encode = lambda s: [c2i.get(c, 0) for c in s.lower()]
                decode = lambda ids: "".join(i2c.get(i % len(i2c), "?") for i in ids)
                gen_results = run_generation(model, encode, decode, args.prompts, cfg, args, device)
            results["generation"] = gen_results
        except Exception as e:
            print(f"WARNING: Generation failed: {e}")

    # Save results
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to: {out_path}")

    # Summary
    print(f"\n{'='*50}")
    print("Evaluation Summary")
    print(f"{'='*50}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Parameters : {results['model_params']:,}")
    print(f"  Step       : {results['step']}")
    if "perplexity" in results:
        print(f"  Perplexity : {results['perplexity']:.2f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
