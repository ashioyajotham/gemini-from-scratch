#!/usr/bin/env python
"""
Benchmark model performance: throughput, latency, and memory.

Usage:
    python scripts/benchmark.py --config configs/small_model.yaml
    python scripts/benchmark.py --checkpoint checkpoints/pretrained/model.pt
    python scripts/benchmark.py --config configs/small_model.yaml --profile
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.transformer import TransformerLM, TransformerConfig
from src.utils.helpers import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark model performance")
    parser.add_argument("--config", type=str, default="configs/small_model.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 4, 16, 32])
    parser.add_argument("--seq_lengths", nargs="+", type=int, default=[64, 128, 256, 512])
    parser.add_argument("--n_runs", type=int, default=20, help="Warmup + timing runs")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup runs (excluded from timing)")
    parser.add_argument("--gen_tokens", type=int, default=50, help="Tokens to generate for latency test")
    parser.add_argument("--profile", action="store_true", help="Run PyTorch profiler")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def get_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / 1e6
    return 0.0


def reset_memory(device: torch.device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_model(args, device) -> TransformerLM:
    """Build model from config or checkpoint."""
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        cfg_dict = ckpt.get("model_config", ckpt.get("config", {}))
        if isinstance(cfg_dict, dict):
            cfg = TransformerConfig(**cfg_dict)
        else:
            cfg = cfg_dict
        model = TransformerLM(cfg)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        import yaml
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            print(f"Config not found: {cfg_path} — using defaults")
            cfg = TransformerConfig()
        else:
            with open(cfg_path) as f:
                raw = yaml.safe_load(f)
            model_cfg = raw.get("model", raw)
            cfg = TransformerConfig(**{k: v for k, v in model_cfg.items() if hasattr(TransformerConfig, k)})
        model = TransformerLM(cfg)

    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters | d_model={cfg.d_model} n_layers={cfg.n_layers}")
    return model


def benchmark_throughput(model, batch_size, seq_len, n_runs, warmup, device):
    """Measure tokens/second during forward pass."""
    if seq_len > model.config.max_seq_len if hasattr(model, "config") else 2048:
        return None

    x = torch.randint(0, 100, (batch_size, seq_len), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
    sync(device)

    reset_memory(device)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model(x)
    sync(device)
    elapsed = time.perf_counter() - start

    tokens = batch_size * seq_len * n_runs
    throughput = tokens / elapsed
    latency_ms = elapsed / n_runs * 1000
    peak_mem = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0

    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "throughput_tok_per_sec": int(throughput),
        "latency_ms": round(latency_ms, 2),
        "peak_memory_mb": round(peak_mem, 1),
    }


def benchmark_generation(model, prompt_len, gen_tokens, n_runs, warmup, device):
    """Measure autoregressive generation speed (tokens/sec)."""
    prompt = torch.randint(0, 100, (1, prompt_len), device=device)

    def generate_naive(n):
        ids = prompt.clone()
        with torch.no_grad():
            for _ in range(n):
                logits = model(ids[:, -512:])[:, -1, :]
                next_t = logits.argmax(-1, keepdim=True)
                ids = torch.cat([ids, next_t], dim=1)
        return ids

    # Warmup
    for _ in range(warmup):
        generate_naive(gen_tokens)
    sync(device)

    start = time.perf_counter()
    for _ in range(n_runs):
        generate_naive(gen_tokens)
    sync(device)
    elapsed = time.perf_counter() - start

    tok_per_sec = gen_tokens * n_runs / elapsed
    return {
        "prompt_len": prompt_len,
        "gen_tokens": gen_tokens,
        "tok_per_sec": round(tok_per_sec, 1),
        "ms_per_token": round(1000 / tok_per_sec, 2),
    }


def print_throughput_table(results):
    header = f"{'Batch':>6} {'SeqLen':>8} {'Tok/s':>12} {'Latency(ms)':>13} {'PeakMem(MB)':>13}"
    print(header)
    print("─" * len(header))
    for r in results:
        if r is None:
            continue
        print(
            f"{r['batch_size']:>6} {r['seq_len']:>8} "
            f"{r['throughput_tok_per_sec']:>12,} "
            f"{r['latency_ms']:>13.1f} "
            f"{r['peak_memory_mb']:>13.1f}"
        )


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else get_device()

    print(f"\n{'='*60}")
    print("  Model Benchmark")
    print(f"{'='*60}")
    print(f"  Device    : {device}")
    if device.type == "cuda":
        print(f"  GPU       : {torch.cuda.get_device_name(0)}")
    print(f"  n_runs    : {args.n_runs}  (warmup: {args.warmup})")
    print(f"{'='*60}\n")

    model = build_model(args, device)

    # ── Forward pass throughput ────────────────────────────────────────────
    print("\nForward Pass Throughput\n" + "─" * 60)
    throughput_results = []
    for bs in args.batch_sizes:
        for sl in args.seq_lengths:
            r = benchmark_throughput(model, bs, sl, args.n_runs - args.warmup, args.warmup, device)
            if r:
                throughput_results.append(r)

    print_throughput_table(throughput_results)

    # ── Generation latency ────────────────────────────────────────────────
    print("\n\nAutoregressive Generation Latency\n" + "─" * 60)
    gen_results = []
    for plen in [8, 32, 128]:
        r = benchmark_generation(model, plen, args.gen_tokens, max(1, args.n_runs // 4), args.warmup, device)
        gen_results.append(r)
        print(
            f"  Prompt={plen:>4} -> {args.gen_tokens} tokens: "
            f"{r['tok_per_sec']:>8.1f} tok/s  ({r['ms_per_token']:.1f} ms/tok)"
        )

    # ── Memory footprint ─────────────────────────────────────────────────
    print("\n\nMemory Footprint\n" + "─" * 60)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    print(f"  Parameters : {param_bytes / 1e6:.1f} MB (fp32)")
    print(f"  Parameters : {param_bytes / 2e6:.1f} MB (fp16)")
    print(f"  Buffers    : {buffer_bytes / 1e6:.2f} MB")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {n_params:,}")

    # ── Optional profiler ─────────────────────────────────────────────────
    if args.profile and device.type == "cuda":
        print("\n\nPyTorch Profiler (top ops by CUDA time)\n" + "─" * 60)
        x = torch.randint(0, 100, (4, 128), device=device)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_flops=True,
        ) as prof:
            with torch.no_grad():
                for _ in range(10):
                    model(x)

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    # ── Save results ──────────────────────────────────────────────────────
    all_results = {
        "device": str(device),
        "n_params": n_params,
        "throughput": throughput_results,
        "generation": gen_results,
        "memory_mb_fp32": round(param_bytes / 1e6, 1),
        "memory_mb_fp16": round(param_bytes / 2e6, 1),
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(all_results, indent=2))
        print(f"\nResults saved to: {out_path}")

    print(f"\n{'='*60}")
    print("Benchmark complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
