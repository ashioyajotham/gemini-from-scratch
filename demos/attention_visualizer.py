#!/usr/bin/env python
"""
Interactive attention weight visualiser for a trained model.

Shows which tokens each head attends to for a given input sequence.

Usage:
    python demos/attention_visualizer.py --checkpoint checkpoints/pretrained/model.pt \\
                                         --text "once upon a time"
    python demos/attention_visualizer.py --checkpoint checkpoints/pretrained/model.pt \\
                                         --layer 0 --output outputs/visualizations/attn.png
"""

import argparse
import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.transformer import TransformerLM, TransformerConfig
from src.utils.helpers import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Visualise attention weights")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text", type=str, default="once upon a time there was a small rabbit")
    parser.add_argument("--layer", type=int, default=0, help="Which transformer layer to visualise (0-indexed)")
    parser.add_argument("--output", type=str, default=None, help="Save figure to file")
    parser.add_argument("--figsize", type=float, default=4.0, help="Size per head subplot (inches)")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_model(checkpoint_path, device):
    path = Path(checkpoint_path)
    ckpt = torch.load(path, map_location=device)
    cfg_dict = ckpt.get("model_config", ckpt.get("config", {}))
    cfg = TransformerConfig(**(cfg_dict if isinstance(cfg_dict, dict) else {}))
    model = TransformerLM(cfg).to(device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.eval()
    return model, cfg


def get_char_tokenizer(cfg):
    all_chars = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'\"-:;()\n"
    chars = sorted(set(all_chars))
    c2i = {c: i % cfg.vocab_size for i, c in enumerate(chars)}
    encode = lambda s: [c2i.get(c, 0) for c in s]
    return encode


def visualize_attention(model, cfg, text, layer_idx, args, device):
    encode = get_char_tokenizer(cfg)
    tokens = encode(text)
    token_labels = list(text)

    # Clip to model max length
    if len(tokens) > cfg.max_seq_len:
        tokens = tokens[:cfg.max_seq_len]
        token_labels = token_labels[:cfg.max_seq_len]

    x = torch.tensor([tokens], device=device)

    # Hook to capture attention weights
    captured = {}

    def make_hook(name):
        def hook(module, input, output):
            if hasattr(module, '_attention_weights') and module._attention_weights is not None:
                captured[name] = module._attention_weights.cpu()
        return hook

    handles = []
    for i, block in enumerate(model.blocks):
        attn_module = getattr(block, 'attn', getattr(block, 'self_attn', None))
        if attn_module is not None:
            h = attn_module.register_forward_hook(make_hook(f'block_{i}'))
            handles.append(h)

    with torch.no_grad():
        _ = model(x)

    for h in handles:
        h.remove()

    key = f'block_{layer_idx}'
    if key not in captured:
        available = list(captured.keys())
        print(f"No attention weights captured for layer {layer_idx}.")
        print(f"Available layers: {available}")
        if not available:
            print("The model's attention modules may not expose _attention_weights.")
            return
        key = available[0]
        print(f"Using {key} instead.")

    weights = captured[key][0]  # (n_heads, seq, seq)
    n_heads = weights.shape[0]
    seq_len = len(token_labels)
    weights = weights[:, :seq_len, :seq_len]

    n_cols = min(n_heads, 4)
    n_rows = (n_heads + n_cols - 1) // n_cols
    fig_w = args.figsize * n_cols
    fig_h = args.figsize * n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    if n_heads == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]

    for h in range(n_heads):
        row, col = h // n_cols, h % n_cols
        ax = axes[row][col]
        w = weights[h].numpy()

        im = ax.imshow(w, cmap='Blues', vmin=0, vmax=w.max() or 1)
        ax.set_xticks(range(seq_len))
        ax.set_xticklabels(token_labels, fontfamily='monospace', fontsize=7, rotation=90)
        ax.set_yticks(range(seq_len))
        ax.set_yticklabels(token_labels, fontfamily='monospace', fontsize=7)
        ax.set_title(f'Head {h+1}', fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused subplots
    for h in range(n_heads, n_rows * n_cols):
        axes[h // n_cols][h % n_cols].set_visible(False)

    layer_name = key.replace('_', ' ').title()
    fig.suptitle(
        f'Attention Weights — {layer_name}\n"{text[:50]}{"..." if len(text) > 50 else ""}"',
        fontsize=11, y=1.01
    )
    plt.tight_layout()

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        print(f"Saved to: {out_path}")
    else:
        plt.show()


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else get_device()

    try:
        model, cfg = load_model(args.checkpoint, device)
    except FileNotFoundError:
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        print("\nRun training first:")
        print("  python scripts/download_data.py --dataset sample")
        print("  python scripts/train.py --config configs/small_model.yaml")
        sys.exit(1)

    print(f"Visualising attention for: \"{args.text}\"")
    print(f"Layer: {args.layer}  |  Device: {device}")
    visualize_attention(model, cfg, args.text, args.layer, args, device)


if __name__ == "__main__":
    main()
