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

from src.models.transformer import Transformer as TransformerLM, TransformerConfig
from src.utils.device import get_device


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


def get_tokenizer(checkpoint_path, cfg):
    """Return (encode, token_fn) — SentencePiece if available, else char-level."""
    tokenizer_path = Path(checkpoint_path).parent / "tokenizer.model"
    if tokenizer_path.exists():
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.Load(str(tokenizer_path))
        encode = lambda s: sp.EncodeAsIds(s)
        tokens_to_labels = lambda ids, text: sp.IdToPiece(ids)
    else:
        all_chars = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'\"-:;()\n"
        chars = sorted(set(all_chars))
        c2i = {c: i % cfg.vocab_size for i, c in enumerate(chars)}
        encode = lambda s: [c2i.get(c, 0) for c in s]
        tokens_to_labels = None
    return encode, tokens_to_labels


def visualize_attention(model, cfg, text, layer_idx, args, device):
    encode, tokens_to_labels = get_tokenizer(args.checkpoint, cfg)
    tokens = encode(text)

    # Clip to model max length
    if len(tokens) > cfg.max_seq_len:
        tokens = tokens[:cfg.max_seq_len]

    # Build per-token labels for axis ticks
    if tokens_to_labels is not None:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.Load(str(Path(args.checkpoint).parent / "tokenizer.model"))
        token_labels = [sp.IdToPiece(t).replace('\u2581', '_') for t in tokens]
    else:
        token_labels = list(text[:len(tokens)])

    x = torch.tensor([tokens], device=device)

    # Run forward pass, then read attention weights via get_attention_weights()
    layers = getattr(model, 'layers', getattr(model, 'blocks', []))
    n_layers = len(layers)

    with torch.no_grad():
        _ = model(x)

    if layer_idx >= n_layers:
        print(f"Layer {layer_idx} out of range (model has {n_layers} layers). Using layer 0.")
        layer_idx = 0

    attn_module = getattr(layers[layer_idx], 'attention',
                  getattr(layers[layer_idx], 'attn', None))
    if attn_module is None or not hasattr(attn_module, 'get_attention_weights'):
        print("Attention module does not expose get_attention_weights().")
        return

    weights = attn_module.get_attention_weights()
    if weights is None:
        print("No attention weights available — forward pass may not have stored them.")
        return

    weights = weights[0].cpu()  # (n_heads, seq, seq)
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

    layer_name = f"Layer {layer_idx}"
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
