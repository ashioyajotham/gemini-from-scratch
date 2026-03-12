#!/usr/bin/env python
"""
Simple terminal chat interface for a trained language model.

Usage:
    python demos/chat_interface.py --checkpoint checkpoints/pretrained/model.pt
    python demos/chat_interface.py --checkpoint checkpoints/pretrained/model.pt \\
                                   --temperature 0.8 --top_k 40

Controls:
    Type your prompt and press Enter to generate.
    Type 'quit' or Ctrl-C to exit.
    Type 'reset' to clear conversation history.
    Type 'config' to show current settings.
"""

import argparse
import sys
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.transformer import TransformerLM, TransformerConfig
from src.generation.sampling import sample_decode, greedy_decode
from src.utils.helpers import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Terminal chat with a language model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device):
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device)

    cfg_dict = ckpt.get("model_config", ckpt.get("config", {}))
    if isinstance(cfg_dict, dict):
        cfg = TransformerConfig(**cfg_dict)
    else:
        cfg = cfg_dict

    model = TransformerLM(cfg).to(device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.eval()

    # Build char tokenizer (fallback when no tokenizer file)
    all_chars = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'\"-:;()\n"
    chars = sorted(set(all_chars))
    c2i = {c: i % cfg.vocab_size for i, c in enumerate(chars)}
    i2c = {i: c for c, i in c2i.items()}
    encode = lambda s: [c2i.get(c, 0) for c in s]
    decode = lambda ids: "".join(i2c.get(i % len(i2c), "") for i in ids)

    return model, cfg, encode, decode


def print_banner(cfg, args):
    n_params = 0  # would need model reference; skip for simplicity
    print("\n" + "="*60)
    print("  Mini-Gemini Chat Interface")
    print("="*60)
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Device     : {args.device or 'auto'}")
    print(f"  Mode       : {'greedy' if args.greedy else f'sample (T={args.temperature}, k={args.top_k}, p={args.top_p})'}")
    print(f"  Max tokens : {args.max_new_tokens}")
    print("="*60)
    print("  Commands: 'reset', 'config', 'quit'")
    print("="*60 + "\n")


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else get_device()

    print("Loading model...", end=" ", flush=True)
    try:
        model, cfg, encode, decode = load_model(args.checkpoint, device)
        print("OK")
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nTo create a checkpoint, run:")
        print("  python scripts/download_data.py --dataset sample")
        print("  python scripts/train.py --config configs/small_model.yaml")
        sys.exit(1)

    print_banner(cfg, args)

    conversation_prefix = ""

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        if user_input.lower() == "reset":
            conversation_prefix = ""
            print("[Conversation reset]\n")
            continue

        if user_input.lower() == "config":
            print(f"\n  temperature={args.temperature}  top_k={args.top_k}  top_p={args.top_p}")
            print(f"  max_new_tokens={args.max_new_tokens}  greedy={args.greedy}\n")
            continue

        # Append user input to conversation
        prompt = conversation_prefix + user_input + "\n"
        prompt_ids = encode(prompt)

        # Clip to model's max context
        max_ctx = cfg.max_seq_len - args.max_new_tokens
        if len(prompt_ids) > max_ctx:
            prompt_ids = prompt_ids[-max_ctx:]

        prompt_tensor = torch.tensor([prompt_ids], device=device)

        print("Model: ", end="", flush=True)
        with torch.no_grad():
            if args.greedy:
                out = greedy_decode(model, prompt_tensor, max_new_tokens=args.max_new_tokens)
            else:
                out = sample_decode(
                    model, prompt_tensor,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                )

        generated_ids = out[0, len(prompt_ids):].tolist()
        response = decode(generated_ids).strip()
        print(response + "\n")

        # Update conversation context
        conversation_prefix = prompt + response + "\n"


if __name__ == "__main__":
    main()
