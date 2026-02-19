#!/usr/bin/env python
"""
Interactive chat interface for the language model.

Usage:
    python scripts/chat.py --checkpoint checkpoints/final_model.pt
"""

import argparse
import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import get_device, set_seed
from src.data import BPETokenizer
from src.models import Transformer
from src.generation import generate, SamplingConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Interactive chat with the model")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to tokenizer model",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=100,
        help="Maximum tokens per response",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling threshold",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


def print_banner():
    """Print welcome banner."""
    print("=" * 60)
    print("  Gemini-Q Interactive Chat")
    print("=" * 60)
    print("Commands:")
    print("  /quit or /exit - Exit the chat")
    print("  /clear         - Clear conversation history")
    print("  /temp <value>  - Set temperature")
    print("  /tokens <n>    - Set max tokens")
    print("=" * 60)
    print()


def main():
    """Main chat function."""
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = Transformer.from_pretrained(args.checkpoint, device=device)
    model.eval()

    # Load tokenizer
    tokenizer_path = args.tokenizer
    if tokenizer_path is None:
        checkpoint_dir = Path(args.checkpoint).parent
        tokenizer_path = checkpoint_dir / "tokenizer.model"

    if not Path(tokenizer_path).exists():
        print(f"Error: Tokenizer not found at {tokenizer_path}")
        sys.exit(1)

    tokenizer = BPETokenizer(tokenizer_path)

    # Generation config
    config = SamplingConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_tokens,
        eos_token_id=tokenizer.eos_id,
        pad_token_id=tokenizer.pad_id,
    )

    print_banner()

    # Chat loop
    history = ""

    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.lower().split()
                if cmd[0] in ["/quit", "/exit"]:
                    print("Goodbye!")
                    break
                elif cmd[0] == "/clear":
                    history = ""
                    print("Conversation cleared.")
                    continue
                elif cmd[0] == "/temp" and len(cmd) > 1:
                    try:
                        config.temperature = float(cmd[1])
                        print(f"Temperature set to {config.temperature}")
                    except ValueError:
                        print("Invalid temperature value")
                    continue
                elif cmd[0] == "/tokens" and len(cmd) > 1:
                    try:
                        config.max_new_tokens = int(cmd[1])
                        print(f"Max tokens set to {config.max_new_tokens}")
                    except ValueError:
                        print("Invalid token count")
                    continue
                else:
                    print("Unknown command")
                    continue

            # Build prompt with history
            prompt = history + user_input

            # Encode
            input_ids = torch.tensor(
                [tokenizer.encode(prompt)],
                device=device
            )

            # Generate
            with torch.no_grad():
                output_ids = generate(model, input_ids, config)

            # Decode response
            response_ids = output_ids[0].tolist()
            full_text = tokenizer.decode(response_ids)

            # Extract just the new part
            response = full_text[len(prompt):].strip()

            print(f"Bot: {response}")
            print()

            # Update history
            history = prompt + " " + response + " "

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue


if __name__ == "__main__":
    main()
