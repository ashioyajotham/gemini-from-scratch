#!/usr/bin/env python
"""
Text generation script.

Usage:
    python scripts/generate.py --checkpoint checkpoints/final_model.pt --prompt "Once upon a time"
    python scripts/generate.py --checkpoint checkpoints/final_model.pt --prompt "The" --temperature 0.8
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
from src.generation import generate, SamplingConfig, beam_search


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate text with a trained model")

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
        help="Path to tokenizer model (default: same directory as checkpoint)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The",
        help="Text prompt to continue",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (lower = more deterministic)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="Top-k sampling (0 to disable)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) sampling (1.0 to disable)",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty (1.0 to disable)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--beam_search",
        action="store_true",
        help="Use beam search instead of sampling",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="Number of beams for beam search",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )

    return parser.parse_args()


def main():
    """Main generation function."""
    args = parse_args()

    # Set random seed
    if args.seed is not None:
        set_seed(args.seed)

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = Transformer.from_pretrained(args.checkpoint, device=device)
    model.eval()

    # Load tokenizer
    tokenizer_path = args.tokenizer
    if tokenizer_path is None:
        # Look for tokenizer in same directory as checkpoint
        checkpoint_dir = Path(args.checkpoint).parent
        tokenizer_path = checkpoint_dir / "tokenizer.model"

    if not Path(tokenizer_path).exists():
        print(f"Error: Tokenizer not found at {tokenizer_path}")
        print("Please specify --tokenizer path")
        sys.exit(1)

    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = BPETokenizer(tokenizer_path)

    # Encode prompt
    prompt_ids = tokenizer.encode(args.prompt)
    input_ids = torch.tensor([prompt_ids], device=device)

    print(f"\nPrompt: {args.prompt}")
    print(f"Prompt tokens: {len(prompt_ids)}")
    print("-" * 50)

    # Generate
    for sample_idx in range(args.num_samples):
        if args.num_samples > 1:
            print(f"\n--- Sample {sample_idx + 1} ---")

        if args.beam_search:
            # Beam search
            output_ids, scores = beam_search(
                model,
                input_ids,
                num_beams=args.num_beams,
                max_new_tokens=args.max_tokens,
                eos_token_id=tokenizer.eos_id,
                pad_token_id=tokenizer.pad_id,
            )
            generated_ids = output_ids[0].tolist()
        else:
            # Sampling
            config = SamplingConfig(
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                max_new_tokens=args.max_tokens,
                eos_token_id=tokenizer.eos_id,
                pad_token_id=tokenizer.pad_id,
            )

            output_ids = generate(model, input_ids, config)
            generated_ids = output_ids[0].tolist()

        # Decode
        generated_text = tokenizer.decode(generated_ids)
        print(generated_text)

    print("-" * 50)
    print(f"Generated {len(generated_ids) - len(prompt_ids)} new tokens")


if __name__ == "__main__":
    main()
