#!/usr/bin/env python
"""
Download and prepare training datasets.

Usage:
    python scripts/download_data.py --dataset tinystories --size small
    python scripts/download_data.py --dataset wikitext2
    python scripts/download_data.py --dataset sample   # built-in sample, no download
"""

import argparse
import json
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ─── Dataset registry ────────────────────────────────────────────────────────

DATASETS = {
    "tinystories": {
        "description": "TinyStories — short children's stories, ideal for quick training",
        "hf_name": "roneneldan/TinyStories",
        "splits": {"small": 10_000, "medium": 100_000, "full": None},
        "text_column": "text",
    },
    "wikitext2": {
        "description": "WikiText-2 — standard NLP benchmark",
        "hf_name": "Salesforce/wikitext",
        "hf_config": "wikitext-2-raw-v1",
        "splits": {"full": None},
        "text_column": "text",
    },
}

SAMPLE_STORIES = [
    "Once upon a time there was a small rabbit named Max who lived in a cozy burrow under an old oak tree. "
    "Every morning Max would hop out to explore the forest. One day he found a shiny red apple near the stream. "
    "He brought it home and shared it with his friends.",
    "Lily loved to paint. She painted sunsets, mountains, and tiny frogs sitting on lily pads. "
    "One rainy afternoon she painted a door on the wall of her room. When she opened it, she found a garden "
    "full of everything she had ever drawn.",
    "Tom was a young knight who had never fought a dragon. When the time came he discovered the dragon "
    "was just lonely. They became friends and the dragon used its fire to light the village lanterns every night.",
    "The little star was afraid of the dark, which was funny because she lived in the night sky. "
    "The moon told her that her light was the reason others were not afraid. From then on she shone as brightly "
    "as she could.",
    "Every morning the old lighthouse keeper climbed the spiral stairs to polish the great lamp. "
    "He had guided ships safely for forty years and knew every rock in the bay by name.",
] * 200  # repeat to make a usable dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Download and prepare training datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        default="sample",
        choices=list(DATASETS.keys()) + ["sample"],
        help="Dataset to download (default: sample — no download required)",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="small",
        choices=["small", "medium", "full"],
        help="Dataset size variant (for datasets that support it)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(project_root / "data" / "raw"),
        help="Directory to save downloaded data",
    )
    parser.add_argument(
        "--samples_dir",
        type=str,
        default=str(project_root / "data" / "samples"),
        help="Directory for sample datasets",
    )
    return parser.parse_args()


def write_text_file(path: Path, texts: list, max_examples: int = None):
    """Write a list of text strings to a plain text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if max_examples:
        texts = texts[:max_examples]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(t.strip() for t in texts if t.strip()))
    print(f"  Wrote {len(texts):,} examples -> {path}")
    size_mb = path.stat().st_size / 1e6
    print(f"  File size: {size_mb:.1f} MB")


def write_sample_dataset(samples_dir: Path):
    """Write the built-in sample dataset (no internet required)."""
    samples_dir.mkdir(parents=True, exist_ok=True)
    out = samples_dir / "sample_stories.txt"
    write_text_file(out, SAMPLE_STORIES)

    # Also write a tiny 100-example subset for unit tests
    tiny_out = samples_dir / "tiny_stories.txt"
    write_text_file(tiny_out, SAMPLE_STORIES[:100])

    # Metadata
    meta = {
        "dataset": "sample",
        "n_examples": len(SAMPLE_STORIES),
        "description": "Built-in sample stories for testing (no download required)",
    }
    (samples_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSample dataset ready in {samples_dir}/")


def download_huggingface(dataset_name: str, size: str, output_dir: Path):
    """Download a dataset from HuggingFace Datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed.")
        print("  Run: pip install datasets")
        sys.exit(1)

    info = DATASETS[dataset_name]
    hf_name = info["hf_name"]
    hf_config = info.get("hf_config")
    text_col = info["text_column"]
    max_examples = info["splits"].get(size)

    print(f"Downloading {dataset_name} ({size}) from HuggingFace...")
    print(f"  Source: {hf_name}")

    try:
        kwargs = dict(split="train", streaming=True, trust_remote_code=True)
        if hf_config:
            ds = load_dataset(hf_name, hf_config, **kwargs)
        else:
            ds = load_dataset(hf_name, **kwargs)

        texts = []
        for i, example in enumerate(ds):
            if max_examples and i >= max_examples:
                break
            text = example.get(text_col, "")
            if text and len(text.strip()) > 20:
                texts.append(text.strip())

        if not texts:
            print("WARNING: No usable text found. Check dataset structure.")
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{dataset_name}_{size}.txt"
        write_text_file(out_path, texts)

        meta = {
            "dataset": dataset_name,
            "size": size,
            "n_examples": len(texts),
            "hf_source": hf_name,
        }
        (output_dir / f"{dataset_name}_metadata.json").write_text(json.dumps(meta, indent=2))

    except Exception as e:
        print(f"ERROR downloading {dataset_name}: {e}")
        print("Falling back to sample dataset...")
        return False

    return True


def main():
    args = parse_args()
    output_dir  = Path(args.output_dir)
    samples_dir = Path(args.samples_dir)

    print(f"\n{'='*50}")
    print(f"  Dataset Downloader")
    print(f"{'='*50}")
    print(f"  Dataset : {args.dataset}")
    print(f"  Size    : {args.size}")
    print(f"  Output  : {output_dir}")
    print(f"{'='*50}\n")

    if args.dataset == "sample":
        print("Writing built-in sample dataset (no internet required)...")
        write_sample_dataset(samples_dir)
    else:
        info = DATASETS.get(args.dataset)
        if info is None:
            print(f"ERROR: Unknown dataset '{args.dataset}'")
            sys.exit(1)
        print(f"Dataset: {info['description']}")
        ok = download_huggingface(args.dataset, args.size, output_dir)
        if not ok:
            write_sample_dataset(samples_dir)

    # Always write sample dataset for tests
    if args.dataset != "sample":
        tiny_out = samples_dir / "tiny_stories.txt"
        if not tiny_out.exists():
            write_sample_dataset(samples_dir)

    print("\nDone. You can now run:")
    print("  python scripts/train.py --config configs/small_model.yaml")


if __name__ == "__main__":
    main()
