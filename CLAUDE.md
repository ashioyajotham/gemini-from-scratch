# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational Python workshop for building transformer-based language models from scratch using PyTorch. The project teaches modern LLM architectures through hands-on Jupyter notebooks and reusable Python modules.

**Status:** Project is in planning phase. See `project_structure.md` for full implementation plan.

## Commands

```bash
# Environment setup
conda env create -f environment.yml
conda activate gemini-workshop
pip install -e .

# Data preparation
python scripts/download_data.py --dataset tinystories --size small

# Training
python scripts/train.py --config configs/small_model.yaml

# Generation
python scripts/generate.py --checkpoint checkpoints/pretrained/model.pt --prompt "Once upon a time"

# Interactive chat
python scripts/chat.py --checkpoint checkpoints/pretrained/model.pt

# Testing
pytest tests/

# Code quality
black src/ tests/ scripts/
flake8 src/ tests/ scripts/
mypy src/
```

## Architecture

```
src/
├── models/          # Transformer components (attention, embeddings, FFN, blocks)
├── training/        # Training loop, optimizers, losses, callbacks
├── data/            # Tokenizer, dataset, dataloader, preprocessing
├── generation/      # Sampling strategies, beam search, KV-cache
└── utils/           # Visualization, metrics, checkpointing
```

**Key data flow:** Raw text → Tokenizer → Dataset/DataLoader → Embeddings → Transformer Blocks → Output Projection → (Training: Loss/Backprop | Inference: Sampling) → Generated Text

**Entry points:**
- `scripts/train.py` - main training script
- `scripts/generate.py` - text generation
- `scripts/chat.py` - interactive chat
- `notebooks/00_setup_and_verification.ipynb` - workshop entry point

## Configuration

YAML-based configs in `configs/`:
- `small_model.yaml` - demo model (runs on CPU)
- `medium_model.yaml` - workshop model (single GPU)
- `large_model.yaml` - reference implementation

## Notebook Structure

Five-part progressive curriculum:
1. `part1_evolution/` - RNN limitations and why attention matters
2. `part2_fundamentals/` - Attention, embeddings, transformer blocks
3. `part3_innovations/` - Efficient attention, MoE, multimodal
4. `part4_training/` - Tokenization, training loop, text generation
5. `part5_integration/` - Mini-gemini project combining all components

Each topic has `*_starter.ipynb` (exercises) and `*_solution.ipynb` (completed code).

## Key Dependencies

- PyTorch >=2.0.0
- SentencePiece / Hugging Face Tokenizers
- Hugging Face Datasets
- Weights & Biases / TensorBoard for experiment tracking
