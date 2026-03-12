# Workshop Guide

**Building a Gemini-Level Model from Scratch**

A hands-on, five-part workshop for building transformer-based language models using PyTorch.

---

## Overview

This workshop takes you from "what is attention?" to a fully working language model — implementing every component from scratch. By the end you will have:

- Written every layer of a modern transformer (attention, FFN, normalisation, positional encoding)
- Trained a small language model that generates coherent text
- Understood the key innovations that power Gemini, LLaMA, and GPT-4

**Audience:** ML practitioners, software engineers with Python experience, or advanced students. Familiarity with PyTorch basics is helpful but not required.

**Duration:** Full workshop: ~6 hours. Can be split across two sessions.

---

## Prerequisites

| Requirement | Minimum |
|-------------|---------|
| Python | 3.10+ |
| PyTorch | 2.0+ |
| RAM | 8 GB |
| GPU | Optional (CPU is fine for all exercises) |
| Background | Basic linear algebra, some Python/PyTorch |

---

## Setup (10 minutes)

```bash
# 1. Clone the repository
git clone https://github.com/ashioyajotham/gemini-from-scratch.git
cd gemini-from-scratch

# 2. Create environment
conda env create -f environment.yml
conda activate gemini-workshop

# 3. Install package in editable mode
pip install -e .

# 4. Verify everything works
python scripts/verify_setup.py

# 5. Download sample data (no internet needed — uses built-in corpus)
python scripts/download_data.py --dataset sample

# 6. Open the setup notebook
jupyter notebook notebooks/00_setup_and_verification.ipynb
```

If `00_setup_and_verification.ipynb` runs all cells without error, you're ready.

---

## Curriculum

### Part 1 — Why Transformers? (45 min)

**Notebook:** `part1_evolution/01_rnn_limitations`

We start by *breaking* RNNs — deliberately demonstrating the problems that motivate everything that follows.

Key topics:
- Recurrent neural networks and the sequential bottleneck
- The vanishing gradient problem (measured, not just described)
- Long-range copy task: RNN vs attention
- Wall-clock speed comparison: sequential vs parallel

**Key insight:** Attention gives every token a direct O(1) path to every other token.

---

### Part 2 — Transformer Fundamentals (2 hours)

Five notebooks, each building on the last:

| Notebook | What you build | Time |
|----------|---------------|------|
| `02_attention_mechanism` | Scaled dot-product attention from scratch | 25 min |
| `03_multihead_attention` | Multi-head attention + GQA | 25 min |
| `04_positional_encoding` | Sinusoidal, learned, RoPE | 20 min |
| `05_feedforward_network` | FFN + SwiGLU | 20 min |
| `06_transformer_block` | Complete block + full model | 30 min |

---

### Part 3 — Modern Innovations (1.5 hours)

| Notebook | Topic | Key paper |
|----------|-------|-----------|
| `07_efficient_attention` | Sliding window, FlashAttention | Dao et al. 2022 |
| `08_mixture_of_experts` | Router, load balancing | Fedus et al. 2022 |
| `09_multimodal_fusion` | ViT patches, cross-attention | Dosovitskiy et al. 2020 |

---

### Part 4 — Training (1 hour)

| Notebook | Topic |
|----------|-------|
| `10_tokenization` | BPE from scratch + SentencePiece |
| `11_training_loop` | LM loss, warmup+cosine LR, gradient clipping |
| `12_text_generation` | Greedy, temperature, top-k, nucleus sampling |

---

### Part 5 — Integration (45 min)

| Notebook | Topic |
|----------|-------|
| `13_mini_gemini_project` | Full end-to-end pipeline (capstone) |
| `14_advanced_extensions` | KV-cache, MoE vs dense, multimodal |

---

## Running the Scripts

### Download data

```bash
# Built-in sample (no internet required)
python scripts/download_data.py --dataset sample

# TinyStories (requires HuggingFace datasets)
python scripts/download_data.py --dataset tinystories --size small

# WikiText-2
python scripts/download_data.py --dataset wikitext2
```

### Train a model

```bash
# Small model on CPU (~10 min)
python scripts/train.py --config configs/small_model.yaml \
                        --data data/samples/sample_stories.txt

# Medium model on GPU
python scripts/train.py --config configs/medium_model.yaml \
                        --data data/raw/tinystories_small.txt
```

### Evaluate

```bash
python scripts/evaluate.py --checkpoint checkpoints/pretrained/model.pt \
                            --data data/samples/sample_stories.txt \
                            --prompts "Once upon a time" "The princess"
```

### Generate text interactively

```bash
python scripts/chat.py --checkpoint checkpoints/pretrained/model.pt
```

### Benchmark performance

```bash
python scripts/benchmark.py --config configs/small_model.yaml
```

---

## Model Configurations

| Config | d_model | Layers | Heads | Params | Hardware |
|--------|---------|--------|-------|--------|----------|
| `small_model.yaml` | 256 | 4 | 4 | ~8M | CPU |
| `medium_model.yaml` | 512 | 6 | 8 | ~50M | Single GPU |
| `large_model.yaml` | 1024 | 12 | 16 | ~300M | Multi-GPU |

---

## Facilitator Notes

### Timing breakdown (full 6-hour session)

| Section | Duration | Break |
|---------|----------|-------|
| Setup + Part 0 | 20 min | — |
| Part 1: RNN Limitations | 45 min | 10 min |
| Part 2: Fundamentals | 2h | 15 min |
| Part 3: Innovations | 1.5h | 15 min |
| Part 4: Training | 1h | — |
| Part 5: Integration | 45 min | — |

### Common issues

See `troubleshooting.md` for solutions to common setup problems.

### Tips for participants

1. Read the docstrings — every function has a detailed explanation
2. Run starter cells top to bottom before filling in TODOs
3. The solution notebooks are there if you get stuck — use them
4. It's fine to skip exercises and use the library implementation to keep pace
5. The Mini-Gemini capstone (notebook 13) is the most important part — prioritise it

---

## Folder Structure Reference

```
notebooks/          ← Workshop notebooks (work here)
src/                ← Reusable Python library
scripts/            ← CLI tools (train, generate, evaluate)
configs/            ← YAML model configurations
data/               ← Datasets (created by download_data.py)
checkpoints/        ← Saved model weights
docs/               ← This guide and references
```
