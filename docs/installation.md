# Installation Guide

Step-by-step setup for the Gemini-from-Scratch workshop on Linux, macOS, and Windows.

---

## Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.9 | 3.11 |
| RAM | 8 GB | 16 GB |
| Disk space | 5 GB | 10 GB |
| GPU (optional) | — | NVIDIA CUDA 11.8+ / Apple M1+ |

---

## Option A — Conda (recommended)

Conda handles Python version isolation and binary dependencies (PyTorch, CUDA) cleanly.

### 1. Install Miniconda

Download the installer for your platform from the [Miniconda page](https://docs.conda.io/en/latest/miniconda.html) and follow the on-screen instructions.

Verify:
```bash
conda --version
```

### 2. Clone the repository

```bash
git clone https://github.com/ashioyajotham/gemini-from-scratch.git
cd gemini-from-scratch
```

### 3. Create and activate the environment

```bash
conda env create -f environment.yml
conda activate gemini-workshop
```

### 4. Install the project package

```bash
pip install -e .
```

### 5. Verify

```bash
python scripts/verify_setup.py
```

All checks should show ✓. If anything fails, see [Troubleshooting](troubleshooting.md).

---

## Option B — pip + venv

If you prefer not to use Conda:

```bash
git clone https://github.com/ashioyajotham/gemini-from-scratch.git
cd gemini-from-scratch

python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# Windows (cmd)
venv\Scripts\activate.bat

pip install --upgrade pip
pip install -e ".[dev]"
```

> **Note:** With pip+venv you must install PyTorch manually (see below) to ensure you get the right CUDA variant.

---

## PyTorch Installation

### CPU only (works everywhere)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### NVIDIA GPU — CUDA 12.1

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### NVIDIA GPU — CUDA 11.8

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Apple Silicon (M1 / M2 / M3)

PyTorch ships with MPS support in the standard macOS wheel:

```bash
pip install torch torchvision
```

Verify MPS is available:
```python
import torch
print(torch.backends.mps.is_available())  # should print True
```

---

## Platform-specific notes

### Windows

- Use **Git Bash** or **WSL2** for all shell commands in this guide.
- Line-ending warnings (`LF → CRLF`) from git are harmless — the code works correctly.
- `sentencepiece` may need a no-binary install if the wheel fails:
  ```bash
  pip install sentencepiece --no-binary sentencepiece
  ```

### macOS

- Some PyTorch operations fall back to CPU on MPS. Set the fallback env var to avoid errors:
  ```bash
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  ```
  Add it to your `~/.zshrc` to make it permanent.

### Linux (headless / HPC)

- Jupyter needs a port-forwarding tunnel if you are on a remote server:
  ```bash
  # On the remote server
  jupyter notebook --no-browser --port=8888
  # On your local machine
  ssh -L 8888:localhost:8888 user@remote-host
  ```

---

## Jupyter Notebooks

```bash
# Start Jupyter from the project root (important for path resolution)
cd gemini-from-scratch
jupyter notebook
```

Open `notebooks/00_setup_and_verification.ipynb` to confirm everything works before starting Part 1.

If plots don't render, add this to the top of any notebook cell:
```python
%matplotlib inline
```

---

## Downloading Data

The downloader script fetches datasets from HuggingFace or falls back to a built-in sample corpus:

```bash
# Offline sample (no internet required, ~10 KB)
python scripts/download_data.py --dataset sample

# TinyStories (~500 MB compressed)
python scripts/download_data.py --dataset tinystories --size small

# WikiText-2 (~12 MB)
python scripts/download_data.py --dataset wikitext2
```

Data is saved to `data/raw/`.

---

## Running the Verification Script

```bash
python scripts/verify_setup.py
```

Expected output:
```
✓ Python 3.11.x
✓ PyTorch 2.x.x  (device: cuda:0  |  CUDA 12.1)
✓ src.models imports OK
✓ src.training imports OK
✓ src.data imports OK
✓ src.generation imports OK
✓ Forward pass OK  (output shape: torch.Size([1, 16, 256]))

All checks passed. You are ready for the workshop!
```

---

## Updating the Environment

If new dependencies are added to `environment.yml` or `pyproject.toml`:

```bash
# Conda
conda env update -f environment.yml --prune

# pip
pip install -e ".[dev]" --upgrade
```

---

## Uninstalling

```bash
# Remove conda environment
conda remove -n gemini-workshop --all

# Or remove venv
rm -rf venv/
```

---

## Getting Help

- [Troubleshooting Guide](troubleshooting.md) — common errors and fixes
- [GitHub Issues](https://github.com/ashioyajotham/gemini-from-scratch/issues) — report bugs or ask questions
