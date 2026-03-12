# Troubleshooting Guide

Solutions to common problems during setup and the workshop.

---

## Installation Issues

### `conda env create` fails

```
CondaError: prefix already exists
```
**Fix:** Remove the existing environment first:
```bash
conda remove -n gemini-workshop --all
conda env create -f environment.yml
```

### `pip install -e .` fails with build error

**Fix:** Make sure you have the right Python version and try:
```bash
pip install --upgrade pip setuptools wheel
pip install -e .
```

### `import torch` → `ModuleNotFoundError`

The conda environment may not be activated:
```bash
conda activate gemini-workshop
python -c "import torch; print(torch.__version__)"
```

---

## CUDA / GPU Issues

### `torch.cuda.is_available()` returns `False`

1. Check your CUDA version: `nvcc --version`
2. Check PyTorch CUDA support: `python -c "import torch; print(torch.version.cuda)"`
3. If mismatched, reinstall PyTorch for your CUDA version:
   ```bash
   # For CUDA 12.1
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

### GPU out of memory (`CUDA out of memory`)

Reduce batch size or sequence length:
```bash
# In your config YAML
training:
  batch_size: 8      # down from 32
data:
  seq_length: 256    # down from 512
```

Or free GPU memory between notebook cells:
```python
import torch, gc
gc.collect()
torch.cuda.empty_cache()
```

### Apple MPS (Mac M1/M2) issues

Some operations are not yet supported on MPS. Fall back to CPU:
```python
device = torch.device("cpu")
```
Or set environment variable:
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 jupyter notebook
```

---

## Notebook Issues

### Kernel dies immediately on startup

**Cause:** Usually insufficient memory.
- Close other applications
- Reduce model size (use `tiny` config in notebooks)
- Restart with a fresh kernel: `Kernel → Restart & Clear Output`

### `ModuleNotFoundError: No module named 'src'`

The project root is not on Python's path. Add this to the top of the notebook:
```python
import sys, os
sys.path.insert(0, os.path.abspath('../..'))  # adjust depth if needed
```

Or run from the project root:
```bash
cd /path/to/gemini-from-scratch
jupyter notebook
```

### Notebook runs but plots don't appear

Enable inline plotting:
```python
%matplotlib inline
import matplotlib
matplotlib.use('Agg')   # for environments without a display
```

---

## Data Issues

### `FileNotFoundError: data/raw/...`

Run the data downloader first:
```bash
python scripts/download_data.py --dataset sample
```

### HuggingFace download fails / slow

Use the offline sample dataset:
```bash
python scripts/download_data.py --dataset sample
```

### `sentencepiece` not found

```bash
pip install sentencepiece
```
If still failing on Windows, try:
```bash
pip install sentencepiece --no-binary sentencepiece
```

---

## Training Issues

### Loss is `nan` or `inf` from the start

**Cause:** Learning rate too high, bad initialisation, or overflow.
- Reduce learning rate: `learning_rate: 1e-4` (down from `3e-4`)
- Enable gradient clipping: `gradient_clip: 1.0`
- Check your data for empty strings or very long sequences

### Loss decreases then suddenly spikes

Gradient spike — ensure gradient clipping is enabled:
```yaml
training:
  gradient_clip: 1.0
```

### Training is very slow on CPU

Expected for larger models. Use the smallest config:
```yaml
model:
  d_model: 64
  n_layers: 2
  n_heads: 4
  d_ff: 256
```
Or reduce `max_steps` for the exercise:
```bash
python scripts/train.py --config configs/small_model.yaml --max_steps 500
```

### `RuntimeError: Expected all tensors to be on the same device`

Ensure all tensors are moved to the same device:
```python
x = x.to(device)
model = model.to(device)
```

---

## Generation Issues

### Model generates only repeated tokens

**Cause:** Greedy decoding collapses to the most probable token.
- Add temperature: `temperature=0.8`
- Use top-k: `top_k=40`
- Use nucleus sampling: `top_p=0.9`

### `IndexError` during generation

The generated token ID exceeds the model's vocabulary:
```python
next_token = next_token.clamp(0, cfg.vocab_size - 1)
```

---

## Import Errors in src/

### `ImportError: cannot import name 'X' from 'src.models'`

The module may not be exported. Check `src/models/__init__.py` and add the missing import, or import directly:
```python
from src.models.attention import MultiHeadAttention  # direct import
```

---

## Getting Help

1. Check this guide first
2. Search existing GitHub Issues
3. Open a new issue at https://github.com/ashioyajotham/gemini-from-scratch/issues

When opening an issue, include:
- Your OS and Python/PyTorch version
- The full error traceback
- The notebook name and cell number
