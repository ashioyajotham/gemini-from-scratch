# Additional Resources

Courses, tools, and communities for going deeper after the workshop.

---

## Courses & Tutorials

| Resource | Description | Link |
|----------|-------------|------|
| **Andrej Karpathy — "Let's build GPT"** | Build GPT from scratch in 2 hours | YouTube |
| **Andrej Karpathy — makemore** | Character-level language model series | YouTube / GitHub |
| **fast.ai Practical Deep Learning** | Excellent practical introduction | fast.ai |
| **CS224N: Natural Language Processing with DL** | Stanford NLP course (free lectures) | Stanford |
| **Hugging Face NLP Course** | HuggingFace ecosystem from scratch | huggingface.co/learn |
| **Spinning Up in Deep RL** | Foundation for RLHF understanding | OpenAI / GitHub |

---

## Blogs & Articles

| Article | What it covers |
|---------|---------------|
| **The Illustrated Transformer** (Alammar) | Best visual explanation of the Transformer |
| **The Illustrated GPT-2** (Alammar) | Decoder-only architecture walkthrough |
| **The Annotated Transformer** (Harvard NLP) | Transformer paper with line-by-line code |
| **Lilian Weng's Blog** | Deep dives on attention, RL, diffusion |
| **Sebastian Ruder's NLP Newsletter** | State of NLP research |
| **Chip Huyen's Blog** | LLM systems, inference optimisation |

---

## Tools

### Training Frameworks
| Tool | Purpose |
|------|---------|
| **PyTorch Lightning** | High-level training loop abstraction |
| **HuggingFace Transformers** | Pre-trained model hub + fine-tuning |
| **Axolotl** | Fine-tuning framework for LLMs |
| **LitGPT** | Clean, hackable GPT implementation |

### Experiment Tracking
| Tool | Purpose |
|------|---------|
| **Weights & Biases (wandb)** | Experiment tracking and visualisation |
| **TensorBoard** | Built-in PyTorch logging |
| **MLflow** | Open-source experiment tracking |

### Data
| Tool | Purpose |
|------|---------|
| **HuggingFace Datasets** | Thousands of NLP datasets |
| **RedPajama** | Open reproduction of LLaMA training data |
| **Dolma** | Open dataset for pretraining |
| **The Pile** | 800GB diverse text corpus |

### Inference & Deployment
| Tool | Purpose |
|------|---------|
| **vLLM** | Fast LLM inference with PagedAttention |
| **llama.cpp** | CPU inference for LLaMA models |
| **GGUF / GGML** | Quantised model format |
| **TensorRT-LLM** | NVIDIA optimised inference |

---

## Pre-trained Models to Explore

All of these share the decoder-only transformer architecture you built:

| Model | Params | Open weights | Notes |
|-------|--------|-------------|-------|
| **GPT-2** | 124M–1.5B | Yes | The classic starting point |
| **LLaMA 3** | 8B–70B | Yes | Strong open baseline |
| **Mistral 7B** | 7B | Yes | Efficient, sliding window attn |
| **Mixtral 8x7B** | 47B (13B active) | Yes | MoE model from this workshop |
| **Gemma 2** | 2B–27B | Yes | Google's open model |
| **Phi-3** | 3.8B–14B | Yes | Small but capable |
| **Qwen2** | 0.5B–72B | Yes | Strong multilingual model |

---

## Next Steps After This Workshop

### 1. Fine-tune a pre-trained model
Take a GPT-2 or LLaMA checkpoint and fine-tune on a domain-specific dataset:
```bash
# HuggingFace makes this easy
pip install transformers datasets peft
```

### 2. Implement RLHF / DPO
Teach your model to follow instructions using human preference data:
- **TRL** library (HuggingFace) for PPO/DPO
- **InstructLab** for synthetic data generation

### 3. Quantisation
Make your model 4x smaller with almost no quality loss:
- **bitsandbytes** — 8-bit and 4-bit quantisation
- **GPTQ** — post-training quantisation
- **AWQ** — activation-aware weight quantisation

### 4. Efficient training
Scale to larger models with limited hardware:
- **DeepSpeed ZeRO** — shard model across GPUs
- **FSDP** — PyTorch native model sharding
- **LoRA / QLoRA** — parameter-efficient fine-tuning

### 5. Build something
Ideas for projects that use what you learned:
- A domain-specific chatbot (code, medical, legal)
- An image captioning system using the multimodal extension
- A story generation app with Streamlit/Gradio
- A coding assistant fine-tuned on your company's codebase

---

## Community

| Community | Where to find it |
|-----------|-----------------|
| HuggingFace Forums | discuss.huggingface.co |
| EleutherAI Discord | discord.gg/eleutherai |
| r/LocalLLaMA | reddit.com/r/LocalLLaMA |
| ML Collective | mlcollective.org |
| Papers With Code | paperswithcode.com |
