# Paper References

Key papers behind every concept in this workshop, grouped by topic.

---

## Foundational Transformers

| Paper | Authors | Year | Link |
|-------|---------|------|------|
| **Attention Is All You Need** | Vaswani et al. | 2017 | https://arxiv.org/abs/1706.03762 |
| **BERT: Pre-training of Deep Bidirectional Transformers** | Devlin et al. | 2018 | https://arxiv.org/abs/1810.04805 |
| **Language Models are Unsupervised Multitask Learners (GPT-2)** | Radford et al. | 2019 | OpenAI blog |
| **GPT-3: Language Models are Few-Shot Learners** | Brown et al. | 2020 | https://arxiv.org/abs/2005.14165 |

---

## Architectural Innovations

### Normalisation
| Paper | Key contribution |
|-------|-----------------|
| **Layer Normalisation** (Ba et al., 2016) | LayerNorm — https://arxiv.org/abs/1607.06450 |
| **Root Mean Square Layer Normalisation** (Zhang & Sennrich, 2019) | RMSNorm — https://arxiv.org/abs/1910.07467 |
| **On Layer Normalization in the Transformer Architecture** (Xiong et al., 2020) | Pre-LN stability — https://arxiv.org/abs/2002.04745 |

### Activation Functions
| Paper | Key contribution |
|-------|-----------------|
| **Gaussian Error Linear Units (GELU)** (Hendrycks & Gimpel, 2016) | GELU activation — https://arxiv.org/abs/1606.08415 |
| **GLU Variants Improve Transformer** (Noam, 2020) | SwiGLU/GeGLU — https://arxiv.org/abs/2002.05202 |

### Positional Encoding
| Paper | Key contribution |
|-------|-----------------|
| **Attention Is All You Need** (Vaswani et al., 2017) | Sinusoidal PE |
| **RoFormer: Enhanced Transformer with Rotary Position Embedding** (Su et al., 2021) | RoPE — https://arxiv.org/abs/2104.09864 |
| **ALiBi: Train Short, Test Long** (Press et al., 2021) | Linear biases — https://arxiv.org/abs/2108.12409 |

---

## Efficient Attention

| Paper | Key contribution |
|-------|-----------------|
| **Longformer** (Beltagy et al., 2020) | Sliding window attention — https://arxiv.org/abs/2004.05150 |
| **BigBird** (Zaheer et al., 2020) | Sparse attention — https://arxiv.org/abs/2007.14062 |
| **FlashAttention** (Dao et al., 2022) | IO-aware attention — https://arxiv.org/abs/2205.14135 |
| **FlashAttention-2** (Dao, 2023) | Improved parallelism — https://arxiv.org/abs/2307.08691 |
| **GQA: Training Generalized Multi-Query Transformer Models** (Ainslie et al., 2023) | Grouped-query attention — https://arxiv.org/abs/2305.13245 |
| **Multi-Query Attention** (Shazeer, 2019) | Single KV head — https://arxiv.org/abs/1911.02150 |

---

## Scaling & Mixture of Experts

| Paper | Key contribution |
|-------|-----------------|
| **Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer** (Shazeer et al., 2017) | Original MoE — https://arxiv.org/abs/1701.06538 |
| **Switch Transformers** (Fedus et al., 2022) | Simplified MoE routing — https://arxiv.org/abs/2101.03961 |
| **Mixtral of Experts** (Mistral AI, 2023) | Open-weights MoE LLM — https://arxiv.org/abs/2401.04088 |
| **Scaling Laws for Neural Language Models** (Kaplan et al., 2020) | Chinchilla predecessors — https://arxiv.org/abs/2001.08361 |
| **Training Compute-Optimal Large Language Models (Chinchilla)** (Hoffmann et al., 2022) | Optimal token-to-param ratio — https://arxiv.org/abs/2203.15556 |

---

## Multimodal Models

| Paper | Key contribution |
|-------|-----------------|
| **An Image is Worth 16x16 Words (ViT)** (Dosovitskiy et al., 2020) | Vision Transformer — https://arxiv.org/abs/2010.11929 |
| **CLIP** (Radford et al., 2021) | Contrastive image-text pretraining — https://arxiv.org/abs/2103.00020 |
| **Flamingo** (Alayrac et al., 2022) | Vision-language few-shot — https://arxiv.org/abs/2204.14198 |
| **Gemini: A Family of Highly Capable Multimodal Models** (Google, 2023) | Gemini — https://arxiv.org/abs/2312.11805 |

---

## Training & Optimisation

| Paper | Key contribution |
|-------|-----------------|
| **Adam: A Method for Stochastic Optimization** (Kingma & Ba, 2014) | Adam optimizer — https://arxiv.org/abs/1412.6980 |
| **Decoupled Weight Decay Regularization (AdamW)** (Loshchilov & Hutter, 2017) | AdamW — https://arxiv.org/abs/1711.05101 |
| **Dropout: A Simple Way to Prevent Neural Networks from Overfitting** (Srivastava et al., 2014) | Dropout regularisation |
| **Mixed Precision Training** (Micikevicius et al., 2017) | fp16 training — https://arxiv.org/abs/1710.03740 |

---

## Modern LLMs (Key Reference Models)

| Model | Paper | Year |
|-------|-------|------|
| **LLaMA** | Touvron et al., 2023 | https://arxiv.org/abs/2302.13971 |
| **LLaMA 2** | Touvron et al., 2023 | https://arxiv.org/abs/2307.09288 |
| **Mistral 7B** | Jiang et al., 2023 | https://arxiv.org/abs/2310.06825 |
| **Gemma** | Google DeepMind, 2024 | https://arxiv.org/abs/2403.08295 |
| **Gemma 2** | Google DeepMind, 2024 | https://arxiv.org/abs/2408.00118 |
| **Phi-2** | Javaheripi et al., 2023 | Microsoft Research blog |

---

## Text Generation

| Paper | Key contribution |
|-------|-----------------|
| **The Curious Case of Neural Text Degeneration** (Holtzman et al., 2019) | Nucleus (top-p) sampling — https://arxiv.org/abs/1904.09751 |
| **Hierarchical Neural Story Generation** (Fan et al., 2018) | Top-k sampling — https://arxiv.org/abs/1805.04833 |
| **Fast Transformer Decoding: One Write-Head is All You Need** (Shazeer, 2019) | Multi-query attention for inference — https://arxiv.org/abs/1911.02150 |
| **Speculative Decoding** (Leviathan et al., 2022) | Draft-model acceleration — https://arxiv.org/abs/2211.17192 |

---

## Recommended Reading Order

1. **Vaswani et al. (2017)** — the original Transformer (read first)
2. **Su et al. (2021)** — RoPE (positional encoding used in most modern models)
3. **Dao et al. (2022)** — FlashAttention (key to efficient training)
4. **Fedus et al. (2022)** — Switch Transformers (MoE)
5. **Touvron et al. (2023)** — LLaMA (the open-weights baseline everything is compared to)
6. **Google (2023)** — Gemini (the model this workshop is named after)
