# TinyLM

> Building a language model from scratch — not to ship (maybe), but to *understand*.

TinyLM is a from-scratch implementation of LLM training and inference, built progressively in phases of increasing complexity. The goal isn't a production system — it's to build deep, mechanistic understanding of every component: tokenization, attention, backpropagation, and hardware-level optimization.

Each phase is implemented naively first, then profiled, then optimized. Nothing is a black box.

---

## Phase 0 — Tokenization

**Goals:** Understand and implement popular tokenization algorithms like BPE and SentencePiece from scratch. A fully usable and correctly functioning implementation.

- A raw first run of the actual algorithm as described in the original [Sennrich et al. 2015 paper](https://arxiv.org/abs/1508.07909). → [`experiments/tokenizer/bpe/1-Understanding_BPE.ipynb`](experiments/tokenizer/bpe/1-Understanding_BPE.ipynb)

- A naive implementation from scratch using basic Python. → [`experiments/tokenizer/bpe/2-Naive_BPE.ipynb`](experiments/tokenizer/bpe/2-Naive_BPE.ipynb)

### Naive BPE — What's implemented

- `char_spaced_word_freq()` — converts raw text into character-spaced word frequency dict with `</w>` end-of-word markers
- `bigram_pair_freq()` — counts bigram pair frequencies weighted by word frequency
- `merge()` — merges the most frequent pair across the vocabulary
- `build_vocab()` — constructs `token→id` mapping from base chars + special tokens + merge results
- `save_merge_rules()` / `save_vocab()` — persists training artifacts to disk
- `encode()` — converts raw text to token IDs by replaying merge rules in order
- `decode()` — converts token IDs back to text via reverse vocab lookup + `</w>` boundary reconstruction

**Design decisions:**
- Uses `</w>` end-of-word marker (original paper convention) instead of leading-space convention (GPT-2 style)
- Case-sensitive
- Space-split pre-tokenization (no regex)
- Special tokens: `[UNK]=0, [PAD]=1, [BOS]=2, [EOS]=3`
- Merge rules saved as plain text, one pair per line, in merge order
- Vocab saved as JSON

**Verified:**
- Full encode → decode roundtrip reconstructs original text exactly
- OOV characters correctly map to `[UNK]`
- Merge rule replay during encode is consistent with training

---

## Environment

| Component | Version |
|-----------|---------|
| OS | Ubuntu 24.04.3 LTS |
| GPU | NVIDIA RTX 4050 Laptop (6GB VRAM) |
| CUDA | 12.8 (PyTorch bundled) |
| cuDNN | 9.13.1 |
| Python | 3.11.9 (via uv) |
| PyTorch | 2.10.0+cu128 |
| Package manager | uv |

---

## Repository Structure

```
tinylm/
├── tokenizer/       # BPE tokenizer
├── model/           # Transformer architecture
├── training/        # Training loop
├── inference/       # Inference + sampling
└── observability/   # Metrics, logging

tests/
├── tokenizer/
├── model/
├── training/
└── inference/

experiments/         # Notebooks — one per iterative refinement, 
data/
├── raw/
└── processed/

docs/                # MkDocs documentation (live at thataverageguy.github.io/TinyLM)
```

---

## Documentation

Full implementation notes, design decisions, and phase-by-phase writeups live at:
**[thataverageguy.github.io/TinyLM](https://thataverageguy.github.io/TinyLM)**

---

## License

MIT