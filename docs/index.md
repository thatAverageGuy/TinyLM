# TinyLM

> Building a language model from scratch — pure Python loops to CUDA kernels.

## What is this?

TinyLM is a documented journey of building a language model from absolute scratch.
Not "from scratch using PyTorch's transformer module" — actually from scratch.
Every operation implemented by hand, every gradient derived manually, every
optimization measured before and after.

## Why?

Most LLM learning resources either:

- Hand-wave the internals ("the transformer processes tokens")
- Dump production code without building intuition first

This project does neither. We start with Python loops, break things intentionally,
fix them, profile them, and optimize them one step at a time.

## The Journey

| Phase | What | Why |
|-------|------|-----|
| 0 | Pure Python — forward pass | See the naked math |
| 0.1 | Manual backprop | Truly understand gradients |
| 0.2 | NumPy + PyTorch autograd | Verify your gradients |
| 0.3 | PyTorch manual ops + GPU | Real training |
| 0.4 | PyTorch proper | Establish baseline |
| 1 | Modernization (RoPE, GQA, Flash Attention) | One change at a time |
| 2 | Hardware optimization (CUDA, Triton, quantization) | Squeeze the hardware |

## Hardware

- GPU: NVIDIA RTX 4050 Laptop (6GB VRAM)
- OS: Ubuntu 24.04.3 LTS
- CUDA: 12.8 (via PyTorch wheel)

## Quick Start
```bash
git clone https://github.com/thatAverageGuy/TinyLM.git
cd TinyLM
uv venv --python 3.11.9
source .venv/bin/activate
uv sync
```
