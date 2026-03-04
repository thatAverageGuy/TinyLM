# Setup Overview

Before writing a single line of model code, we need a clean, reproducible environment. This section documents exactly what we installed, why we made each choice, and how to verify everything works.

## What we're setting up

| Component | Choice | Why |
|-----------|--------|-----|
| OS | Ubuntu 24.04.3 LTS | Bare metal Linux — no WSL2 overhead, direct CUDA access |
| GPU Driver | NVIDIA 590.48.01 | Latest stable, supports CUDA 12.x runtime |
| CUDA | 12.8 (via PyTorch) | PyTorch bundles its own runtime — system CUDA irrelevant |
| Python | 3.11.9 via uv | Best PyTorch ecosystem compatibility |
| Package manager | uv | Replaces pip + virtualenv + pyenv. Written in Rust. Fast. |

## Why Ubuntu on bare metal?

We could have used WSL2 on Windows or a Docker container. We didn't, for one specific reason: **Phase 2 of this project involves writing CUDA kernels**. When you get to custom CUDA ops, Triton kernels, and `torch.compile`, WSL2 adds a friction layer that causes real pain — driver mismatches, PCIe passthrough quirks, profiler limitations.

Bare metal Ubuntu means direct access to the GPU with zero abstraction layers between your code and the hardware. For a project about understanding the hardware, that matters.

## Setup pages

- [Ubuntu & CUDA](ubuntu.md) — driver verification, CUDA situation explained
- [Python & uv](python.md) — why uv, how to install, environment setup
- [Project Structure](project.md) — repo layout and what lives where