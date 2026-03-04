# Python & uv

## Why uv?

The Python packaging ecosystem has historically been fragmented — `pip` for packages, `virtualenv` for environments, `pyenv` for Python versions, `pip-tools` for lockfiles. Each tool solves one problem and they don't always play nicely together.

**uv** (by Astral, the people behind `ruff`) replaces all of them:

| Old way | uv equivalent |
|---------|--------------|
| `pyenv install 3.11.9` | `uv python install 3.11.9` |
| `python -m venv .venv` | `uv venv` |
| `pip install torch` | `uv add torch` |
| `pip-compile requirements.in` | `uv pip compile` |

It's written in Rust. Installing a package that would take pip 30 seconds takes uv under 2 seconds. For a project where you'll be iterating on dependencies frequently, this matters.

---

## Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv --version
```

---

## Why Python 3.11.9?

PyTorch's ecosystem (triton, flash-attn, various CUDA extensions) is most stable on Python 3.11. Python 3.12 introduced some packaging changes that cause occasional friction with ML packages. Since we'll be writing CUDA kernels in later phases, 3.11 is the safer choice.

```bash
uv python install 3.11.9
```

---

## Project setup

```bash
git clone https://github.com/thatAverageGuy/TinyLM.git
cd TinyLM

# Create virtualenv pinned to 3.11.9
uv venv --python 3.11.9

# Activate
source .venv/bin/activate

# Verify
python --version  # Python 3.11.9
```

---

## Installing PyTorch

On Linux, PyPI now hosts GPU-accelerated PyTorch wheels directly. The default `torch` package on PyPI for Linux targets CUDA 12.8 as of PyTorch 2.9+. So:

```bash
uv add torch
```

That's it. No `--index-url`, no special flags. uv resolves the CUDA 12.8 GPU wheel automatically on Linux.

This works because PyTorch embeds its own CUDA 12.8 runtime — you don't need CUDA 12.8 installed system-wide. Your NVIDIA driver just needs to support CUDA 12.8 runtime (any driver 520+ does).

---

## Installing all dependencies

```bash
# Core
uv add torch numpy

# Dev tooling
uv add --dev pytest pytest-cov black ruff

# Docs
uv add --dev mkdocs==1.6.1 mkdocs-material==9.5.49
```

We pin `mkdocs` and `mkdocs-material` versions explicitly because MkDocs 2.0 introduced breaking changes incompatible with the Material theme. Pinning prevents the CI from silently upgrading and breaking the docs build.

---

## Verify the full setup

```python
import torch
import numpy as np

# PyTorch version and CUDA
print(torch.__version__)              # 2.10.0+cu128
print(torch.cuda.is_available())      # True
print(torch.cuda.get_device_name(0))  # NVIDIA GeForce RTX 4050 Laptop GPU

# Quick tensor on GPU
x = torch.randn(3, 3).cuda()
print(x.device)                       # cuda:0

# NumPy
print(np.__version__)                 # 2.x.x
```

---

## pyproject.toml

uv manages dependencies through `pyproject.toml`. After running the install commands above, your `pyproject.toml` will look like:

```toml
[project]
name = "tinylm"
version = "0.1.0"
requires-python = ">=3.11.9"
dependencies = [
    "torch>=2.10.0",
    "numpy>=2.0.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=5.0.0",
    "black>=24.0.0",
    "ruff>=0.4.0",
    "mkdocs==1.6.1",
    "mkdocs-material==9.5.49",
]
```

Anyone cloning the repo can reproduce the exact environment with:

```bash
uv sync
```

That single command reads `pyproject.toml`, creates the virtualenv, and installs everything. No more "works on my machine".

---

## What's next

With Python and dependencies in place, let's look at how the project is structured. → [Project Structure](project.md)