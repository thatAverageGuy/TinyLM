# Project Structure

## Repository layout

```
tinylm/
│
├── tinylm/                        # Source code
│   ├── __init__.py
│   ├── tokenizer/                 # BPE tokenizer — pure Python → optimized
│   │   └── __init__.py
│   ├── model/                     # Transformer model — pure Python → PyTorch
│   │   └── __init__.py
│   ├── training/                  # Training loop, optimizer, scheduler
│   │   └── __init__.py
│   ├── inference/                 # Prefill, decode loop, sampling, KV cache
│   │   └── __init__.py
│   └── observability/             # Profiling, metrics, logging
│       └── __init__.py
│
├── tests/                         # One test directory per source module
│   ├── __init__.py
│   ├── tokenizer/
│   ├── model/
│   ├── training/
│   └── inference/
│
├── docs/                          # MkDocs documentation source
│   ├── index.md
│   ├── setup/
│   ├── phase0/                    # Pure Python
│   ├── phase01/                   # Manual autograd
│   ├── phase02/                   # NumPy + PyTorch autograd
│   ├── phase03/                   # PyTorch manual ops
│   ├── phase04/                   # PyTorch proper
│   ├── phase1/                    # Modernization
│   ├── phase2/                    # Hardware optimization
│   └── assets/diagrams/           # ASCII and rendered diagrams
│
├── experiments/                   # One-off scripts, notebooks, explorations
│   └── (not committed to main)
│
├── data/
│   ├── raw/                       # Original downloaded datasets
│   └── processed/                 # Tokenized, binary-format training data
│
├── .github/
│   └── workflows/
│       └── docs.yml               # Auto-deploy docs on push to main
│
├── mkdocs.yml                     # Docs site configuration
├── pyproject.toml                 # Dependencies managed by uv
├── uv.lock                        # Lockfile — exact versions, reproducible
├── .python-version                # Python version pin for uv
├── .gitignore                     # Python + data files ignored
└── README.md
```

---

## Design decisions

**Why separate `tinylm/` source from `tests/`?**

Standard Python project layout. Keeps source and tests cleanly separated. pytest discovers tests automatically in the `tests/` directory.

**Why `data/raw` and `data/processed` separately?**

Raw data is the source of truth — never modified. Processing (tokenization, binary encoding) is reproducible from raw. If the processing pipeline changes, you re-run it from raw. This pattern prevents the nightmare of "which version of the data did I train on?"

Neither `data/raw` nor `data/processed` are committed to git (both in `.gitignore`). Datasets are downloaded separately.

**Why `experiments/` not committed?**

Experiments are exploratory — one-off scripts to test an idea, Jupyter notebooks to visualize attention weights, quick benchmarks. They're not production code and shouldn't pollute the main branch. Keep them local.

**Why `observability/` as its own module?**

From day one, every training run and inference call is instrumented. Profiling, loss curves, GPU utilization, tokens/sec — these aren't bolted on later. They're first-class. The observability module contains the logging setup, metric collectors, and profiler wrappers that every other module imports.

---

## Running tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=tinylm --cov-report=term-missing

# Run specific module tests
pytest tests/tokenizer/
```

---

## Code style

```bash
# Format
black tinylm/ tests/

# Lint
ruff check tinylm/ tests/

# Both (run before every commit)
black tinylm/ tests/ && ruff check tinylm/ tests/
```

---

## What's next

Environment is set up, repo is structured, docs are live. Time to build. → [Phase 0 Overview](../phase0/overview.md)