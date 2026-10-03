# AGENTS.md — vocab-tax

Portfolio project for Raihan Sikder. Target roles: Noeon Research (Senior ML Engineer, LLMOps), PayPay Card, Money Forward, Treasure AI, Citadel AI.

## Project: Vocab Tax

A compute-matched vocabulary-size study for small code models. Train 12 small LLaMA-style decoders from scratch (3 sizes × 4 BPE vocabulary sizes) on TypeScript/JavaScript and measure, at equal training FLOPs, which vocabulary gives the lowest bits-per-byte.

## Why this project

Most portfolios fine-tune someone else's model with someone else's tokenizer. This treats the tokenizer as the variable under test, in the spirit of Tao et al. 2024 ("Scaling Laws with Vocabulary"), in a narrow code domain. It fixes a real weakness in my own work: ORCH-Fusion (2,103 tokens, 272.7M) and ORCH-Next.js-350M-v2 (16,000 tokens, 287M) share a backbone but differ in tokenizer, data and training at once, so today I cannot say whether the small vocabulary helped.

## Stack

Python, PyTorch, Hugging Face Transformers, tokenizers (byte-level BPE), numpy, scipy, pytest.

## Compute

One RTX 3060 12GB. Grid: 3 sizes (10M, 25M, 50M non-embedding) × 4 vocab sizes (2k, 8k, 16k, 32k) = 12 runs + 4 second-seed runs. Realistic budget about 60-80 GPU-hours including overheads, run overnight across about 2-3 weeks.

## Milestones

1. **Data and tokenizers** (4d) — deduplicated TS/JS shard, 4 BPE tokenizers, fertility table
2. **Harness and sanity runs** (4d) — YAML training, FLOP counter, bpb eval, smoke runs
3. **Full grid** (10d) — 12 runs + 4 second-seed runs, checkpoints to HF
4. **Fit and write-up** (5d) — bpb vs FLOPs curves, best vocabulary, dev.to article

## Conventions

- Python 3.10+, pytest for all model and FLOP tests.
- All models compared at equal training FLOPs.
- bpb is the only comparable metric across tokenizers.
- FLOPs include embedding and output layers.
- Every result cites the exact config and seed.

## Development

```bash
cd vocab-tax
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v

# Train tokenizers
PYTHONPATH=src python -m voctax.tokenizers

# Run full grid
PYTHONPATH=src python scripts/run_grid.py

# Evaluate
PYTHONPATH=src python scripts/evaluate.py

# Fit curves
PYTHONPATH=src python scripts/fit_curves.py
```

## Current status

- Repo structure created
- Model, FLOP counter, bpb eval implemented
- Tests written
- Pending: data pipeline, tokenizer training, full grid, curve fitting
