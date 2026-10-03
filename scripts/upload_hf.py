#!/usr/bin/env python3
"""Upload vocab-tax artifacts to Hugging Face."""
import json
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi()
REPO = "raihan-js/vocab-tax-grid"

api.create_repo(REPO, repo_type="dataset", exist_ok=True)

for local, remote in [
    ("data/tokenizers/fertility.json", "fertility.json"),
    ("data/results/eval.json", "eval.json"),
    ("data/results/grid.json", "grid.json"),
]:
    p = Path(local)
    if p.exists():
        api.upload_file(path_or_fileobj=str(p), path_in_repo=remote,
                        repo_id=REPO, repo_type="dataset")
        print(f"  uploaded {remote}")

readme = """---
license: apache-2.0
task_categories:
- text-generation
tags:
- vocabulary
- scaling
- tokenizer
pretty_name: Vocab Tax Grid
---

# Vocab Tax Grid

Compute-matched vocabulary-size study for small code models.

- Fertility table: tokens/byte for 2k/8k/16k/32k BPE on TS/JS
- eval.json: bits-per-byte for 16 runs (3 sizes x 4 vocabs + 4 second seeds)
- grid.json: training logs (steps, loss, FLOPs)

Key finding: 8k-16k wins at every size; 10M+8k beats 50M+2k.
"""
api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md",
                repo_id=REPO, repo_type="dataset")
print("  uploaded README.md")
print(f"Done: {REPO}")
