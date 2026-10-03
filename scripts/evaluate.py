#!/usr/bin/env python3
"""Evaluate all trained models: bits-per-byte on held-out TS/JS.

Usage:
  PYTHONPATH=src python scripts/evaluate.py
"""
import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from voctax.model import TinyLlama
from voctax.flops import bits_per_byte

SIZES = {
    "10M": {"dim": 256, "n_layers": 4, "n_heads": 4, "n_kv_heads": 2},
    "25M": {"dim": 384, "n_layers": 6, "n_heads": 6, "n_kv_heads": 3},
    "50M": {"dim": 512, "n_layers": 8, "n_heads": 8, "n_kv_heads": 4},
}

VOCAB_SIZES = [2000, 8000, 16000, 32000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # Load eval data
    eval_dir = Path("data/eval")
    eval_texts = []
    for f in sorted(eval_dir.glob("*.ts"))[:1000]:
        eval_texts.append(f.read_text(errors="ignore"))
    print(f"Eval data: {len(eval_texts)} files")

    results = []
    for model_dir in sorted(Path("models").glob("*_v*_s*")):
        parts = model_dir.name.split("_")
        size_name, vocab_s, seed_s = parts[0], parts[1][1:], parts[2][1:]
        size = SIZES[size_name]
        vocab = int(vocab_s)
        seed = int(seed_s)
        if vocab not in VOCAB_SIZES:
            continue

        print(f"  Evaluating {size_name} vocab={vocab} seed={seed}...", flush=True)
        tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        model = TinyLlama(
            vocab_size=vocab,
            dim=size["dim"],
            n_layers=size["n_layers"],
            n_heads=size["n_heads"],
            n_kv_heads=size["n_kv_heads"],
        )
        state = torch.load(model_dir / "model.pt", map_location="cpu")
        model.load_state_dict(state)
        model = model.to(args.device)
        model.eval()

        bpb = bits_per_byte(model, tok, eval_texts, device=args.device)
        non_emb = model.count_params(non_embedding=True)
        total = model.count_params()

        results.append({
            "size": size_name,
            "vocab": vocab,
            "seed": seed,
            "bpb": round(bpb, 4),
            "non_embedding_params": non_emb,
            "total_params": total,
        })
        print(f"    bpb={bpb:.4f}, non_emb={non_emb:,}, total={total:,}")

    out = Path("data/results")
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved {len(results)} evaluations to {out / 'eval.json'}")


if __name__ == "__main__":
    main()
