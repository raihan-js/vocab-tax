#!/usr/bin/env python3
"""Run the full vocab-tax grid: 3 sizes x 4 vocab sizes = 12 runs + 4 second-seed runs.

Usage:
  PYTHONPATH=src python scripts/run_grid.py
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

SIZES = ["10M", "25M", "50M"]
VOCAB_SIZES = [2000, 8000, 16000, 32000]
SEEDS = [42, 123]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grid.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    runs = []

    # Main grid: 3 sizes x 4 vocab sizes, seed 42
    for size in SIZES:
        for vocab in VOCAB_SIZES:
            runs.append({"size": size, "vocab": vocab, "seed": 42})

    # Second seed for 10M row (noise band)
    for vocab in VOCAB_SIZES:
        runs.append({"size": "10M", "vocab": vocab, "seed": 123})

    print(f"Total runs: {len(runs)}")
    for r in runs:
        print(f"  {r['size']} vocab={r['vocab']} seed={r['seed']}")

    if args.dry_run:
        return

    results = []
    for i, r in enumerate(runs):
        model_dir = Path(f"models/{r['size']}_v{r['vocab']}_s{r['seed']}")
        if (model_dir / "results.json").exists():
            print(f"=== Run {i+1}/{len(runs)}: {r['size']} vocab={r['vocab']} seed={r['seed']} — SKIPPED (done) ===", flush=True)
            res = json.loads((model_dir / "results.json").read_text())
            res.update(r)
            results.append(res)
            continue
        print(f"\n=== Run {i+1}/{len(runs)}: {r['size']} vocab={r['vocab']} seed={r['seed']} ===", flush=True)
        cmd = [
            sys.executable, "-m", "voctax.train",
            "--size", r["size"],
            "--vocab", str(r["vocab"]),
            "--seed", str(r["seed"]),
            "--batch-size", "4",
            "--device", args.device,
        ]
        import os
        env = dict(os.environ)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        print(result.stdout)
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr[-500:]}")
            continue

        # Load results
        if (model_dir / "results.json").exists():
            res = json.loads((model_dir / "results.json").read_text())
            res.update(r)
            results.append(res)

        # Allow GPU memory to be freed between runs
        import time
        time.sleep(30)

    # Save all results
    out = Path("data/results")
    out.mkdir(parents=True, exist_ok=True)
    (out / "grid.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved {len(results)} runs to {out / 'grid.json'}")


if __name__ == "__main__":
    main()
