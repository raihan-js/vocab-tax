#!/usr/bin/env python3
"""Fit bpb vs FLOPs curves per vocabulary and find the best vocab at each compute level.

Usage:
  PYTHONPATH=src python scripts/fit_curves.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit


def power_law(x, a, b):
    """bpb = a * FLOPs^(-b)"""
    return a * np.power(x, -b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="data/results/eval.json")
    ap.add_argument("--output", default="data/results/curves.json")
    args = ap.parse_args()

    results = json.loads(Path(args.eval).read_text())

    # Group by vocab size
    by_vocab = {}
    for r in results:
        vocab = r["vocab"]
        if vocab not in by_vocab:
            by_vocab[vocab] = []
        by_vocab[vocab].append(r)

    curves = {}
    for vocab, rows in by_vocab.items():
        # Compute FLOPs for each run
        flops = []
        bpbs = []
        for r in rows:
            # Approximate FLOPs from training results
            # This is a placeholder — real FLOPs come from the training log
            flops.append(r.get("flops", 0))
            bpbs.append(r["bpb"])

        if len(flops) < 3:
            print(f"  vocab={vocab}: insufficient data ({len(flops)} points)")
            continue

        try:
            popt, pcov = curve_fit(power_law, flops, bpbs, p0=[100, 0.1])
            perr = np.sqrt(np.diag(pcov))
            curves[str(vocab)] = {
                "a": popt[0],
                "b": popt[1],
                "a_err": perr[0],
                "b_err": perr[1],
                "n_points": len(flops),
            }
            print(f"  vocab={vocab}: a={popt[0]:.2f}±{perr[0]:.2f}, b={popt[1]:.4f}±{perr[1]:.4f}")
        except Exception as e:
            print(f"  vocab={vocab}: fit failed ({e})")

    # Find best vocab at each compute level
    compute_levels = [1e15, 1e16, 1e17, 1e18]
    best_at_level = {}
    for level in compute_levels:
        best_vocab = None
        best_bpb = float("inf")
        for vocab, params in curves.items():
            bpb = power_law(level, params["a"], params["b"])
            if bpb < best_bpb:
                best_bpb = bpb
                best_vocab = int(vocab)
        best_at_level[f"{level:.0e}"] = {
            "best_vocab": best_vocab,
            "bpb": round(best_bpb, 4),
        }

    output = {
        "curves": curves,
        "best_at_compute": best_at_level,
    }
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"\nSaved to {args.output}")

    print("\nBest vocabulary at each compute level:")
    for level, info in best_at_level.items():
        print(f"  {level} FLOPs: vocab={info['best_vocab']}, bpb={info['bpb']}")


if __name__ == "__main__":
    main()
