"""YAML-driven training harness for vocab-tax.

Trains TinyLlama models from scratch on TS/JS code.
Grid: 3 sizes x 4 vocab sizes = 12 runs + 4 second-seed runs.
"""
import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from voctax.model import TinyLlama
from voctax.flops import count_training_flops, bits_per_byte

# Model sizes (non-embedding params)
SIZES = {
    "10M": {"dim": 256, "n_layers": 4, "n_heads": 4, "n_kv_heads": 2},
    "25M": {"dim": 384, "n_layers": 6, "n_heads": 6, "n_kv_heads": 3},
    "50M": {"dim": 512, "n_layers": 8, "n_heads": 8, "n_kv_heads": 4},
}

VOCAB_SIZES = [2000, 8000, 16000, 32000]


def train_model(model, tokenizer, train_files: list[str], epochs: int = 1,
                batch_size: int = 32, lr: float = 3e-4, device: str = "cuda",
                max_steps: int = 1000) -> dict:
    """Train a model from scratch."""
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Load all training data
    texts = []
    for f in train_files:
        texts.append(Path(f).read_text(errors="ignore"))

    # Tokenize
    encoded = [tokenizer.encode(t).ids for t in texts]
    # Create batches
    seq_len = 1024
    batches = []
    for e in encoded:
        for i in range(0, len(e) - seq_len, seq_len):
            batches.append(e[i : i + seq_len + 1])

    step = 0
    total_loss = 0.0
    start = time.time()

    for epoch in range(epochs):
        for i in range(0, len(batches), batch_size):
            batch = batches[i : i + batch_size]
            if len(batch) < 2:
                continue
            max_len = max(len(b) for b in batch)
            input_ids = torch.zeros(len(batch), max_len - 1, dtype=torch.long)
            labels = torch.zeros(len(batch), max_len - 1, dtype=torch.long)
            for j, b in enumerate(batch):
                input_ids[j] = torch.tensor(b[:-1])
                labels[j] = torch.tensor(b[1:])
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step += 1
            if step >= max_steps:
                break
        if step >= max_steps:
            break

    elapsed = time.time() - start
    return {
        "steps": step,
        "avg_loss": total_loss / step if step > 0 else 0,
        "elapsed": round(elapsed, 1),
        "flops": count_training_flops(model, batch_size, seq_len) * step,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grid.yaml")
    ap.add_argument("--size", required=True, choices=SIZES)
    ap.add_argument("--vocab", type=int, required=True, choices=VOCAB_SIZES)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from tokenizers import Tokenizer

    # Load config
    config = yaml.safe_load(Path(args.config).read_text())

    # Load tokenizer
    tok = Tokenizer.from_file(f"data/tokenizers/tokenizer_{args.vocab}/tokenizer.json")

    # Create model
    size = SIZES[args.size]
    model = TinyLlama(
        vocab_size=args.vocab,
        dim=size["dim"],
        n_layers=size["n_layers"],
        n_heads=size["n_heads"],
        n_kv_heads=size["n_kv_heads"],
    )

    # Train
    train_files = [str(f) for f in Path("data/dedup").rglob("*") if f.is_file()]
    results = train_model(model, tok, train_files, epochs=args.epochs,
                          batch_size=args.batch_size,
                          max_steps=args.max_steps, device=args.device)

    # Save
    out = Path(f"models/{args.size}_v{args.vocab}_s{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    tok.save(str(out / "tokenizer.json"))
    (out / "results.json").write_text(json.dumps(results, indent=2))

    print(f"Trained {args.size} vocab={args.vocab} seed={args.seed}")
    print(f"  steps={results['steps']}, avg_loss={results['avg_loss']:.4f}")
    print(f"  elapsed={results['elapsed']}s, flops={results['flops']:.2e}")


if __name__ == "__main__":
    main()
