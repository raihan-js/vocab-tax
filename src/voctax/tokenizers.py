"""Train byte-level BPE tokenizers at 4 vocabulary sizes.

Trains on the same deduplicated TS/JS bytes so the only variable is vocab size.
Outputs: tokenizer.json per size + fertility table (tokens per byte).
"""
import argparse
import json
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer


VOCAB_SIZES = [2000, 8000, 16000, 32000]


def train_tokenizers(data_dir: Path, output_dir: Path,
                     vocab_sizes: list[int] = VOCAB_SIZES) -> dict:
    """Train one BPE tokenizer per vocabulary size on the same data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(data_dir.rglob("*"))
    files = [str(f) for f in files if f.is_file()]
    print(f"Training on {len(files)} files")

    results = {}
    for size in vocab_sizes:
        print(f"\nTraining vocab size {size}...", flush=True)
        tok = ByteLevelBPETokenizer()
        tok.train(files, vocab_size=size, min_frequency=2,
                  special_tokens=["<pad>", "<eos>", "<bos>", "<unk>"])
        out = output_dir / f"tokenizer_{size}"
        out.mkdir(parents=True, exist_ok=True)
        tok.save(str(out / "tokenizer.json"))
        # Fertility: tokens per byte on a sample
        sample = files[: min(1000, len(files))]
        total_tokens = 0
        total_bytes = 0
        for f in sample:
            text = Path(f).read_text(errors="ignore")
            total_bytes += len(text.encode("utf-8"))
            total_tokens += len(tok.encode(text).ids)
        fertility = total_tokens / total_bytes if total_bytes > 0 else 0
        results[str(size)] = {
            "vocab_size": size,
            "fertility": round(fertility, 4),
            "total_tokens": total_tokens,
            "total_bytes": total_bytes,
        }
        print(f"  fertility: {fertility:.4f} tokens/byte")

    # Save fertility table
    (output_dir / "fertility.json").write_text(json.dumps(results, indent=2))
    print(f"\nFertility table saved to {output_dir / 'fertility.json'}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dedup")
    ap.add_argument("--output", default="data/tokenizers")
    args = ap.parse_args()

    train_tokenizers(Path(args.data), Path(args.output))


if __name__ == "__main__":
    main()
