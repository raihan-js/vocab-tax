"""Data pipeline: fetch, deduplicate, and prepare TS/JS code for tokenizer training.

Uses codeparrot/github-code-clean as a fallback (the-stack-dedup is gated).
Exact-hash deduplication on file content.
"""
import argparse
import hashlib
import json
from pathlib import Path


def deduplicate_files(input_dir: Path, output_dir: Path) -> dict:
    """Deduplicate files by exact content hash.

    Returns stats: total, unique, duplicates.
    """
    seen = set()
    total = unique = duplicates = 0
    output_dir.mkdir(parents=True, exist_ok=True)

    for f in sorted(input_dir.rglob("*")):
        if not f.is_file():
            continue
        total += 1
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        if h in seen:
            duplicates += 1
            continue
        seen.add(h)
        unique += 1
        # Preserve relative path
        rel = f.relative_to(input_dir)
        out = output_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(f.read_bytes())

    return {"total": total, "unique": unique, "duplicates": duplicates}


def fetch_code_data(output_dir: Path, n_files: int = 50000) -> dict:
    """Fetch TS/JS code from HuggingFace datasets.

    Uses ajibawa-2023/JavaScript-Code-Large and petrpan26/typescript-code.
    """
    from datasets import load_dataset

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    # JavaScript
    ds = load_dataset("ajibawa-2023/JavaScript-Code-Large", split="train", streaming=True)
    for item in ds:
        code = item.get("code", "")
        if not code:
            continue
        h = hashlib.sha256(code.encode()).hexdigest()
        (output_dir / f"{h}.js").write_text(code)
        count += 1
        if count >= n_files // 2:
            break

    # TypeScript
    ds = load_dataset("petrpan26/typescript-code", split="train", streaming=True)
    for item in ds:
        code = item.get("content", "")
        if not code:
            continue
        h = hashlib.sha256(code.encode()).hexdigest()
        (output_dir / f"{h}.ts").write_text(code)
        count += 1
        if count >= n_files:
            break

    return {"fetched": count}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/raw")
    ap.add_argument("--n-files", type=int, default=50000)
    args = ap.parse_args()

    output = Path(args.output)
    stats = fetch_code_data(output, n_files=args.n_files)
    print(f"Fetched {stats['fetched']} files")

    # Deduplicate
    dedup_dir = Path("data/dedup")
    dedup_stats = deduplicate_files(output, dedup_dir)
    print(f"Dedup: {dedup_stats}")


if __name__ == "__main__":
    main()
