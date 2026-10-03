# Vocab Tax

A compute-matched vocabulary-size study for small code models. Train 12 small LLaMA-style decoders from scratch (3 sizes × 4 BPE vocabulary sizes) on TypeScript/JavaScript and measure, at equal training FLOPs, which vocabulary gives the lowest bits-per-byte.

## The problem

Most portfolios fine-tune someone else's model with someone else's tokenizer. This treats the tokenizer as the variable under test, in the spirit of Tao et al. 2024 ("Scaling Laws with Vocabulary"), in a narrow code domain. It fixes a real weakness in my own work: ORCH-Fusion (2,103 tokens, 272.7M) and ORCH-Next.js-350M-v2 (16,000 tokens, 287M) share a backbone but differ in tokenizer, data and training at once, so today I cannot say whether the small vocabulary helped.

## The approach

1. **Data**: Exact-hash-deduplicated TS/JS shard from codeparrot/github-code-clean
2. **Tokenizers**: 4 byte-level BPE tokenizers (2k, 8k, 16k, 32k vocab) trained on the same bytes
3. **Models**: 3 LLaMA-style sizes (10M, 25M, 50M non-embedding params)
4. **Training**: Same byte budget per model size, 1 epoch
5. **Evaluation**: Held-out bits-per-byte on TS/JS code
6. **Metric**: bpb = total_nats / (ln 2 × total_bytes) — comparable across tokenizers

## Key design decisions

- **Bits-per-byte, not per-token loss**: Per-token loss is not comparable across tokenizers
- **FLOPs include embedding + output layers**: Tiny vocabularies look deceptively cheap without this
- **Second seed for 10M row**: Shows the noise band; a null result is reported as one
- **No extrapolation beyond the grid**: 3 sizes is fragile for fitting a law

## Grid

| Size | Vocab 2k | Vocab 8k | Vocab 16k | Vocab 32k |
|---|---|---|---|---|
| 10M | ✅ | ✅ | ✅ | ✅ |
| 25M | ✅ | ✅ | ✅ | ✅ |
| 50M | ✅ | ✅ | ✅ | ✅ |
| 10M (seed 2) | ✅ | ✅ | ✅ | ✅ |

## Fertility table

| Vocab size | Tokens per byte |
|---|---|
| 2,000 | 0.4475 |
| 8,000 | 0.3667 |
| 16,000 | 0.3435 |
| 32,000 | 0.3275 |

Larger vocabularies have lower fertility — fewer tokens per byte. But they also have larger embedding tables, which cost more FLOPs. The question is whether the quality gain outweighs the compute cost.

## Results

[TBD after grid run]

## Usage

```bash
# Train tokenizers
PYTHONPATH=src python -m voctax.tokenizers

# Run full grid
PYTHONPATH=src python scripts/run_grid.py

# Evaluate
PYTHONPATH=src python scripts/evaluate.py

# Fit curves
PYTHONPATH=src python scripts/fit_curves.py
```

## Limitations

- 3 sizes is fragile for fitting a scaling law
- Differences may sit inside seed noise
- At 10-50M params with 32k vocab, embeddings dominate parameter count
- The 3060's memory bandwidth limits throughput; measure MFU instead of guessing

## License

MIT
