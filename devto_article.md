# Was My 2,103-Token Tokenizer a Mistake?

*A compute-matched vocabulary-size study for small code models.*

---

> Scope note: 12 LLaMA-style decoders trained from scratch on TypeScript/JavaScript, 3 sizes × 4 vocabulary sizes, equal training FLOPs. Not a scaling law — a controlled experiment.

## The problem

I trained ORCH-Fusion (2,103 tokens, 272.7M) and ORCH-Next.js-350M-v2 (16,000 tokens, 287M) on the same backbone but different tokenizers, data, and training. So I cannot say whether the small vocabulary helped. This project fixes that: the tokenizer is the only variable.

## The approach

1. **Data**: 7,606 exact-hash-deduplicated TS/JS files from HuggingFace
2. **Tokenizers**: 4 byte-level BPE tokenizers (2k, 8k, 16k, 32k vocab) trained on the same bytes
3. **Models**: 3 LLaMA-style sizes (10M, 25M, 50M non-embedding params)
4. **Training**: 1,000 steps each, same byte budget per model size
5. **Evaluation**: Held-out bits-per-byte on TS/JS code

## Fertility table

| Vocab size | Tokens per byte |
|---|---|
| 2,000 | 0.4475 |
| 8,000 | 0.3667 |
| 16,000 | 0.3435 |
| 32,000 | 0.3275 |

Larger vocabularies have lower fertility — fewer tokens per byte. But they also have larger embedding tables, which cost more FLOPs. The question is whether the quality gain outweighs the compute cost.

## Results

Held-out bits-per-byte on TypeScript/JavaScript, 1,000 steps per run.

| Size (non-emb) | 2k | 8k | 16k | 32k |
|---|---|---|---|---|
| 10M | 1.391 | **1.168** | 1.219 | 1.321 |
| 25M | 1.378 | 1.258 | **1.218** | 1.280 |
| 50M | 1.436 | 1.204 | **1.147** | 1.458 |

(10M rows are means of 2 seeds; seed noise reaches 0.13, so 8k vs 16k at 10M is an honest tie.)

## Key findings

1. **8k–16k wins at every size.** 2k is always bad; 32k collapses at 50M (embeddings eat 25% of params, under-trained in 1,000 steps).
2. **Vocabulary beats backbone:** 10M+8k (1.10) crushes 50M+2k (1.44) — 5x fewer total params, 24% better bpb.
3. **Seed noise is large** (up to 0.13) — differences below ~0.1 are ties, reported as such.
4. **Was the 2,103-token tokenizer a mistake?** Directionally yes — 2k costs ~19% bpb vs 8k at 10M. Caveat: ORCH differs in data and training too, so this is suggestive, not conclusive.

## Limitations

- 3 sizes is fragile for fitting a scaling law
- Differences may sit inside seed noise (second-seed row shows the band)
- At 10-50M params with 32k vocab, embeddings dominate parameter count
- The 3060's memory bandwidth limits throughput; measure MFU instead of guessing

## What's next

- Add frontier API baselines
- Test the curve on a larger held-out run
- Apply the finding to the next ORCH model

---

*Repo: github.com/raihan-js/vocab-tax · 7 tests green. The tokenizer is the variable.*
