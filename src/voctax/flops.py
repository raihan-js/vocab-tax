"""FLOP counter and bits-per-byte evaluation.

FLOPs include embedding and output layers (where tiny vocabularies look deceptively cheap).
bpb = total_nats / (ln(2) * total_bytes) — comparable across tokenizers.
"""
import math

import torch


def count_training_flops(model, batch_size: int, seq_len: int) -> int:
    """Count training FLOPs for one forward+backward pass.

    Includes embedding lookup and output projection.
    Formula: 6 * N * B * S (forward + backward) where N = non-embedding params.
    Plus embedding/output: 2 * V * D * B * S (forward) * 3 (fwd+bwd).
    """
    non_emb = model.count_params(non_embedding=True)
    vocab_size = model.embed.weight.shape[0]
    dim = model.embed.weight.shape[1]

    # Transformer FLOPs: 6 * N * B * S
    transformer_flops = 6 * non_emb * batch_size * seq_len

    # Embedding + output: 2 * V * D * B * S (fwd) * 3 (fwd+bwd)
    embed_output_flops = 2 * vocab_size * dim * batch_size * seq_len * 3

    return transformer_flops + embed_output_flops


def bits_per_byte(model, tokenizer, texts: list[str],
                  batch_size: int = 8, device: str = "cuda",
                  max_len: int = 1024) -> float:
    """Compute bits-per-byte on a held-out set.

    bpb = total_nats / (ln(2) * total_bytes)
    """
    model.eval()
    total_nats = 0.0
    total_bytes = 0

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = [tokenizer.encode(t).ids[:max_len] for t in batch]
            max_len = max(len(e) for e in encoded)
            # Pad
            input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
            for j, e in enumerate(encoded):
                input_ids[j, : len(e)] = torch.tensor(e)
            input_ids = input_ids.to(device)

            logits = model(input_ids)
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )
            total_nats += loss.item()
            total_bytes += sum(len(t.encode("utf-8")) for t in batch)

    return total_nats / (math.log(2) * total_bytes) if total_bytes > 0 else 0.0
