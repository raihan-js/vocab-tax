import pytest
import torch
from voctax.model import TinyLlama
from voctax.flops import count_training_flops, bits_per_byte


class TestModel:
    def test_forward_shape(self):
        model = TinyLlama(vocab_size=2000, dim=256, n_layers=4, n_heads=4, n_kv_heads=2)
        x = torch.randint(0, 2000, (2, 16))
        out = model(x)
        assert out.shape == (2, 16, 2000)

    def test_param_count(self):
        model = TinyLlama(vocab_size=2000, dim=256, n_layers=4, n_heads=4, n_kv_heads=2)
        total = model.count_params()
        non_emb = model.count_params(non_embedding=True)
        assert total > non_emb
        assert non_emb > 0

    def test_different_sizes(self):
        for name, cfg in [("10M", {"dim": 256, "n_layers": 4, "n_heads": 4, "n_kv_heads": 2}),
                          ("25M", {"dim": 384, "n_layers": 6, "n_heads": 6, "n_kv_heads": 3}),
                          ("50M", {"dim": 512, "n_layers": 8, "n_heads": 8, "n_kv_heads": 4})]:
            model = TinyLlama(vocab_size=2000, **cfg)
            non_emb = model.count_params(non_embedding=True)
            assert non_emb > 0


class TestFLOPs:
    def test_flops_positive(self):
        model = TinyLlama(vocab_size=2000, dim=256, n_layers=4, n_heads=4, n_kv_heads=2)
        flops = count_training_flops(model, batch_size=32, seq_len=1024)
        assert flops > 0

    def test_flops_scale_with_batch(self):
        model = TinyLlama(vocab_size=2000, dim=256, n_layers=4, n_heads=4, n_kv_heads=2)
        f1 = count_training_flops(model, batch_size=32, seq_len=1024)
        f2 = count_training_flops(model, batch_size=64, seq_len=1024)
        assert f2 > f1

    def test_flops_include_embedding(self):
        model = TinyLlama(vocab_size=2000, dim=256, n_layers=4, n_heads=4, n_kv_heads=2)
        flops = count_training_flops(model, batch_size=32, seq_len=1024)
        # Embedding + output should be a significant fraction
        non_emb = model.count_params(non_embedding=True)
        vocab = 2000
        dim = 256
        embed_flops = 2 * vocab * dim * 32 * 1024 * 3
        assert flops > embed_flops


class TestBPB:
    def test_bpb_positive(self):
        model = TinyLlama(vocab_size=2000, dim=256, n_layers=4, n_heads=4, n_kv_heads=2)
        # Mock tokenizer
        class MockTok:
            def encode(self, text):
                return list(range(min(len(text), 100)))
        bpb = bits_per_byte(model, MockTok(), ["hello world"], batch_size=1, device="cpu")
        assert bpb > 0
