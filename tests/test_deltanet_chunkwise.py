"""Mathematical Verification Tests for Genesis-v8 Native BICA (CPU-friendly CI)."""
from __future__ import annotations

import torch

from training.genesis_v8_native import (
    GenesisV8Native,
    V8Config,
    chunkwise_gated_deltanet,
    recurrent_gated_deltanet_ref,
)


def test_chunkwise_wy_matches_exact_recurrence() -> None:
    """Chunkwise Householder WY Gated DeltaNet matches exact O(T) recurrence."""
    torch.manual_seed(42)
    B, H, T, D = 2, 4, 128, 32
    q = torch.randn(B, H, T, D)
    k = torch.nn.functional.normalize(torch.randn(B, H, T, D), dim=-1)
    v = torch.randn(B, H, T, D)
    log_alpha = torch.nn.functional.logsigmoid(torch.randn(B, H, T) + 1.5)
    beta = torch.sigmoid(torch.randn(B, H, T))

    y_chunk = chunkwise_gated_deltanet(q, k, v, log_alpha, beta, chunk_size=64)
    y_exact = recurrent_gated_deltanet_ref(q, k, v, log_alpha, beta)
    max_diff = (y_chunk.float() - y_exact.float()).abs().max().item()
    assert max_diff < 2e-3, f"Chunkwise WY diverged from exact recurrence: {max_diff}"


def test_genesis_v8_native_forward_recurrent_depths() -> None:
    """Tiny Genesis-v8 Native BICA builds cleanly and runs R=1,2,3 without NaNs."""
    torch.manual_seed(7)
    cfg = V8Config(
        vocab=256,
        d_model=64,
        n_layers=4,
        n_experts=4,
        top_k=2,
        d_ff=128,
        peer_sub_k=16,
        peer_heads=4,
        peer_k=2,
        attn_heads=4,
        attn_window=64,
        delta_heads=4,
        delta_chunk=32,
        engram_buckets=128,
        engram_dim=16,
        engram_heads=2,
        recur_steps=2,
    )
    model = GenesisV8Native(cfg)
    x = torch.randint(0, cfg.vocab, (2, 64))
    for r_depth in (1, 2, 3):
        logits, aux = model(x, recur_steps=r_depth)
        assert logits.shape == (2, 64, cfg.vocab)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(aux)
