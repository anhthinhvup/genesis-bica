"""
Step 0 Micro-Benchmark ($0.05): Head-to-head H100 ($3.95/h) vs B200 ($6.25/h)
on Genesis-v8 Native BICA (Gated DeltaNet + Huginn Shared-KV W=1024 + Engram + PEER).
Also verifies exact mathematical parity of Chunkwise WY Gated DeltaNet vs token-by-token reference.
Strictly <= 140 LOC.
"""
from __future__ import annotations

from pathlib import Path
import time
import modal

HERE = Path(__file__).resolve().parent
app = modal.App("genesis-v8-gpu-bench")

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.6.0", "numpy")
    .add_local_file(str(HERE / "genesis_v8_native.py"), remote_path="/app/genesis_v8_native.py")
)

RATES = {"H100": 3.95, "B200": 6.25}


def _bench_v8(gpu_tag: str, preset: str, batch_size: int, seq_len: int, r_steps: int) -> dict:
    import sys, torch, torch.nn.functional as F
    sys.path.insert(0, "/app")
    from genesis_v8_native import (
        GenesisV8Native, V8_PRESETS, chunkwise_gated_deltanet, recurrent_gated_deltanet_ref,
    )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = "cuda"

    # 1) Exact parity check: Chunkwise WY Gated DeltaNet vs token-by-token reference
    torch.manual_seed(42)
    q = F.normalize(torch.randn(2, 6, 192, 64, device=device), dim=-1)
    k = F.normalize(torch.randn(2, 6, 192, 64, device=device), dim=-1)
    v = torch.randn(2, 6, 192, 64, device=device)
    la = F.logsigmoid(torch.randn(2, 6, 192, device=device) + 1.5).clamp(min=-6.0)
    bt = torch.sigmoid(torch.randn(2, 6, 192, device=device))
    o_chunk = chunkwise_gated_deltanet(q, k, v, la, bt, chunk_size=64)
    o_ref = recurrent_gated_deltanet_ref(q, k, v, la, bt)
    parity_max_diff = float((o_chunk - o_ref).abs().max().item())

    # 2) Build Genesis-v8 Native BICA + compile
    cfg = V8_PRESETS[preset]
    model = GenesisV8Native(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
    compiled = torch.compile(model)
    x = torch.randint(0, cfg.vocab, (batch_size, seq_len), device=device)
    y = torch.randint(0, cfg.vocab, (batch_size, seq_len), device=device)

    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, aux = compiled(x, recur_steps=r_steps)
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab), y.reshape(-1)) + 0.01 * aux
        loss.backward()
        opt.step()
    torch.cuda.synchronize()

    n_iters = 12
    t0 = time.perf_counter()
    for _ in range(n_iters):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, aux = compiled(x, recur_steps=r_steps)
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab), y.reshape(-1)) + 0.01 * aux
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    tok_s = (n_iters * batch_size * seq_len) / dt
    rate = RATES[gpu_tag]
    usd_per_1b = (1e9 / tok_s) * (rate / 3600.0)
    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return {
        "gpu": torch.cuda.get_device_name(0),
        "tag": gpu_tag,
        "preset": preset,
        "params_m": round(cfg.num_params() / 1e6, 2),
        "bs": batch_size,
        "seq": seq_len,
        "R": r_steps,
        "deltanet_parity_max_diff": round(parity_max_diff, 7),
        "tok_s": round(tok_s, 0),
        "usd_per_1B": round(usd_per_1b, 2),
        "toks_for_40usd_B": round(40.0 / usd_per_1b, 2),
        "peak_vram_gb": round(peak_gb, 2),
    }


@app.function(image=gpu_image, gpu="H100", timeout=600)
def bench_h100(preset: str = "v8_465m", batch_size: int = 32, seq_len: int = 1024, r_steps: int = 2) -> dict:
    return _bench_v8("H100", preset, batch_size, seq_len, r_steps)


@app.function(image=gpu_image, gpu="B200", timeout=600)
def bench_b200(preset: str = "v8_465m", batch_size: int = 64, seq_len: int = 1024, r_steps: int = 2) -> dict:
    return _bench_v8("B200", preset, batch_size, seq_len, r_steps)


@app.local_entrypoint()
def main(mode: str = "b200_sweep") -> None:
    if mode == "b200_sweep":
        jobs = [
            ("B200 bs=128 R=1", bench_b200.spawn("v8_465m", 128, 1024, 1)),
            ("B200 bs=192 R=1", bench_b200.spawn("v8_465m", 192, 1024, 1)),
            ("B200 bs=128 R=2", bench_b200.spawn("v8_465m", 128, 1024, 2)),
            ("B200 bs=192 R=2", bench_b200.spawn("v8_465m", 192, 1024, 2)),
        ]
    else:
        jobs = [
            ("H100 R=1", bench_h100.spawn("v8_465m", 32, 1024, 1)),
            ("H100 R=2", bench_h100.spawn("v8_465m", 32, 1024, 2)),
            ("B200 R=1", bench_b200.spawn("v8_465m", 64, 1024, 1)),
            ("B200 R=2", bench_b200.spawn("v8_465m", 64, 1024, 2)),
        ]
    for label, fut in jobs:
        try:
            print(f"[{label}] {fut.get()}", flush=True)
        except Exception as exc:
            print(f"[{label}] ERR: {exc}", flush=True)
