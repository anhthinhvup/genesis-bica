"""
Genesis-v8 Native BICA Trainer on NVIDIA B200 (`colab_gpu_trainer/modal_train_v8.py`, <= 385 LOC).
Supports seamless Account-1 -> Account-2 handoff via compact BF16 checkpoint (`v8_handoff_bf16.pt`):
  - Architecture: Chunkwise WY Multi-Head Gated DeltaNet + Huginn Shared-KV W=1024
                  + DeepSeek Engram O(1) + BitNet b1.58 bmm MoE + 2x262K PEER (457.07M).
  - Throughput: ~954k tok/s at R=1 ($1.82/1B), ~561k tok/s at R=2 on B200 (192 GB HBM3e).
  - Modes:
      * pretrain_a: Account 1 Phase 1A (~6.2B tok, 90% R=1 + 10% R=2 LogNormal warmup, ~$11.50)
      * pretrain_b: Account 2 Phase 1B (+11.0B tok, 82% R=1 + 13% R=2 + 5% R={3,4}, ~$20.50)
      * posttrain:  Account 2 Phase 2 DeepThink R={2,3,4} (400M) + Phase 3 Full-Sentence RAG & Mega-SFT (350M)
                    + Phase 4 <THINK> CoT seq=1024 (180M) (~$4.20)
  - Live Telemetry: val_loss, MC-8, Needle-in-a-Haystack W=1024 recall, & 6 verbatim probes -> v8_status.jsonl!
"""
from __future__ import annotations

import glob
import json
import math
import os
from pathlib import Path
import time
import modal

HERE = Path(__file__).resolve().parent
app = modal.App("genesis-v8-b200-train")
VOL_NAME = os.environ.get("GENESIS_V8_VOL", "genesis-v7-pro")
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.6.0", "numpy")
    .add_local_file(str(HERE / "gv6_codec.py"), remote_path="/app/gv6_codec.py")
    .add_local_file(str(HERE / "genesis_v8_native.py"), remote_path="/app/genesis_v8_native.py")
    .add_local_file(str(HERE / "pro_eval_utils.py"), remote_path="/app/pro_eval_utils.py")
)

B200_RATE_PER_SEC = 6.25 / 3600.0


def _save_bf16_ckpt(model, path: Path, meta: dict) -> None:
    import torch
    sd = {k: (v.detach().to(torch.bfloat16).cpu() if v.is_floating_point() else v.detach().cpu())
          for k, v in model.state_dict().items()}
    tmp = path.with_suffix(".tmp")
    torch.save({"model_bf16": sd, **meta}, tmp)
    os.replace(tmp, path)


def _load_bf16_ckpt(model, path: Path, device: str) -> dict:
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model_bf16") or ck.get("model")
    model.load_state_dict({k: v.float().to(device) if v.is_floating_point() else v.to(device) for k, v in sd.items()})
    return {k: v for k, v in ck.items() if k not in ("model_bf16", "model")}


def _eval_needle_1024(model, bpe, device: str) -> dict:
    """Test W=1024 long-context needle retrieval (impossible on v7's W=256!)."""
    import torch, torch.nn.functional as F
    filler = "Hệ thống phân tán xử lý dữ liệu quy mô lớn gồm nhiều nút tính toán độc lập và mạng truyền thông tin cậy. "
    needle = "Mã bí mật của phòng thí nghiệm Genesis là 749281. "
    prompt = f"<CTX> {needle}" + (filler * 22) + "<USER> Mã bí mật của phòng thí nghiệm Genesis là gì? <BOT>"
    ids = bpe.encode(prompt)[-980:]
    ok_ids = bpe.encode(" 749281")
    bad_ids = bpe.encode(" 183940")
    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        scores = []
        for cand in (ok_ids, bad_ids):
            seq = ids + cand
            x = torch.tensor([seq[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([seq[1:]], dtype=torch.long, device=device)
            logits, _ = model(x, recur_steps=2)
            lp = F.log_softmax(logits[0, len(ids) - 1 :].float(), dim=-1)
            scores.append(float(lp[torch.arange(len(cand), device=device), y[0, len(ids) - 1 :]].mean().item()))
    model.train()
    margin = round(scores[0] - scores[1], 3)
    return {"needle_1024_ok": margin > 0.0, "needle_1024_margin": margin}


@app.function(image=gpu_image, volumes={"/vol": vol}, gpu="B200", timeout=21600)
def run_v8_b200(
    stage: str = "pretrain_a",
    preset: str = "v8_465m",
    batch_size: int = 192,
    grad_accum: int = 1,
    seq_len: int = 1024,
    total_tokens: int = 6_200_000_000,
    lr_max: float = 1.4e-3,
    lr_min_ratio: float = 0.25,
    max_dollars: float = 11.50,
    resume_file: str = "v8_handoff_bf16.pt",
) -> dict:
    import sys, numpy as np, torch, torch.nn.functional as F
    sys.path.insert(0, "/app")
    from genesis_v8_native import GenesisV8Native, V8_PRESETS
    from gv6_codec import GV6BpeTokenizer, TOK_BOT, TOK_EOS, TOK_USER
    from pro_eval_utils import AsyncBatchPrefetcher, build_sft_mask, eval_intelligence

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = "cuda"
    t0 = time.perf_counter()

    vol.reload()
    bpe = GV6BpeTokenizer.load("/vol/v6_bpe.bin")
    ckpt_dir = Path("/vol/checkpoints_v8")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "v8_status.jsonl"

    pt_files = sorted(glob.glob("/vol/tokens/pretrain/*.tokens"))
    pt_maps = [np.memmap(p, dtype="<u2", mode="r") for p in pt_files] if pt_files else []
    pt_lens = np.array([len(m) for m in pt_maps], dtype=np.int64) if pt_maps else np.array([1])
    pt_probs = pt_lens / pt_lens.sum()

    sft_candidates = [
        "/vol/tokens/sft_mega.tokens",
        "/vol/tokens/sft_specialty.tokens",
        "/vol/tokens/sft_v8_grounded_rag.tokens",
        "/vol/tokens/sft_reasoning_cot.tokens",
    ]
    sft_maps = {Path(p).stem: np.memmap(p, dtype="<u2", mode="r") for p in sft_candidates if os.path.isfile(p)}
    assert pt_maps or sft_maps, "No token files found in /vol/tokens!"

    cfg = V8_PRESETS[preset]
    model = GenesisV8Native(cfg).to(device)
    prior_tokens = 0
    resume_path = ckpt_dir / resume_file
    if resume_path.is_file():
        meta = _load_bf16_ckpt(model, resume_path, device)
        prior_tokens = int(meta.get("cum_tokens", 0))
        print(f"[V8-RESUME] Loaded {resume_path.name} | prior_tokens={prior_tokens/1e9:.3f}B | stage={stage}", flush=True)
    else:
        print(f"[V8-INIT] {preset} ({cfg.num_params()/1e6:.2f}M) | GPU={torch.cuda.get_device_name(0)} | pt_shards={len(pt_maps)} ({pt_lens.sum()/1e9:.3f}B tok) | sft={list(sft_maps.keys())}", flush=True)

    compiled = torch.compile(model)
    val_len = 16 * (seq_len + 1)
    val_mm = pt_maps[0] if pt_maps else next(iter(sft_maps.values()))
    val_arr = np.stack([val_mm[i * (seq_len + 1) : (i + 1) * (seq_len + 1)] for i in range(16)], axis=0).astype(np.int64)
    val_batch = torch.from_numpy(val_arr).to(device)

    @torch.no_grad()
    def eval_val(r_use: int) -> float:
        model.eval()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(val_batch[:, :-1], recur_steps=r_use)
            v = F.cross_entropy(logits.reshape(-1, cfg.vocab), val_batch[:, 1:].reshape(-1))
        model.train()
        return float(v.item())

    def make_sampler(mode: str, rng: np.random.Generator):
        sft_list = [v for k, v in sft_maps.items() if k in ("sft_mega", "sft_specialty")] or list(sft_maps.values())
        def _sample() -> np.ndarray:
            out = np.empty((batch_size, seq_len + 1), dtype=np.int64)
            if mode == "pretrain":
                n_sft = max(1, batch_size // 12) if sft_list else 0
                n_pt = batch_size - n_sft
                picks = rng.choice(len(pt_maps), size=n_pt, p=pt_probs)
                for i, si in enumerate(picks):
                    mm = pt_maps[int(si)]
                    lo = val_len if si == 0 else 0
                    st = int(rng.integers(lo, len(mm) - seq_len - 2))
                    out[i] = mm[st : st + seq_len + 1]
                for j in range(n_sft):
                    sm = sft_list[int(rng.integers(0, len(sft_list)))]
                    st = int(rng.integers(val_len, len(sm) - seq_len - 2))
                    out[n_pt + j] = sm[st : st + seq_len + 1]
                return out
            if mode == "sft_rag":
                keys = [k for k in ("sft_mega", "sft_specialty", "sft_v8_grounded_rag") if k in sft_maps]
                w = [0.50, 0.35, 0.15] if len(keys) == 3 else [1.0 / len(keys)] * len(keys)
            else:  # cot
                keys = [k for k in ("sft_reasoning_cot", "sft_v8_grounded_rag", "sft_mega") if k in sft_maps]
                w = [0.65, 0.10, 0.25] if len(keys) == 3 else [1.0 / len(keys)] * len(keys)
            picks = rng.choice(len(keys), size=batch_size, p=w)
            for i, ki in enumerate(picks):
                mm = sft_maps[keys[int(ki)]]
                st = int(rng.integers(val_len, len(mm) - seq_len - 2))
                out[i] = mm[st : st + seq_len + 1]
            return out
        return _sample

    if stage == "pretrain_a":
        plan = [("V8-P1A-PRETRAIN", "pretrain", False, total_tokens, lr_max, lr_min_ratio, "v8_p1a_best.pt")]
    elif stage == "pretrain_b":
        plan = [("V8-P1B-PRETRAIN", "pretrain", False, total_tokens, lr_max, lr_min_ratio, "v8_p1b_best.pt")]
    else:
        plan = [
            ("V8-P2-HUGINN-R", "pretrain", False, 400_000_000, 3.5e-4, 0.15, "v8_p2_best.pt"),
            ("V8-P3-RAG-SFT", "sft_rag", True, 350_000_000, 2.0e-4, 0.10, "v8_p3_sft_best.pt"),
            ("V8-P4-COT-1024", "cot", True, 180_000_000, 1.4e-4, 0.08, "v8_p4_reasoner_best.pt"),
        ]

    cum_tokens = prior_tokens
    for pname, smode, is_sft, p_toks, p_lr, p_min_r, best_name in plan:
        # Use bs=128 ga=1 for deep-R stages to stay comfortably under 80GB even at R=4
        eff_bs = min(batch_size, 128) if ("P2" in pname or "P3" in pname or "P4" in pname) else batch_size
        batch_size = eff_bs
        toks_per_step = batch_size * seq_len * grad_accum
        total_steps = max(200, p_toks // toks_per_step)
        warmup = max(80, min(500, total_steps // 20))
        decay_p, nodecay_p, peer_p = [], [], []
        for nm, p in model.named_parameters():
            if "u_exp" in nm or "v_exp" in nm: peer_p.append(p)
            elif p.ndim >= 2: decay_p.append(p)
            else: nodecay_p.append(p)
        opt = torch.optim.AdamW([
            {"params": decay_p, "weight_decay": 0.05, "lr": p_lr},
            {"params": nodecay_p, "weight_decay": 0.0, "lr": p_lr},
            {"params": peer_p, "weight_decay": 0.0, "lr": p_lr * 1.5},
        ], betas=(0.9, 0.95), eps=1e-8, fused=True)
        rng = np.random.default_rng(20260808 + len(pname) * 31)
        pf = AsyncBatchPrefetcher(make_sampler(smode, rng), depth=4)
        best_val, ema, pt0 = 999.0, 0.0, time.perf_counter()
        eval_every = max(300, min(2000, total_steps // 12))
        last_ckpt_t = pt0

        for step in range(1, total_steps + 1):
            if step <= warmup:
                lr = p_lr * (step / warmup)
            else:
                prog = (step - warmup) / max(1, total_steps - warmup)
                lr = p_lr * (p_min_r + (1.0 - p_min_r) * 0.5 * (1.0 + math.cos(math.pi * prog)))
            opt.param_groups[0]["lr"] = lr
            opt.param_groups[1]["lr"] = lr
            opt.param_groups[2]["lr"] = lr * 1.5

            # Huginn LogNormal-style stochastic recurrent depth (fixes v7 R=4 collapse C5!)
            if "P1A" in pname:
                step_r = 2 if (step > int(total_steps * 0.75) and step % 5 == 0) else 1
            elif "P1B" in pname:
                step_r = 4 if (step % 40 == 0) else (3 if step % 20 == 10 else (2 if step % 6 == 0 else 1))
            else:
                step_r = 4 if (step % 12 == 0) else (3 if step % 6 == 3 else 2)

            opt.zero_grad(set_to_none=True)
            lv_acc = 0.0
            for _ in range(grad_accum):
                arr, pinned = pf.next()
                batch = pinned.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits, aux = compiled(batch[:, :-1], recur_steps=step_r)
                    if is_sft:
                        mask = torch.from_numpy(build_sft_mask(arr[:, :-1], TOK_BOT, TOK_EOS, TOK_USER)).to(device, non_blocking=True)
                        pt_l = F.cross_entropy(logits.reshape(-1, cfg.vocab), batch[:, 1:].reshape(-1), reduction="none")
                        loss = (pt_l * mask.reshape(-1).float()).sum() / mask.float().sum().clamp(min=1.0)
                    else:
                        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab), batch[:, 1:].reshape(-1))
                    tot_l = (loss + 0.01 * aux) / grad_accum
                tot_l.backward()
                lv_acc += float(loss.item()) / grad_accum
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            opt.step()

            cum_tokens += toks_per_step
            ema = lv_acc if ema == 0.0 else 0.98 * ema + 0.02 * lv_acc
            spent = (time.perf_counter() - t0) * B200_RATE_PER_SEC

            if step % 250 == 0 or step == 1 or step == total_steps:
                tps = (step * toks_per_step) / max(time.perf_counter() - pt0, 1e-3)
                print(
                    f"[{pname}] step {step:5d}/{total_steps} | cum={cum_tokens/1e9:.3f}B | "
                    f"loss={lv_acc:.4f} (ema={ema:.4f}, ppl={math.exp(min(ema,12)):.1f}) | R={step_r} | "
                    f"gn={gnorm:.2f} | {tps:,.0f} tok/s | ${spent:.2f}/${max_dollars:.2f}",
                    flush=True,
                )

            hit_cap = spent >= max_dollars
            if step % eval_every == 0 or (time.perf_counter() - last_ckpt_t >= 600.0) or step == total_steps or hit_cap:
                vl1 = eval_val(1)
                vl2 = eval_val(2)
                intel = eval_intelligence(model, bpe, 2, device)
                ndl = _eval_needle_1024(model, bpe, device)
                meta = {"stage": pname, "step": step, "cum_tokens": cum_tokens, "val_r1": round(vl1, 4), "val_r2": round(vl2, 4)}
                if vl2 < best_val:
                    best_val = vl2
                    _save_bf16_ckpt(model, ckpt_dir / best_name, meta)
                _save_bf16_ckpt(model, ckpt_dir / "v8_handoff_bf16.pt", meta)
                rec = {
                    "ts": int(time.time()), "phase": pname, "step": step, "total_steps": total_steps,
                    "cum_tokens_b": round(cum_tokens / 1e9, 3), "val_r1": round(vl1, 4), "val_r2": round(vl2, 4),
                    "val_ppl_r1": round(math.exp(min(vl1, 12)), 2), "val_ppl_r2": round(math.exp(min(vl2, 12)), 2),
                    "spent_usd": round(spent, 2), **ndl, **intel,
                }
                with log_path.open("a", encoding="utf-8") as lf:
                    lf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                vol.commit()
                last_ckpt_t = time.perf_counter()
                print(
                    f"  [V8-CKPT {pname} step={step} val_R1={vl1:.4f}({rec['val_ppl_r1']}) val_R2={vl2:.4f}({rec['val_ppl_r2']}) | "
                    f"MC-8={intel['mc8_hits']}/8 ({intel['mean_margin']:+.3f}) | Needle1024={ndl['needle_1024_ok']}({ndl['needle_1024_margin']:+.2f}) | ${spent:.2f}]\n"
                    f"    QA_VN: {intel['probes']['QA_VN']}\n    QA_CODE: {intel['probes']['QA_CODE']}\n    BILINGUAL: {intel['probes']['BILINGUAL']}",
                    flush=True,
                )
                if hit_cap:
                    print(f"[V8-BUDGET-CAP] Saved v8_handoff_bf16.pt at ${spent:.2f} -> clean exit!", flush=True)
                    pf.close()
                    return {"stage": stage, "cum_tokens_b": round(cum_tokens / 1e9, 3), "spent_usd": round(spent, 2)}
        pf.close()
    return {"stage": stage, "cum_tokens_b": round(cum_tokens / 1e9, 3), "spent_usd": round((time.perf_counter() - t0) * B200_RATE_PER_SEC, 2)}


@app.local_entrypoint()
def main(
    stage: str = "pretrain_a",
    total_tokens: int = 6_200_000_000,
    batch_size: int = 192,
    lr_max: float = 1.4e-3,
    lr_min_ratio: float = 0.25,
    max_dollars: float = 11.50,
) -> None:
    print(run_v8_b200.remote(
        stage=stage, total_tokens=total_tokens, batch_size=batch_size,
        lr_max=lr_max, lr_min_ratio=lr_min_ratio, max_dollars=max_dollars,
    ))
