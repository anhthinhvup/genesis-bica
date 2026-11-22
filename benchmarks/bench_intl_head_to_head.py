"""Official International Standard Benchmark Head-to-Head on Local CPU (`i5-10210U`).
Evaluates `Project Genesis` symmetrically against quantized Multi-B GGUF LLMs
(`Qwen2.5-0.5B`, `Llama-3.2-1B`, `Qwen2.5-1.5B`) on SHA-256 locked HuggingFace test splits:
  1. FOLIO (`tasksource/folio`, Yale NLP First-Order Logic, 3-way True/False/Uncertain)
  2. HumanEval (`openai/openai_humaneval`, Real Isolated Python Unit Tests pass@1)
  3. MATH-500 (`HuggingFaceH4/MATH-500`, Hendrycks Competition Math exact match)
  4. Optional Part-B Public-120 (`MBPP-40 + GSM8K-40 + TruthfulQA-20 + UIT-ViQuAD2-20`)
Strictly <= 320 LOC | 0% `import re` | 0 bytes on `C:\\`.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import shutil
import statistics
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mel.chat3m.disk_guard import DEFAULT_CHAT3M_MODEL_DIR, enforce_e_drive_environment

enforce_e_drive_environment()

from experiments.arena_evaluators import (
    GenesisChatAdapter,
    LlamaCppChatAdapter,
    _run_python_in_sandbox,
    extract_python_block,
)
from experiments.bench_head_to_head_cpu import evaluate_part_b_public
from experiments.public_bench_loader import load_all_public_benchmarks
from mel.chat3m.router_mel import GenesisChat3MEngine

WORLD_DIR = ROOT / ".cache_e" / "world_benchmarks"
ARENA_DIR = ROOT / ".cache_e" / "llm_arena"
SERVER_EXE = ARENA_DIR / "bin" / "llama-server.exe"
MODELS_DIR = ARENA_DIR / "models"
SBX_DIR = ROOT / ".tmp_e" / "intl_h2h_sbx"
OUT_JSON = ROOT / ".tmp_e" / "intl_head_to_head_results.json"

OPPONENTS = [
    ("Qwen2.5-0.5B-Instruct", MODELS_DIR / "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
    ("Llama-3.2-1B-Instruct", MODELS_DIR / "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
    ("Qwen2.5-1.5B-Instruct", MODELS_DIR / "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
]


def _norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFC", str(s)).strip().lower().split())


def _load_jsonl(name: str) -> tuple[list[dict[str, Any]], str]:
    raw = (WORLD_DIR / f"{name}.jsonl").read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    rows = [json.loads(ln) for ln in raw.decode("utf-8").splitlines() if ln.strip()]
    return rows, sha


def _extract_floats(text: str) -> list[float]:
    vals: list[float] = []
    cleaned = text.translate(str.maketrans("=;,()[]{}$%", "           "))
    for raw in cleaned.split():
        tok = raw.strip(".,;:!?\"'`*#")
        if not tok:
            continue
        if tok.count("/") == 1:
            a, b = tok.split("/", 1)
            try:
                fa, fb = float(a), float(b)
                if fb != 0.0 and math.isfinite(fa / fb):
                    vals.append(fa / fb)
                    continue
            except ValueError:
                pass
        nc = tok.replace(",", "") if ("," in tok and all(c.isdigit() or c in ".,-+" for c in tok)) else tok
        try:
            v = float(nc)
            if math.isfinite(v):
                vals.append(v)
        except ValueError:
            pass
    return vals


def _extract_last_boxed(text: str) -> str:
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return ""
    depth = 0
    out: list[str] = []
    for ch in text[idx + 7 :]:
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            if depth == 0:
                return "".join(out).strip()
            depth -= 1
            out.append(ch)
        else:
            out.append(ch)
    return ""


def _norm_math(s: str) -> str:
    t = _norm(s).replace("\\$", "").replace("$", "").replace("\\ ", "").replace(" ", "")
    for pref in ("\\boxed{", "\\text{", "\\mathrm{"):
        if t.startswith(pref) and t.endswith("}"):
            t = t[len(pref) : -1]
    return t.rstrip(".")


def _num_close(a: float, b: float) -> bool:
    return abs(a - b) <= max(1e-4, 1e-4 * abs(b))


def _parse_folio_verdict(reply: str) -> str | None:
    lo = _norm(reply)
    if not lo:
        return None
    abstain_cues = (
        "cannot determine", "cannot be determined", "not enough information",
        "insufficient information", "không đủ dữ kiện", "chưa đủ thông tin",
        "không thể xác định",
    )
    toks = [w.strip(".,:;!?*\"'()[]`") for w in lo.split() if w.strip(".,:;!?*\"'()[]`")]
    # 1. Immediate single-word verdict at very start
    if toks and toks[0] in ("true", "đúng"):
        return "true"
    if toks and toks[0] in ("false", "sai"):
        return "false"
    if toks and toks[0] in ("uncertain", "unknown", "undetermined"):
        return "uncertain"
    # 2. Explicit marker near end
    for mk in ("final answer:", "answer:", "verdict:", "conclusion is", "therefore, the conclusion is", "is therefore"):
        pos = lo.rfind(mk)
        if pos >= 0:
            tail = [w.strip(".,:;!?*\"'()[]`") for w in lo[pos + len(mk) :].strip().split()[:5]]
            for w in tail:
                if w in ("true", "yes", "đúng"):
                    return "true"
                if w in ("false", "no", "sai"):
                    return "false"
                if w in ("uncertain", "unknown", "undetermined"):
                    return "uncertain"
    # 3. Last occurrence of an explicit label word in the final 2 lines
    lines = [_norm(ln) for ln in reply.splitlines() if ln.strip()]
    for ln in reversed(lines[-2:]):
        if any(c in ln for c in abstain_cues):
            return "uncertain"
        lwords = [w.strip(".,:;!?*\"'()[]`") for w in ln.split()]
        for w in reversed(lwords):
            if w in ("uncertain", "unknown", "undetermined"):
                return "uncertain"
            if w in ("true", "đúng"):
                return "true"
            if w in ("false", "sai"):
                return "false"
    # 4. Last occurrence anywhere
    if any(c in lo for c in abstain_cues):
        return "uncertain"
    for w in reversed(toks):
        if w in ("uncertain", "unknown", "undetermined"):
            return "uncertain"
        if w in ("true", "đúng"):
            return "true"
        if w in ("false", "sai"):
            return "false"
    return None


def eval_folio_h2h(adapter: Any, rows: list[dict[str, Any]], limit: int) -> dict[str, Any]:
    sub = rows[:limit]
    ok = tf_ok = tf_tot = unc_ok = unc_tot = unparsed = 0
    lats: list[float] = []
    for i, r in enumerate(sub):
        sid = f"h2h_folio_{i}"
        adapter.reset_session(sid)
        gold = _norm(r["label"])
        if gold == "uncertain":
            unc_tot += 1
        else:
            tf_tot += 1
        prompt = (
            f"Premises:\n{r['premises']}\n\nConclusion: {r['conclusion']}\n"
            "Is the conclusion True, False, or Uncertain?\n"
            "Respond with ONLY one word: True, False, or Uncertain."
        )
        t0 = time.perf_counter()
        try:
            rep = adapter.chat(prompt, sid, max_tokens=48)
        except Exception:
            rep = ""
        lats.append((time.perf_counter() - t0) * 1000.0)
        pred = _parse_folio_verdict(rep)
        if pred is None:
            unparsed += 1
        if pred == gold:
            ok += 1
            if gold == "uncertain":
                unc_ok += 1
            else:
                tf_ok += 1
    n = len(sub)
    return {
        "n": n, "correct": ok, "acc_pct": round(100.0 * ok / max(1, n), 2),
        "tf_correct": tf_ok, "tf_total": tf_tot,
        "unc_correct": unc_ok, "unc_total": unc_tot,
        "unparsed": unparsed,
        "median_ms": round(float(statistics.median(lats)), 2) if lats else 0.0,
    }


def eval_humaneval_h2h(adapter: Any, rows: list[dict[str, Any]], limit: int, tag: str) -> dict[str, Any]:
    sub = rows[:limit]
    ok = syn_ok = 0
    lats: list[float] = []
    ws_base = SBX_DIR / f"he_{tag}"
    shutil.rmtree(ws_base, ignore_errors=True)
    for i, r in enumerate(sub):
        sid = f"h2h_he_{i}"
        adapter.reset_session(sid)
        prompt = f"Write the complete Python implementation for this function:\n\n{r['prompt']}"
        t0 = time.perf_counter()
        try:
            rep = adapter.chat(prompt, sid, max_tokens=260)
        except Exception:
            rep = ""
        lats.append((time.perf_counter() - t0) * 1000.0)
        code = extract_python_block(rep)
        ep = r["entry_point"]
        if code and f"def {ep}" in code:
            full_src = code
        elif code:
            indented = "\n".join(("    " + ln if ln.strip() and not ln.startswith(" ") else ln) for ln in code.splitlines())
            full_src = r["prompt"] + "\n" + indented
        else:
            full_src = ""
        parsed = False
        if full_src:
            try:
                ast.parse(full_src)
                parsed = True
                syn_ok += 1
            except SyntaxError:
                parsed = False
        if parsed:
            drv = f"{full_src}\n\n{r['test']}\ncheck({ep})\nprint('HE_PASS')\n"
            ran, out = _run_python_in_sandbox(drv, ws_base / f"t_{i}", timeout_s=5.0)
            if ran and "HE_PASS" in out:
                ok += 1
    shutil.rmtree(ws_base, ignore_errors=True)
    n = len(sub)
    return {
        "n": n, "pass_at_1": ok, "acc_pct": round(100.0 * ok / max(1, n), 2),
        "syntax_valid": syn_ok,
        "median_ms": round(float(statistics.median(lats)), 2) if lats else 0.0,
    }


def eval_math500_h2h(adapter: Any, rows: list[dict[str, Any]], limit: int) -> dict[str, Any]:
    sub = rows[:limit]
    ok = boxed_cnt = 0
    lats: list[float] = []
    for i, r in enumerate(sub):
        sid = f"h2h_math_{i}"
        adapter.reset_session(sid)
        gold_str = _norm_math(r["answer"])
        gold_nums = _extract_floats(r["answer"])
        t0 = time.perf_counter()
        try:
            rep = adapter.chat(r["problem"], sid, max_tokens=384)
        except Exception:
            rep = ""
        lats.append((time.perf_counter() - t0) * 1000.0)
        boxed = _extract_last_boxed(rep)
        if boxed:
            boxed_cnt += 1
        cand_text = boxed if boxed else (rep.splitlines()[-1] if rep.splitlines() else rep)
        cand_nums = _extract_floats(cand_text) or _extract_floats(rep)
        hit = False
        if boxed and _norm_math(boxed) == gold_str:
            hit = True
        elif len(gold_nums) == 1 and cand_nums and _num_close(cand_nums[-1], gold_nums[0]):
            hit = True
        elif len(gold_str) >= 2 and gold_str in _norm_math(cand_text):
            hit = True
        if hit:
            ok += 1
    n = len(sub)
    return {
        "n": n, "correct": ok, "acc_pct": round(100.0 * ok / max(1, n), 2),
        "boxed_found": boxed_cnt,
        "median_ms": round(float(statistics.median(lats)), 2) if lats else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folio-n", type=int, default=50)
    ap.add_argument("--he-n", type=int, default=30)
    ap.add_argument("--math-n", type=int, default=30)
    ap.add_argument("--run-part-b-genesis", action="store_true", default=False)
    args = ap.parse_args()

    folio_rows, folio_sha = _load_jsonl("folio")
    he_rows, he_sha = _load_jsonl("humaneval")
    math_rows, math_sha = _load_jsonl("math500")

    report: dict[str, Any] = {
        "datasets": {
            "folio": {"source": "tasksource/folio", "n": args.folio_n, "total": len(folio_rows), "sha256": folio_sha[:16]},
            "humaneval": {"source": "openai/openai_humaneval", "n": args.he_n, "total": len(he_rows), "sha256": he_sha[:16]},
            "math500": {"source": "HuggingFaceH4/MATH-500", "n": args.math_n, "total": len(math_rows), "sha256": math_sha[:16]},
        },
        "models": {},
    }

    def _save() -> None:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== [1/4] Project Genesis (Loop 29 Two-Hemisphere CPU, 0% GPU) ===", flush=True)
    eng = GenesisChat3MEngine.load(DEFAULT_CHAT3M_MODEL_DIR, enable_thinking=True, persist_live_writes=False)
    gen_ad = GenesisChatAdapter(eng)
    g_res: dict[str, Any] = {}
    g_res["folio"] = eval_folio_h2h(gen_ad, folio_rows, args.folio_n)
    print(f"  [Genesis] FOLIO: {g_res['folio']}", flush=True)
    g_res["humaneval"] = eval_humaneval_h2h(gen_ad, he_rows, args.he_n, "genesis")
    print(f"  [Genesis] HumanEval: {g_res['humaneval']}", flush=True)
    g_res["math500"] = eval_math500_h2h(gen_ad, math_rows, args.math_n)
    print(f"  [Genesis] MATH-500: {g_res['math500']}", flush=True)
    if args.run_part_b_genesis:
        pub_items = load_all_public_benchmarks(seed=42)
        g_res["part_b_public_120"] = evaluate_part_b_public(gen_ad, pub_items, "genesis_loop29")
        print(f"  [Genesis] Public-120: p@1={g_res['part_b_public_120']['pct_at_1']}% p@3={g_res['part_b_public_120']['pct_at_3']}% | {g_res['part_b_public_120']['by_suite']}", flush=True)
    report["models"]["Project-Genesis-Loop29-CPU"] = g_res
    _save()
    del eng, gen_ad

    for idx, (name, gguf) in enumerate(OPPONENTS, start=2):
        if not gguf.exists():
            print(f"[SKIP] missing {gguf}", flush=True)
            continue
        print(f"\n=== [{idx}/4] {name} ({gguf.stat().st_size / (1024**2):.1f} MiB) ===", flush=True)
        with LlamaCppChatAdapter(SERVER_EXE, gguf, port=18090 + idx) as ad:
            m_res: dict[str, Any] = {"gguf_mb": round(gguf.stat().st_size / (1024**2), 1)}
            m_res["folio"] = eval_folio_h2h(ad, folio_rows, args.folio_n)
            print(f"  [{name}] FOLIO: {m_res['folio']}", flush=True)
            m_res["humaneval"] = eval_humaneval_h2h(ad, he_rows, args.he_n, name.replace(".", "_"))
            print(f"  [{name}] HumanEval: {m_res['humaneval']}", flush=True)
            m_res["math500"] = eval_math500_h2h(ad, math_rows, args.math_n)
            print(f"  [{name}] MATH-500: {m_res['math500']}", flush=True)
            m_res["peak_rss_mb"] = ad.peak_rss_mb
            report["models"][name] = m_res
            _save()

    print(f"\n[DONE] Saved official international head-to-head report -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
