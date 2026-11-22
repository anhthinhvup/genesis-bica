"""Default Mode Network & Wake-Sleep Consolidation Daemon (Spec #91, <= 380 LOC).
SMFT Hard Background-DF Gate + f32 Sparse Delta-Sidecar (GV8DLT01 <= 20 MB)
+ 80/20 Real Interleaved Replay + SEAL/RLVR + Homeostasis Pause (<1s) + Zero-Tradeoff Rollback.
"""
from __future__ import annotations

import array, ctypes, json, math, operator, os, random, shutil, struct, time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable
from ._env_guard import assert_not_c_drive, enforce_e_drive_environment
from .peer_bg_profiler import (
    DEFAULT_BG_DF_BIN, DEFAULT_BPE_BIN, DEFAULT_DLL, DEFAULT_MODEL_BIN, DEFAULT_MODEL_GV8,
    DEFAULT_VAULT_DB, N_EXPERTS_DEFAULT, PeerPkRouter, compute_tau_bg, load_bg_df,
    profile_vault_bg_df, sample_vault_passages, select_smft_experts,
)
from .v6_rlvr_selfplay import DEFAULT_RLVR_OUT, _extract_last_number, _nums_equal
from .v6_unified_policy import _safe_ast_arith
try: from .homeostatic_interoceptor import HomeostaticInteroceptor
except Exception: HomeostaticInteroceptor = None  # type: ignore[misc, assignment]

enforce_e_drive_environment()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIDECAR_BIN, DEFAULT_AUDIT_LOG = PROJECT_ROOT / ".cache_e" / "sleep_peer_delta.bin", PROJECT_ROOT / ".cache_e" / "sleep_audit.jsonl"
SIDECAR_MAGIC, MAX_SIDECAR_BYTES, _MUL = b"GV8DLT01", 20_000_000, operator.mul

def _sigmoid(x: float) -> float: return (1.0 / (1.0 + math.exp(-x))) if x >= 0.0 else (math.exp(x) / (1.0 + math.exp(x)))
def _clamp(v: float, lo: float, hi: float) -> float: return lo if v < lo else (hi if v > hi else v)
def _unit_vec(v: list[float]) -> list[float]:
    inv = 1.0 / math.sqrt(sum(x * x for x in v) + 1e-12); return [x * inv for x in v]


@dataclass
class DeltaRow:
    peer_idx: int; expert_id: int; vu_local: float; vv_local: float; delta_u: list[float]; delta_v: list[float]


class PeerDeltaSidecar:
    """Exact `GV8DLT01` sparse f32 PEER delta sidecar (`<= 20 MB`), matching Rust `peer_delta_sidecar.rs`."""
    def __init__(self, d_model: int = 384, n_peers: int = 2) -> None:
        self.d_model, self.n_peers, self.rows = int(d_model), int(n_peers), {}  # type: ignore[var-annotated]
    @property
    def max_rows(self) -> int: return (MAX_SIDECAR_BYTES - 32) // (16 + 8 * self.d_model)
    def byte_size(self) -> int: return 32 + len(self.rows) * (16 + 8 * self.d_model)
    def clone(self) -> PeerDeltaSidecar:
        cp = PeerDeltaSidecar(self.d_model, self.n_peers)
        for k, r in self.rows.items(): cp.rows[k] = DeltaRow(r.peer_idx, r.expert_id, r.vu_local, r.vv_local, list(r.delta_u), list(r.delta_v))
        return cp
    def ensure_row(self, peer_idx: int, expert_id: int, init_xn: list[float] | None = None) -> DeltaRow | None:
        key = (int(peer_idx), int(expert_id)); row = self.rows.get(key)
        if row is not None: return row
        if len(self.rows) >= self.max_rows: return None
        du = [0.08 * v for v in _unit_vec(init_xn)] if init_xn is not None else [0.0] * self.d_model
        row = DeltaRow(key[0], key[1], 1e-4, 1e-4, du, [0.0] * self.d_model); self.rows[key] = row; return row
    def all_finite(self) -> bool:
        return all(math.isfinite(r.vu_local) and math.isfinite(r.vv_local) and all(math.isfinite(x) for x in r.delta_u) and all(math.isfinite(x) for x in r.delta_v) for r in self.rows.values())
    def serialize(self) -> bytes:
        ordered = sorted(self.rows.values(), key=lambda r: (r.peer_idx, r.expert_id))
        sz = 32 + len(ordered) * (16 + 8 * self.d_model)
        if sz > MAX_SIDECAR_BYTES: raise ValueError(f"Sidecar size {sz} exceeds cap {MAX_SIDECAR_BYTES}")
        chunks = [struct.pack("<8s6I", SIDECAR_MAGIC, 1, self.d_model, self.n_peers, len(ordered), 0, 0)]
        for r in ordered:
            chunks.append(struct.pack("<IIff", r.peer_idx, r.expert_id, float(r.vu_local), float(r.vv_local)))
            chunks.append(array.array("f", r.delta_u).tobytes()); chunks.append(array.array("f", r.delta_v).tobytes())
        return b"".join(chunks)
    def save_atomic(self, path: Path = DEFAULT_SIDECAR_BIN) -> int:
        out_p = assert_not_c_drive(Path(path)); out_p.parent.mkdir(parents=True, exist_ok=True)
        payload, tmp_p = self.serialize(), assert_not_c_drive(out_p.with_suffix(".bin.tmp"))
        with tmp_p.open("wb") as f: f.write(payload); f.flush(); os.fsync(f.fileno())
        os.replace(tmp_p, out_p); return len(payload)
    @classmethod
    def load(cls, path: Path = DEFAULT_SIDECAR_BIN) -> PeerDeltaSidecar:
        raw = Path(path).read_bytes()
        if len(raw) < 32 or len(raw) > 20_971_520: raise ValueError(f"Invalid sidecar size {len(raw)}")
        magic, ver, d_m, n_p, n_r, _, _ = struct.unpack_from("<8s6I", raw, 0)
        if magic != SIDECAR_MAGIC or ver != 1 or len(raw) != 32 + n_r * (16 + 8 * d_m): raise ValueError(f"Corrupt sidecar in {path}")
        sc, off = cls(d_model=d_m, n_peers=n_p), 32
        for _ in range(n_r):
            pi, eid, vu, vv = struct.unpack_from("<IIff", raw, off); off += 16
            du = array.array("f"); du.frombytes(raw[off : off + 4 * d_m]); off += 4 * d_m
            dv = array.array("f"); dv.frombytes(raw[off : off + 4 * d_m]); off += 4 * d_m
            sc.rows[(int(pi), int(eid))] = DeltaRow(int(pi), int(eid), float(vu), float(vv), list(du), list(dv))
        return sc


def _bind_rust_dll(dll_path: Path) -> Any | None:
    if not dll_path.is_file(): return None
    lib, vp, ci, cp = ctypes.CDLL(str(dll_path)), ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p
    for fn, at, rt in (("v6_load_handle", [cp, cp], vp), ("v6_free_handle", [vp], None), ("v6_reset_state", [vp], None),
                       ("v6_vocab_size", [vp], ci), ("v6_encode", [vp, cp, ctypes.POINTER(ctypes.c_uint16), ci], ci),
                       ("v6_feed_token", [vp, ctypes.c_uint16, ctypes.POINTER(ctypes.c_float), ci], ci),
                       ("v6_last_peer_ids", [vp, ci, ctypes.POINTER(ctypes.c_uint32), ci], ci),
                       ("v6_load_peer_delta_sidecar", [vp, cp], ci), ("v6_peer_delta_row_count", [vp], ci),
                       ("v6_clear_peer_delta_sidecar", [vp], ci)):
        if hasattr(lib, fn): f = getattr(lib, fn); f.argtypes, f.restype = at, rt
    return lib


def verify_rust_sidecar_ffi(sidecar_path: Path = DEFAULT_SIDECAR_BIN, model_path: Path | None = None,
                            bpe_path: Path = DEFAULT_BPE_BIN, dll_path: Path = DEFAULT_DLL) -> dict[str, Any]:
    mp, lib = model_path or (DEFAULT_MODEL_GV8 if DEFAULT_MODEL_GV8.is_file() else DEFAULT_MODEL_BIN), _bind_rust_dll(dll_path)
    if lib is None or not (hasattr(lib, "v6_load_peer_delta_sidecar") and mp.is_file() and bpe_path.is_file() and sidecar_path.is_file()):
        return {"available": False, "rows_loaded": -1}
    ptr = lib.v6_load_handle(str(mp).encode("utf-8"), str(bpe_path).encode("utf-8"))
    if not ptr: return {"available": False, "rows_loaded": -1}
    try:
        loaded = int(lib.v6_load_peer_delta_sidecar(ptr, str(sidecar_path).encode("utf-8")))
        count = int(lib.v6_peer_delta_row_count(ptr)); cleared = int(lib.v6_clear_peer_delta_sidecar(ptr))
        return {"available": True, "rows_loaded": loaded, "row_count": count, "cleared_rc": cleared, "after_clear": int(lib.v6_peer_delta_row_count(ptr))}
    finally: lib.v6_free_handle(ptr)


@dataclass
class SleepCycleReport:
    cycle_id: int; status: str; reasons: list[str] = field(default_factory=list)
    n_new: int = 0; n_replay: int = 0; steps_run: int = 0; smft_selected_p0: int = 0; smft_selected_p1: int = 0
    smft_leaked: int = 0; fake_verify: int = 0; val_loss_before: float = 0.0; val_loss_after: float = 0.0
    delta_val_loss: float = 0.0; baseline_before: float = 0.0; baseline_after: float = 0.0; baseline_retention: float = 1.0
    new_packet_acc: float = 0.0; sidecar_rows: int = 0; sidecar_bytes: int = 0; elapsed_ms: float = 0.0; pause_ms: float = 0.0


class GenesisSleepDaemon:
    """DMN Wake-Sleep Consolidator: SMFT Hard-Gate + f32 Sparse Adafactor + 80/20 Replay + Zero-Tradeoff."""
    def __init__(self, bg_df_path: Path = DEFAULT_BG_DF_BIN, sidecar_path: Path = DEFAULT_SIDECAR_BIN,
                 audit_log_path: Path = DEFAULT_AUDIT_LOG, vault_db: Path = DEFAULT_VAULT_DB,
                 model_path: Path = DEFAULT_MODEL_BIN, bpe_path: Path = DEFAULT_BPE_BIN, dll_path: Path = DEFAULT_DLL,
                 interoceptor: Any = None, max_val_loss_delta: float = 0.005, use_rust_router: bool = True) -> None:
        self.bg_df_path, self.sidecar_path = assert_not_c_drive(Path(bg_df_path)), assert_not_c_drive(Path(sidecar_path))
        self.backup_path, self.audit_log_path = assert_not_c_drive(self.sidecar_path.with_suffix(".bin.bak")), assert_not_c_drive(Path(audit_log_path))
        self.vault_db, self.dll_path, self.bpe_path = Path(vault_db), Path(dll_path), Path(bpe_path)
        self.gv8_path = DEFAULT_MODEL_GV8 if DEFAULT_MODEL_GV8.is_file() else Path(model_path)
        self.max_val_loss_delta, self.use_rust_router = float(max_val_loss_delta), bool(use_rust_router)
        self.interoceptor = interoceptor if interoceptor is not None else (HomeostaticInteroceptor() if HomeostaticInteroceptor else None)
        if not self.bg_df_path.is_file():
            profile_vault_bg_df(vault_db=self.vault_db, out_bin=self.bg_df_path, n_docs=2000, model_path=model_path, bpe_path=bpe_path)
        self.df0, self.df1 = load_bg_df(self.bg_df_path); self.tau_bg = (compute_tau_bg(self.df0), compute_tau_bg(self.df1))
        self.router = PeerPkRouter(model_path=model_path, bpe_path=bpe_path); self.d = self.router.d_model
        self._mu_emb = self._compute_emb_mean(); self._route_cache: dict[str, Any] = {}; self._cycle_counter = 0
        self.sidecar = PeerDeltaSidecar.load(self.sidecar_path) if self.sidecar_path.is_file() else PeerDeltaSidecar(self.d, self.router.n_peers)

    def close(self) -> None: self.router.close()
    def _compute_emb_mean(self) -> list[float]:
        d, v, emb = self.d, self.router.vocab, self.router.emb; mu = [0.0] * d
        for t in range(v):
            b = t * d
            for i in range(d): mu[i] += emb[b + i]
        inv = 1.0 / max(1, v); return [m * inv for m in mu]
    def encode_vec(self, text: str) -> list[float]:
        tids = self.router.bpe.encode(text, max_tokens=32); d, mu, emb = self.d, self._mu_emb, self.router.emb; acc = [0.0] * d
        for pos, tid in enumerate(tids):
            b = (tid if 0 <= tid < self.router.vocab else 0) * d; w = 1.0 + 0.04 * (pos / max(1, len(tids)))
            for i in range(d): acc[i] += w * (emb[b + i] - mu[i])
        return _unit_vec(acc)
    def _build_cache_entry(self, prompt: str, raw_rts: list[dict[int, float]]) -> Any:
        x = self.encode_vec(prompt); xns, rts, y_base = [], [], [0.0] * self.d
        for pi in range(self.router.n_peers):
            xn = self.router.rmsnorm(x, self.router.peers[pi][0]); agg = raw_rts[pi]; w_sum = sum(agg.values()) + 1e-12
            rt = [(eid, 4.0 * w / w_sum) for eid, w in agg.items()]; xns.append(xn); rts.append(rt)
            for eid, w in rt[:32]:
                u0, v0 = self.router.read_expert_uv(pi, eid); pre = sum(map(_MUL, u0, xn)); sc = w * pre * _sigmoid(pre)
                for i in range(self.d): y_base[i] += sc * v0[i]
        entry = (x, xns, rts, _unit_vec([x[i] + 0.15 * y_base[i] for i in range(self.d)]))
        if len(self._route_cache) < 4096: self._route_cache[prompt] = entry
        return entry

    def _prefetch_routes(self, prompts: list[str]) -> None:
        missing = [p for p in dict.fromkeys(prompts) if p not in self._route_cache]
        if not missing or not self.use_rust_router: return
        lib = _bind_rust_dll(self.dll_path)
        if lib is None or not (hasattr(lib, "v6_last_peer_ids") and self.gv8_path.is_file()): return
        ptr = lib.v6_load_handle(str(self.gv8_path).encode("utf-8"), str(self.bpe_path).encode("utf-8"))
        if not ptr: return
        vocab = int(lib.v6_vocab_size(ptr)); logits, tbuf, ibuf = (ctypes.c_float * vocab)(), (ctypes.c_uint16 * 24)(), (ctypes.c_uint32 * 64)()
        try:
            for prompt in missing:
                lib.v6_reset_state(ptr); agg: list[dict[int, float]] = [{}, {}]
                n_t = min(20, max(1, int(lib.v6_encode(ptr, prompt[:200].encode("utf-8", "ignore"), tbuf, 20))))
                for i in range(n_t):
                    lib.v6_feed_token(ptr, tbuf[i], logits, vocab)
                    if i >= max(0, n_t - 4):
                        tw = 1.0 + 0.5 * (i - max(0, n_t - 4))
                        for pi in (0, 1):
                            for j in range(max(0, int(lib.v6_last_peer_ids(ptr, pi, ibuf, 64)))):
                                eid = int(ibuf[j])
                                if eid < N_EXPERTS_DEFAULT: agg[pi][eid] = agg[pi].get(eid, 0.0) + tw
                self._build_cache_entry(prompt, agg)
        finally: lib.v6_free_handle(ptr)

    def _route_prompt_cached(self, prompt: str) -> Any:
        hit = self._route_cache.get(prompt)
        if hit is not None: return hit
        tids = self.router.bpe.encode(prompt, max_tokens=20); agg: list[dict[int, float]] = [{}, {}]
        for pi in range(self.router.n_peers):
            for tid in tids:
                for eid, w in self.router.route_token_cached(pi, tid): agg[pi][eid] = agg[pi].get(eid, 0.0) + w
        return self._build_cache_entry(prompt, agg)
    def _peer_delta_vec(self, xns: list[list[float]], rts: list[list[tuple[int, float]]], sc: PeerDeltaSidecar) -> list[float]:
        d, dy = self.d, [0.0] * self.d
        for pi, (xn, rt) in enumerate(zip(xns, rts)):
            for eid, w in rt:
                row = sc.rows.get((pi, eid))
                if row is None: continue
                u0, v0 = self.router.read_expert_uv(pi, eid)
                pre0 = sum(map(_MUL, u0, xn)); act0 = pre0 * _sigmoid(pre0)
                pre1 = pre0 + sum(map(_MUL, row.delta_u, xn)); act1 = pre1 * _sigmoid(pre1)
                da, wa1 = w * (act1 - act0), w * act1
                for i in range(d): dy[i] += da * v0[i] + wa1 * row.delta_v[i]
        return dy
    def predict_vector(self, prompt: str, sidecar: PeerDeltaSidecar | None = None) -> list[float]:
        _, xns, rts, x_base = self._route_prompt_cached(prompt)
        dy = self._peer_delta_vec(xns, rts, self.sidecar if sidecar is None else sidecar)
        return _unit_vec([x_base[i] + dy[i] for i in range(self.d)])

    def evaluate_pairs(self, pairs: list[tuple[str, str]], candidate_targets: list[str] | None = None, sidecar: PeerDeltaSidecar | None = None, beta: float = 10.0) -> tuple[float, float]:
        if not pairs: return 0.0, 1.0
        self._prefetch_routes([p for p, _ in pairs])
        pool = list(dict.fromkeys(candidate_targets if candidate_targets is not None else [t for _, t in pairs]))
        c_vecs = [self.encode_vec(t) for t in pool]; t_idx = {t: i for i, t in enumerate(pool)}; tot_loss, correct = 0.0, 0
        for prompt, target in pairs:
            pred = self.predict_vector(prompt, sidecar=sidecar)
            scores = [beta * sum(map(_MUL, pred, cv)) for cv in c_vecs]
            mx = max(scores); ex = [math.exp(s - mx) for s in scores]; gi = t_idx[target]
            tot_loss += -(scores[gi] - mx - math.log(max(sum(ex), 1e-12)))
            if scores[gi] >= mx - 1e-9: correct += 1
        return tot_loss / len(pairs), correct / len(pairs)

    @staticmethod
    def _episode_to_pair(ep: dict[str, Any]) -> tuple[str, str] | None:
        if "prompt" in ep and "target" in ep:
            p, t = str(ep["prompt"]).strip(), str(ep["target"]).strip(); return (p, t) if (p and t) else None
        ent, attr, txt = str(ep.get("entity", "")).strip(), str(ep.get("attribute", "")).strip(), str(ep.get("text", "")).strip()
        if ent and txt: return (f"{ent} {attr}".strip(), txt)
        if ":" in txt:
            a, b = txt.split(":", 1)
            if a.strip() and b.strip(): return (a.strip(), b.strip())
        ws = txt.split(); return (" ".join(ws[: len(ws) // 2]), " ".join(ws[len(ws) // 2 :])) if len(ws) >= 4 else None

    @staticmethod
    def _check_symbolic_fake_verify(pairs: list[tuple[str, str]]) -> int:
        return sum(1 for p, t in pairs if (sym := _safe_ast_arith(p)) is not None and not _nums_equal(_extract_last_number(t), sym))

    def harvest_episodes(self, episodic_memory: Any = None, min_da: float = 0.7, limit: int = 256, rlvr_path: Path = DEFAULT_RLVR_OUT, enable_rem_seal: bool = True) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if episodic_memory is not None and hasattr(episodic_memory, "export_high_da_episodes"):
            out.extend(episodic_memory.export_high_da_episodes(min_da=min_da, limit=limit))
        if Path(rlvr_path).is_file():
            for ln in Path(rlvr_path).read_text(encoding="utf-8", errors="ignore").splitlines()[-32:]:
                if "<USER>" in ln and "<BOT>" in ln:
                    q, a = ln.split("<USER>", 1)[1].split("<BOT>", 1)[0].strip(), ln.split("<BOT>", 1)[1].replace("<EOS>", "").strip()
                    if q and a: out.append({"prompt": q, "target": a, "da_reward": 0.95})
        if enable_rem_seal and not out:
            for a, b in ((17, 23), (31, 19), (42, 18), (64, 27)):
                expr = f"{a} + {b}"; ans = _safe_ast_arith(expr)
                if ans is not None: out.append({"prompt": expr, "target": ans, "da_reward": 1.0})
        return out

    def _sleep_ok(self, idle_sec_fn: Callable[[], float] | None, idle_sec: float, min_idle_sec: float, max_cpu_pct: float) -> bool:
        return True if self.interoceptor is None else bool(self.interoceptor.sleep_allowed(idle_sec=float(idle_sec_fn() if idle_sec_fn else idle_sec), min_idle_sec=min_idle_sec, max_cpu_pct=max_cpu_pct))
    def _append_audit(self, rep: SleepCycleReport) -> None:
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True); rec = asdict(rep); rec["ts"] = round(time.time(), 3)
        with self.audit_log_path.open("a", encoding="utf-8") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    def _backup_current(self) -> None:
        if not self.sidecar_path.is_file(): self.sidecar.save_atomic(self.sidecar_path)
        tmp_b = assert_not_c_drive(self.backup_path.with_suffix(".bak.tmp")); shutil.copy2(self.sidecar_path, tmp_b); os.replace(tmp_b, self.backup_path)
    def _rollback_from_backup(self, snapshot: PeerDeltaSidecar) -> None:
        self.sidecar = snapshot
        if self.backup_path.is_file():
            tmp_r = assert_not_c_drive(self.sidecar_path.with_suffix(".bin.rb")); shutil.copy2(self.backup_path, tmp_r); os.replace(tmp_r, self.sidecar_path)
        else: self.sidecar.save_atomic(self.sidecar_path)
    def _select_cycle_smft(self, new_cache: list[Any], k_sleep: int, per_item_k: int = 12) -> tuple[set[int], set[int]]:
        sel: list[set[int]] = [set(), set()]; dfs = (self.df0, self.df1)
        for (_, _, rts, _), _ in new_cache:
            for pi in range(self.router.n_peers):
                for eid, _ in select_smft_experts({e: w for e, w in rts[pi]}, dfs[pi], self.tau_bg[pi], per_item_k): sel[pi].add(eid)
        for pi in range(self.router.n_peers):
            if len(sel[pi]) > k_sleep:
                tf_all: dict[int, float] = {}
                for (_, _, rts, _), _ in new_cache:
                    for eid, w in rts[pi]:
                        if eid in sel[pi]: tf_all[eid] = tf_all.get(eid, 0.0) + w
                sel[pi] = {e for e, _ in select_smft_experts(tf_all, dfs[pi], self.tau_bg[pi], k_sleep)}
        return sel[0], sel[1]
    def _finish_report(self, rep: SleepCycleReport, t0: float) -> SleepCycleReport:
        rep.sidecar_rows, rep.sidecar_bytes = len(self.sidecar.rows), self.sidecar.byte_size()
        rep.elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2); self._append_audit(rep); return rep

    def run_consolidation_cycle(
        self, new_episodes: list[dict[str, Any]] | None = None, episodic_memory: Any = None,
        held_out_val: list[tuple[str, str]] | None = None, baseline_tasks: list[tuple[str, str]] | None = None,
        steps: int = 60, k_sleep: int = 320, lr: float = 0.015, wd: float = 1e-4, replay_weight: float = 3.0,
        idle_sec: float = 900.0, min_idle_sec: float = 600.0, idle_sec_fn: Callable[[], float] | None = None,
        enable_rem_seal: bool = False, seed: int = 42,
    ) -> SleepCycleReport:
        t0 = time.perf_counter(); self._cycle_counter += 1; cid = self._cycle_counter; rep = SleepCycleReport(cycle_id=cid, status="INIT")
        if not self._sleep_ok(idle_sec_fn, idle_sec, min_idle_sec, max_cpu_pct=15.0):
            rep.status, rep.reasons = "PAUSED_HOMEOSTASIS", ["PRE_CHECK_DENIED"]
            rep.pause_ms = round((time.perf_counter() - t0) * 1000.0, 2); return self._finish_report(rep, t0)
        raw_eps = list(new_episodes) if new_episodes is not None else self.harvest_episodes(episodic_memory=episodic_memory, enable_rem_seal=enable_rem_seal)
        new_pairs = [p for ep in raw_eps if float(ep.get("da_reward", 1.0)) >= 0.7 and (p := self._episode_to_pair(ep)) is not None]
        if not new_pairs:
            rep.status, rep.reasons = "NO_NEW_EPISODES", ["EMPTY_HIGH_DA_STREAM"]; return self._finish_report(rep, t0)
        rep.n_new, rep.fake_verify = len(new_pairs), self._check_symbolic_fake_verify(new_pairs)
        rep_pairs = [pair for txt in sample_vault_passages(self.vault_db, n_docs=max(4 * len(new_pairs), 80), offset=5000 + cid * 97) if (pair := self._episode_to_pair({"text": txt})) is not None]
        rep.n_replay = len(rep_pairs)
        if held_out_val is None:
            held_out_val = [pair for txt in sample_vault_passages(self.vault_db, n_docs=32, offset=150_000) if (pair := self._episode_to_pair({"text": txt})) is not None]
        base_tasks = baseline_tasks if baseline_tasks is not None else held_out_val[:16]
        self._prefetch_routes([p for p, _ in new_pairs] + [p for p, _ in rep_pairs] + [p for p, _ in held_out_val] + [p for p, _ in base_tasks])
        self._backup_current(); snapshot = self.sidecar.clone()
        rep.val_loss_before, _ = self.evaluate_pairs(held_out_val, sidecar=snapshot)
        _, rep.baseline_before = self.evaluate_pairs(base_tasks, sidecar=snapshot)
        if rep.fake_verify > 0:
            self._rollback_from_backup(snapshot); rep.status, rep.reasons = "ROLLED_BACK", ["FAKE_VERIFY_DETECTED"]
            rep.val_loss_after, rep.baseline_after = rep.val_loss_before, rep.baseline_before; return self._finish_report(rep, t0)
        new_cache = [(self._route_prompt_cached(p), self.encode_vec(t)) for p, t in new_pairs]
        rep_cache = [self._route_prompt_cached(p) for p, _ in rep_pairs]; all_y = [y for _, y in new_cache]
        allowed = self._select_cycle_smft(new_cache, k_sleep=k_sleep); rep.smft_selected_p0, rep.smft_selected_p1 = len(allowed[0]), len(allowed[1])
        cand, rng, d, inv_d = snapshot.clone(), random.Random(seed + cid * 131), self.d, 1.0 / self.d
        b_new, b_rep, idxs, n_st = min(5, len(new_cache)), min(20, len(rep_cache)), list(range(len(new_cache))), max(1, int(steps))
        for st in range(n_st):
            t_chk = time.perf_counter()
            if not self._sleep_ok(idle_sec_fn, idle_sec, min_idle_sec, max_cpu_pct=25.0):
                self._rollback_from_backup(snapshot); rep.status, rep.reasons, rep.steps_run = "PAUSED_HOMEOSTASIS", ["MID_BATCH_ABORT"], st
                rep.pause_ms = round((time.perf_counter() - t_chk) * 1000.0, 2); return self._finish_report(rep, t0)
            grads: dict[tuple[int, int], tuple[list[float], list[float]]] = {}
            for idx in rng.sample(idxs, b_new):
                (_, xns, rts, x_base), y_tgt = new_cache[idx]
                dy = self._peer_delta_vec(xns, rts, cand); pred = [x_base[i] + dy[i] for i in range(d)]; pred_u = _unit_vec(pred)
                sc_all = [10.0 * sum(map(_MUL, pred_u, yv)) for yv in all_y]; mx = max(sc_all)
                ex = [math.exp(s - mx) for s in sc_all]; inv_z = 1.0 / max(sum(ex), 1e-12)
                gy = [0.4 * (pred[i] - y_tgt[i]) - 2.0 * y_tgt[i] for i in range(d)]
                for j, ej in enumerate(ex):
                    pj = 2.0 * ej * inv_z; yj = all_y[j]
                    for i in range(d): gy[i] += pj * yj[i]
                self._accum_grad(xns, rts, gy, 1.0 / b_new, allowed, cand, grads, alloc=True)
            if b_rep > 0:
                for _, xns, rts, _ in rng.sample(rep_cache, b_rep):
                    dy = self._peer_delta_vec(xns, rts, cand)
                    if any(abs(v) > 1e-9 for v in dy): self._accum_grad(xns, rts, dy, replay_weight / b_rep, allowed, cand, grads, alloc=False)
            lr_t = lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * st / n_st)))
            for key, (gu, gv) in grads.items():
                row = cand.rows[key]
                row.vu_local = 0.98 * row.vu_local + 0.02 * (sum(x * x for x in gu) * inv_d)
                row.vv_local = 0.98 * row.vv_local + 0.02 * (sum(x * x for x in gv) * inv_d)
                su, sv, dec = (0.18 * lr_t) / (math.sqrt(row.vu_local) + 1e-5), lr_t / (math.sqrt(row.vv_local) + 1e-5), 1.0 - lr_t * wd
                for i in range(d):
                    row.delta_u[i] = _clamp(row.delta_u[i] * dec - su * _clamp(gu[i], -2.0, 2.0), -0.12, 0.12)
                    row.delta_v[i] = _clamp(row.delta_v[i] * dec - sv * _clamp(gv[i], -2.0, 2.0), -0.35, 0.35)
            rep.steps_run = st + 1
        rep.smft_leaked = sum(1 for (pi, e) in cand.rows if (self.df0[e] > self.tau_bg[0] if pi == 0 else self.df1[e] > self.tau_bg[1]))
        rep.val_loss_after, _ = self.evaluate_pairs(held_out_val, sidecar=cand)
        _, rep.baseline_after = self.evaluate_pairs(base_tasks, sidecar=cand); _, rep.new_packet_acc = self.evaluate_pairs(new_pairs, sidecar=cand)
        rep.delta_val_loss = round(rep.val_loss_after - rep.val_loss_before, 6)
        rep.baseline_retention = round(rep.baseline_after / max(rep.baseline_before, 1e-9), 4) if rep.baseline_before > 0 else 1.0
        reasons = [r for cond, r in ((rep.smft_leaked > 0, "SMFT_LEAKAGE"), (not cand.all_finite(), "NON_FINITE_DELTA"),
                                     (cand.byte_size() > MAX_SIDECAR_BYTES, "SIZE_CAP_EXCEEDED"),
                                     (rep.delta_val_loss > self.max_val_loss_delta, "VAL_LOSS_DEGRADED"),
                                     (rep.baseline_after + 1e-9 < rep.baseline_before or rep.baseline_retention < 0.98, "BASELINE_REGRESSION")) if cond]
        if reasons: self._rollback_from_backup(snapshot); rep.status, rep.reasons = "ROLLED_BACK", reasons
        else: self.sidecar = cand; self.sidecar.save_atomic(self.sidecar_path); rep.status = "COMMITTED"
        return self._finish_report(rep, t0)

    def _accum_grad(self, xns: list[list[float]], rts: list[list[tuple[int, float]]], grad_y: list[float], scale: float,
                    allowed: tuple[set[int], set[int]], sc: PeerDeltaSidecar, grads: dict[tuple[int, int], tuple[list[float], list[float]]], alloc: bool) -> None:
        d = self.d
        for pi, (xn, rt) in enumerate(zip(xns, rts)):
            for eid, w in rt:
                if eid not in allowed[pi]: continue
                row = sc.ensure_row(pi, eid, init_xn=xn) if alloc else sc.rows.get((pi, eid))
                if row is None: continue
                u0, v0 = self.router.read_expert_uv(pi, eid)
                pre = sum(map(_MUL, u0, xn)) + sum(map(_MUL, row.delta_u, xn))
                s = _sigmoid(pre); act = pre * s; act_bwd = s * (1.0 + pre * (1.0 - s))
                gv_dot = sum((v0[i] + row.delta_v[i]) * grad_y[i] for i in range(d))
                d_pre, w_act = scale * w * act_bwd * gv_dot, scale * w * act
                gu, gv = grads.setdefault((pi, eid), ([0.0] * d, [0.0] * d))
                for i in range(d): gu[i] += d_pre * xn[i]; gv[i] += w_act * grad_y[i]
