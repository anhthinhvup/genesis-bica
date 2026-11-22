"""PEER Background-DF Profiler & SMFT Hard Background-DF Gate (Spec #91, <= 220 LOC)."""
from __future__ import annotations

import argparse, array, ctypes, heapq, json, math, operator, os, sqlite3, struct, time
from pathlib import Path
from typing import Any
from ._env_guard import DEFAULT_CHAT3M_MODEL_DIR, assert_not_c_drive, enforce_e_drive_environment

enforce_e_drive_environment()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VAULT_DB, DEFAULT_BG_DF_BIN = PROJECT_ROOT / ".cache_e" / "knowledge_vault.sqlite", PROJECT_ROOT / ".cache_e" / "peer_bg_df.bin"
DEFAULT_MODEL_GV8 = DEFAULT_CHAT3M_MODEL_DIR / "v6_lm" / "v7_ultra_p1_preview.gv8"
DEFAULT_MODEL_BIN = DEFAULT_CHAT3M_MODEL_DIR / "v6_lm" / "v7_ultra_p1_preview.bin"
DEFAULT_BPE_BIN, DEFAULT_DLL = DEFAULT_CHAT3M_MODEL_DIR / "v6_lm" / "v6_bpe.bin", DEFAULT_CHAT3M_MODEL_DIR / "bin" / "genesis_v6.dll"
N_PEERS_DEFAULT, N_EXPERTS_DEFAULT, Q14_INV, _MUL = 2, 512 * 512, 1.0 / 16384.0, operator.mul


class ByteBpeLite:
    def __init__(self, bpe_path: Path = DEFAULT_BPE_BIN) -> None:
        with Path(bpe_path).open("rb") as f:
            if f.read(8) != b"GV6BPE01": raise ValueError(f"Invalid BPE magic: {bpe_path}")
            n_m = struct.unpack("<I", f.read(4))[0]; raw = f.read(n_m * 4)
        self.merge_rank = {struct.unpack_from("<HH", raw, r * 4): (r, 272 + r) for r in range(n_m)}

    def encode(self, text: str, max_tokens: int = 48) -> list[int]:
        raw = text.encode("utf-8", errors="ignore")[: max_tokens * 6]
        out: list[int] = []; st = 0; mr = self.merge_rank
        for i, ch in enumerate(raw):
            if (ch == 32 or ch == 10) and i > st:
                out.extend(self._enc_chunk(raw[st:i], mr))
                if len(out) >= max_tokens: return out[:max_tokens]
                st = i
        if st < len(raw): out.extend(self._enc_chunk(raw[st:], mr))
        return out[:max_tokens] or [1]

    @staticmethod
    def _enc_chunk(chunk: bytes, mr: dict[tuple[int, int], tuple[int, int]]) -> list[int]:
        seq = [16 + b for b in chunk]
        while len(seq) >= 2:
            br, bp, bi = 10**9, -1, 0
            for i in range(len(seq) - 1):
                hit = mr.get((seq[i], seq[i + 1]))
                if hit is not None and hit[0] < br: br, bp, bi = hit[0], i, hit[1]
            if bp < 0: break
            seq[bp] = bi; del seq[bp + 1]
        return seq


class PeerPkRouter:
    """Exact Product-Key router reading real `emb` & `PEER_0/1` keys from `GV6MOD03` (<15 MB RAM)."""
    def __init__(self, model_path: Path = DEFAULT_MODEL_BIN, bpe_path: Path = DEFAULT_BPE_BIN) -> None:
        self.model_path, self.bpe = Path(model_path), ByteBpeLite(bpe_path)
        self._fh = self.model_path.open("rb")
        if self._fh.read(8) != b"GV6MOD03": raise ValueError(f"Unsupported model magic in {model_path}")
        hdr = struct.unpack("<11I", self._fh.read(44))
        self.vocab, self.d_model, self.n_layers, n_moe, _, d_ff, self.sub_k, self.n_heads, self.k_head, _, _ = hdr
        d, sk, ph = self.d_model, self.sub_k, self.n_heads
        self.sd, self.n_experts = (d // ph) // 2, sk * sk
        self.n_peers = 2 if (sk > 0 and self.n_layers >= 4) else (1 if sk > 0 else 0)
        self.emb = array.array("f"); self.emb.frombytes(self._fh.read(self.vocab * d * 4))
        self.norm_f = list(array.array("f", self._fh.read(d * 4)))
        self._fh.seek((d + self.n_layers * (5 * d + 3 * d * d + n_moe * d + n_moe * 2 * d_ff * d)) * 4, 1)
        self.peers: list[tuple[list[float], list[list[float]], list[list[list[float]]], list[list[list[float]]]]] = []
        self._uv_offs: list[tuple[int, int]] = []
        for _ in range(self.n_peers):
            norm = list(array.array("f", self._fh.read(d * 4)))
            wq_a = array.array("f"); wq_a.frombytes(self._fh.read(d * d * 4))
            ka_a = array.array("f"); ka_a.frombytes(self._fh.read(ph * sk * self.sd * 4))
            kb_a = array.array("f"); kb_a.frombytes(self._fh.read(ph * sk * self.sd * 4))
            self._fh.seek(2 * self.n_experts * 4, 1)
            u_off = self._fh.tell(); v_off = u_off + self.n_experts * d * 2
            self._fh.seek(2 * self.n_experts * d * 2, 1); self._uv_offs.append((u_off, v_off))
            self.peers.append((norm, [list(wq_a[r * d : (r + 1) * d]) for r in range(d)],
                [[list(ka_a[(h * sk + i) * self.sd : (h * sk + i + 1) * self.sd]) for i in range(sk)] for h in range(ph)],
                [[list(kb_a[(h * sk + i) * self.sd : (h * sk + i + 1) * self.sd]) for i in range(sk)] for h in range(ph)]))
        self._tid_cache: list[dict[int, list[tuple[int, float]]]] = [{} for _ in range(self.n_peers)]
        self._uv_cache: dict[tuple[int, int], tuple[list[float], list[float]]] = {}
    def close(self) -> None:
        if getattr(self, "_fh", None) and not self._fh.closed: self._fh.close()
    def embed_token(self, tid: int) -> list[float]:
        d, t = self.d_model, (tid if 0 <= tid < self.vocab else 0)
        return list(self.emb[t * d : (t + 1) * d])
    def rmsnorm(self, x: list[float], w: list[float]) -> list[float]:
        d = self.d_model; inv = 1.0 / math.sqrt(sum(v * v for v in x) / d + 1e-5)
        return [x[i] * inv * w[i] for i in range(d)]

    def route_vector(self, peer_idx: int, x: list[float]) -> list[tuple[int, float]]:
        norm, wq, ka, kb = self.peers[peer_idx]
        xn = self.rmsnorm(x, norm); q = [sum(map(_MUL, row, xn)) for row in wq]
        hd, sd, sk, kh = self.d_model // self.n_heads, self.sd, self.sub_k, self.k_head
        top_m, out = min(kh, sk, 10), []
        for h in range(self.n_heads):
            qa, qb = q[h * hd : h * hd + sd], q[h * hd + sd : (h + 1) * hd]
            ta = heapq.nlargest(top_m, ((sum(map(_MUL, ka[h][i], qa)), i) for i in range(sk)))
            tb = heapq.nlargest(top_m, ((sum(map(_MUL, kb[h][i], qb)), i) for i in range(sk)))
            pairs = heapq.nlargest(kh, ((va + vb, ia * sk + ib) for va, ia in ta for vb, ib in tb))
            mx = pairs[0][0]; ex = [math.exp(sc - mx) for sc, _ in pairs]; inv = 1.0 / max(sum(ex), 1e-12)
            out.extend((eid, e_w * inv) for (_, eid), e_w in zip(pairs, ex))
        return out

    def route_token_cached(self, peer_idx: int, tid: int) -> list[tuple[int, float]]:
        hit = self._tid_cache[peer_idx].get(tid)
        if hit is None: hit = self.route_vector(peer_idx, self.embed_token(tid)); self._tid_cache[peer_idx][tid] = hit
        return hit

    def read_expert_uv(self, peer_idx: int, eid: int) -> tuple[list[float], list[float]]:
        key = (peer_idx, eid); hit = self._uv_cache.get(key)
        if hit is not None: return hit
        d, (u_off, v_off) = self.d_model, self._uv_offs[peer_idx]
        self._fh.seek(u_off + eid * d * 2); ur = array.array("h"); ur.frombytes(self._fh.read(d * 2))
        self._fh.seek(v_off + eid * d * 2); vr = array.array("h"); vr.frombytes(self._fh.read(d * 2))
        pair = ([q * Q14_INV for q in ur], [q * Q14_INV for q in vr])
        if len(self._uv_cache) < 16384: self._uv_cache[key] = pair
        return pair


def sample_vault_passages(vault_db: Path = DEFAULT_VAULT_DB, n_docs: int = 10000, offset: int = 0) -> list[str]:
    conn = sqlite3.connect(str(vault_db))
    try:
        total = int(conn.execute("SELECT COUNT(*) FROM docs;").fetchone()[0])
        stride = max(1, (total - offset) // max(1, n_docs))
        rows = conn.execute("SELECT body FROM docs WHERE rowid > ? AND ((rowid - ?) % ?) = 0 LIMIT ?;", (offset, offset, stride, n_docs)).fetchall()
        return [str(r[0]) for r in rows if r and r[0]]
    finally: conn.close()


def _profile_via_rust_ffi(passages: list[str], dll_path: Path, model_path: Path, bpe_path: Path, tokens_per_doc: int) -> tuple[array.array, array.array] | None:
    if not (dll_path.is_file() and model_path.is_file() and bpe_path.is_file()): return None
    lib = ctypes.CDLL(str(dll_path))
    if not hasattr(lib, "v6_last_peer_ids"): return None
    vp, ci, cp = ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p
    for fn, at, rt in (
        ("v6_load_handle", [cp, cp], vp), ("v6_free_handle", [vp], None), ("v6_reset_state", [vp], None),
        ("v6_vocab_size", [vp], ci), ("v6_encode", [vp, cp, ctypes.POINTER(ctypes.c_uint16), ci], ci),
        ("v6_feed_token", [vp, ctypes.c_uint16, ctypes.POINTER(ctypes.c_float), ci], ci),
        ("v6_last_peer_ids", [vp, ci, ctypes.POINTER(ctypes.c_uint32), ci], ci),
    ):
        f = getattr(lib, fn); f.argtypes, f.restype = at, rt
    ptr = lib.v6_load_handle(str(model_path).encode("utf-8"), str(bpe_path).encode("utf-8"))
    if not ptr: return None
    df0, df1 = array.array("I", [0]) * N_EXPERTS_DEFAULT, array.array("I", [0]) * N_EXPERTS_DEFAULT
    vocab = int(lib.v6_vocab_size(ptr)); logits, tbuf, ibuf = (ctypes.c_float * vocab)(), (ctypes.c_uint16 * max(8, tokens_per_doc))(), (ctypes.c_uint32 * 64)()
    try:
        for text in passages:
            lib.v6_reset_state(ptr); s0: set[int] = set(); s1: set[int] = set()
            n_t = min(tokens_per_doc, max(0, int(lib.v6_encode(ptr, text[:240].encode("utf-8", "ignore"), tbuf, tokens_per_doc))))
            for i in range(n_t):
                lib.v6_feed_token(ptr, tbuf[i], logits, vocab)
                for pi, seen in ((0, s0), (1, s1)):
                    for j in range(max(0, int(lib.v6_last_peer_ids(ptr, pi, ibuf, 64)))):
                        if ibuf[j] < N_EXPERTS_DEFAULT: seen.add(int(ibuf[j]))
            for e in s0: df0[e] += 1
            for e in s1: df1[e] += 1
    finally: lib.v6_free_handle(ptr)
    return df0, df1


def save_bg_df(path: Path, df0: array.array, df1: array.array) -> Path:
    out_p = assert_not_c_drive(Path(path)); out_p.parent.mkdir(parents=True, exist_ok=True)
    tmp_p = assert_not_c_drive(out_p.with_suffix(".bin.tmp"))
    with tmp_p.open("wb") as f: f.write(df0.tobytes()); f.write(df1.tobytes()); f.flush(); os.fsync(f.fileno())
    os.replace(tmp_p, out_p); return out_p

def load_bg_df(path: Path = DEFAULT_BG_DF_BIN) -> tuple[array.array, array.array]:
    raw, half = Path(path).read_bytes(), N_EXPERTS_DEFAULT * 4
    if len(raw) != 2 * half: raise ValueError(f"Expected {2 * half} bytes in {path}, got {len(raw)}")
    df0, df1 = array.array("I"), array.array("I"); df0.frombytes(raw[:half]); df1.frombytes(raw[half:])
    return df0, df1

def compute_tau_bg(df_arr: array.array, percentile: float = 80.0) -> int:
    p, srt = max(0.0, min(100.0, percentile)) / 100.0, sorted(df_arr)
    tau_all = int(srt[min(len(srt) - 1, int(p * len(srt)))])
    if tau_all >= 2: return tau_all
    nz = [v for v in srt if v > 0]
    return int(nz[min(len(nz) - 1, int(p * len(nz)))]) if nz else 0

def select_smft_experts(tf_episode: dict[int, float], df_bg: array.array, tau_bg: int, top_k: int) -> list[tuple[int, float]]:
    scored = sorted(((tf / (df_bg[e] + 1.0), e) for e, tf in tf_episode.items() if 0 <= e < len(df_bg) and df_bg[e] <= tau_bg and tf > 0.0), reverse=True)
    return [(e, lift) for lift, e in scored[: max(0, top_k)]]

def compare_smft_vs_tfidf(tf_episode: dict[int, float], df_bg: array.array, tau_bg: int, top_k: int = 200, n_docs: int = 10000, shared_top_n: int = 200) -> dict[str, int]:
    shared = {e for _, e in heapq.nlargest(shared_top_n, ((df_bg[i], i) for i in range(len(df_bg)))) if df_bg[e] > tau_bg}
    smft_ids = {e for e, _ in select_smft_experts(tf_episode, df_bg, tau_bg, top_k)}
    tfidf_r = sorted(((tf * math.log((n_docs + 1.0) / (df_bg[e] + 1.0)), e) for e, tf in tf_episode.items() if 0 <= e < len(df_bg) and tf > 0.0), reverse=True)
    return {"shared_total": len(shared), "smft_leaked": len(smft_ids & shared), "tfidf_leaked": len({e for _, e in tfidf_r[: max(0, top_k)]} & shared), "tau_bg": int(tau_bg)}


def profile_vault_bg_df(
    vault_db: Path = DEFAULT_VAULT_DB, out_bin: Path = DEFAULT_BG_DF_BIN, n_docs: int = 10000, tokens_per_doc: int = 8,
    mode: str = "auto", model_path: Path | None = None, bpe_path: Path = DEFAULT_BPE_BIN, dll_path: Path = DEFAULT_DLL,
) -> dict[str, Any]:
    t0, passages, used_mode, res = time.perf_counter(), sample_vault_passages(vault_db, n_docs=n_docs), "pk_weights", None
    if mode in ("auto", "rust_ffi"):
        res = _profile_via_rust_ffi(passages, dll_path, model_path or (DEFAULT_MODEL_GV8 if DEFAULT_MODEL_GV8.is_file() else DEFAULT_MODEL_BIN), bpe_path, tokens_per_doc)
        if res is not None: used_mode = "rust_ffi"
    if res is None:
        router = PeerPkRouter(model_path or DEFAULT_MODEL_BIN, bpe_path)
        df0, df1 = array.array("I", [0]) * N_EXPERTS_DEFAULT, array.array("I", [0]) * N_EXPERTS_DEFAULT
        try:
            for text in passages:
                s0: set[int] = set(); s1: set[int] = set()
                for tid in router.bpe.encode(text, max_tokens=tokens_per_doc):
                    s0.update(e for e, _ in router.route_token_cached(0, tid)); s1.update(e for e, _ in router.route_token_cached(1, tid))
                for e in s0: df0[e] += 1
                for e in s1: df1[e] += 1
        finally: router.close()
        res = (df0, df1)
    saved = save_bg_df(out_bin, res[0], res[1])
    return {"path": str(saved), "size_bytes": saved.stat().st_size, "n_docs": len(passages), "mode": used_mode,
            "tau_bg_p0": compute_tau_bg(res[0]), "tau_bg_p1": compute_tau_bg(res[1]),
            "active_p0": sum(1 for v in res[0] if v > 0), "active_p1": sum(1 for v in res[1] if v > 0), "elapsed_s": round(time.perf_counter() - t0, 2)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Genesis-v8 PEER Background-DF Profiler")
    ap.add_argument("--n-docs", type=int, default=10000); ap.add_argument("--tokens-per-doc", type=int, default=8)
    ap.add_argument("--mode", default="auto", choices=("auto", "rust_ffi", "pk_weights")); ap.add_argument("--out", type=Path, default=DEFAULT_BG_DF_BIN)
    a = ap.parse_args(); print(json.dumps(profile_vault_bg_df(out_bin=a.out, n_docs=a.n_docs, tokens_per_doc=a.tokens_per_doc, mode=a.mode)))
