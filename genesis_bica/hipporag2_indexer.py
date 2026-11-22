"""
Genesis-v8 Offline CPU HippoRAG-2 Bipartite Graph Indexer (`mel.v6_lm.hipporag2_indexer`).
Scans all 265,500 docs in Drive E:\\ `knowledge_vault.sqlite` with a zero-regex pure-CPU
proper-noun, canonical-title, wiki-anchor & technical-code extractor, filters ultra-hubs
(`df > 5000`), and writes aligned dual-CSR `hipporag2_bipartite.bin` (< 60 MB RAM).
Strictly <= 300 LOC | 0% `import re` | 0% neural/API | 0% hardcoded NL containers | 0 B on C:\\.
"""
from __future__ import annotations

import argparse
from array import array
from concurrent.futures import ProcessPoolExecutor
import hashlib, os, sqlite3, struct, time, unicodedata
from pathlib import Path
import numpy as np

from ._env_guard import assert_not_c_drive, enforce_e_drive_environment

enforce_e_drive_environment()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VAULT_DB = PROJECT_ROOT / ".cache_e" / "knowledge_vault.sqlite"
DEFAULT_INDEX_BIN = PROJECT_ROOT / ".cache_e" / "hipporag2_bipartite.bin"
DEFAULT_WSD_DB = PROJECT_ROOT / "models" / "genesis_chat_3m" / "v4_layers" / "concept_wsd.sqlite"

MAGIC, FORMAT_VERSION, HEADER_FMT = b"HP2BIP01", 1, "<8s14I"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
_STRIP = ".,;:!?\"'()[]{}<>|/*+=#@&$%^~`-–—“”«»_"
_ENDS = frozenset(".!?;:|)}],=>")
_TITLE_TF_SENTINEL = 1000


def hash_phrase64(phrase: str) -> int:
    """Deterministic 64-bit BLAKE2b hash for normalized lowercase NFC phrases."""
    return int.from_bytes(hashlib.blake2b(phrase.encode("utf-8"), digest_size=8).digest(), "little", signed=False)


def normalize_phrase(raw: str) -> str:
    s = unicodedata.normalize("NFC", raw).strip().lower()
    return " ".join(s.split()) if "  " in s else s


def _is_file_or_url(tok: str) -> bool:
    if "://" in tok or tok.startswith("www."):
        return True
    dot = tok.rfind(".")
    return dot > 0 and 2 <= len(tok) - dot - 1 <= 4 and tok[dot + 1 :].isascii() and tok[dot + 1 :].isalpha()


def _flush_span(span: list[str], at_start: bool, counts: dict[str, int], title_set: set[str]) -> None:
    sln = len(span)
    if sln >= 2:
        ph = " ".join(span).lower()
        if len(ph) >= 4:
            counts[ph] = counts.get(ph, 0) + 1
        if sln >= 3:
            sub = " ".join(span[1:]).lower()
            if sub in title_set:
                counts[sub] = counts.get(sub, 0) + 1
    elif sln == 1:
        w0 = span[0]
        if (not at_start and len(w0) >= 3) or (2 <= len(w0) <= 8 and w0.isupper()):
            counts[w0.lower()] = counts.get(w0.lower(), 0) + 1
    span.clear()


def _load_title_sets(con: sqlite3.Connection) -> tuple[set[str], set[str], set[str]]:
    title_set, t2w, f2 = set(), set(), set()
    for (tit,) in con.execute("SELECT c1 FROM docs_content WHERE c1 != ''"):
        t = tit.strip().lower()
        if not t:
            continue
        pi = t.find(" (")
        for c in ((t, t[:pi].strip()) if pi > 1 else (t,)):
            if 2 <= len(c) <= 64 and not c.isdigit():
                title_set.add(c)
                if " " in c:
                    ws = c.split()
                    if len(ws) == 2:
                        t2w.add(c); f2.add(ws[0])
    return title_set, t2w, f2


def _extract_shard(args: tuple[str, int, int]) -> tuple[bytes, bytes, bytes, bytes, bytes, bytes]:
    """Worker scanning a disjoint `rowid` shard of `knowledge_vault.sqlite` (read-only)."""
    enforce_e_drive_environment()
    vault_str, lo_id, hi_id = args
    con = sqlite3.connect(f"file:{vault_str}?mode=ro", uri=True)
    title_set, t2w, f2 = _load_title_sets(con)
    doc_ids, p_flags, raw_p, raw_e, raw_tf = array("I"), array("B"), array("I"), array("I"), array("H")
    p2id: dict[str, int] = {}
    id2ph: list[str] = []
    title_lids: set[int] = set()
    strip_c, ends, p_local = _STRIP, _ENDS, 0

    for rid, dom, tit, body in con.execute("SELECT id, c0, c1, c2 FROM docs_content WHERE id >= ? AND id <= ? ORDER BY id", (lo_id, hi_id)):
        doc_ids.append(int(rid)); p_flags.append((1 if dom == "viwiki" else 0) | (2 if tit else 0))
        counts: dict[str, int] = {}
        if tit:
            tn = tit.strip().lower()
            if len(tn) >= 2 and not tn.isdigit():
                counts[tn] = _TITLE_TF_SENTINEL
                pi = tn.find(" (")
                if pi > 1 and len(tn[:pi].strip()) >= 2:
                    counts[tn[:pi].strip()] = _TITLE_TF_SENTINEL
        if body:
            if "<br" in body:
                body = body.replace("<br/>", " | ").replace("<br />", " | ").replace("<br>", " | ")
            toks = body.split()
            n_tok, span, at_start = len(toks), [], True
            for i, raw in enumerate(toks):
                c0, last = raw[0], raw[-1]
                if c0 == "{" or c0 == "<":
                    if span:
                        _flush_span(span, at_start, counts, title_set)
                    at_start = True; continue
                if "|" in raw and len(raw) > 1:
                    if span:
                        _flush_span(span, at_start, counts, title_set)
                    for part in raw.split("|"):
                        pw = part.strip(strip_c)
                        if len(pw) >= 3 and not _is_file_or_url(pw):
                            pl = pw.lower()
                            if pl in title_set or (pw[0].isupper() and pw.isalpha()):
                                counts[pl] = counts.get(pl, 0) + 1
                    at_start = True; continue
                if c0.isupper():
                    w = raw.strip(strip_c)
                    if w and w[0].isupper() and not _is_file_or_url(w):
                        if w.isalpha():
                            if not span and i + 1 < n_tok:
                                lw = w.lower()
                                if lw in f2:
                                    bi = lw + " " + toks[i + 1].strip(strip_c).lower()
                                    if bi in t2w:
                                        counts[bi] = counts.get(bi, 0) + 1
                            if len(span) < 4:
                                span.append(w)
                            if last in ends:
                                _flush_span(span, at_start, counts, title_set); at_start = True
                            else:
                                at_start = False
                            continue
                        if w.isalnum() and 2 <= len(w) <= 10:
                            counts[w.lower()] = counts.get(w.lower(), 0) + 1
                if span:
                    _flush_span(span, at_start, counts, title_set)
                if not c0.isalpha() and 2 <= len(raw) <= 12:
                    w = raw.strip(strip_c)
                    if 2 <= len(w) <= 10 and w.isalnum() and not w.isdigit() and not w.isalpha():
                        counts[w.lower()] = counts.get(w.lower(), 0) + 1
                at_start = last in ends
            if span:
                _flush_span(span, at_start, counts, title_set)
        for ph, tf_v in counts.items():
            lid = p2id.get(ph)
            if lid is None:
                lid = len(id2ph); p2id[ph] = lid; id2ph.append(ph)
                if ph in title_set:
                    title_lids.add(lid)
            raw_p.append(p_local); raw_e.append(lid); raw_tf.append(tf_v if tf_v < 65535 else 65535)
        p_local += 1
    con.close()
    h_tbl = np.fromiter((hash_phrase64(ph) for ph in id2ph), dtype=np.uint64, count=len(id2ph))
    e_hashes = h_tbl[np.frombuffer(raw_e, dtype=np.uint32)]
    t_hashes = h_tbl[np.fromiter(title_lids, dtype=np.int32, count=len(title_lids))] if title_lids else np.empty(0, dtype=np.uint64)
    return (doc_ids.tobytes(), p_flags.tobytes(), raw_p.tobytes(), e_hashes.tobytes(), raw_tf.tobytes(), t_hashes.tobytes())


def _load_wsd_aliases(wsd_path: Path, phrase_hashes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not wsd_path.is_file() or len(phrase_hashes) == 0:
        return np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.int32)
    h_to_eid = {int(h): i for i, h in enumerate(phrase_hashes.tolist())}
    alias_map: dict[int, int] = {}
    con = sqlite3.connect(f"file:{wsd_path}?mode=ro", uri=True)
    try:
        for vi_n, en_n in con.execute("SELECT vi_norm, en_norm FROM langlinks"):
            if vi_n and en_n and 3 <= len(en_n) <= 64 and "#" not in en_n:
                eid = h_to_eid.get(hash_phrase64(vi_n.strip().lower()))
                if eid is not None:
                    eh = hash_phrase64(en_n.strip().lower())
                    if eh not in h_to_eid and eh not in alias_map:
                        alias_map[eh] = eid
    except sqlite3.Error:
        pass
    finally:
        con.close()
    if not alias_map:
        return np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.int32)
    items = sorted(alias_map.items())
    return (np.fromiter((k for k, _ in items), dtype=np.uint64, count=len(items)),
            np.fromiter((v for _, v in items), dtype=np.int32, count=len(items)))


def _write_aligned(fh, arr: np.ndarray) -> None:
    raw = arr.tobytes(); fh.write(raw); pad = (-len(raw)) & 7
    if pad:
        fh.write(b"\x00" * pad)


def build_bipartite_index(
    vault_path: Path = DEFAULT_VAULT_DB, out_path: Path = DEFAULT_INDEX_BIN,
    wsd_path: Path = DEFAULT_WSD_DB, max_df: int = 5000, min_df: int = 2,
    gamma: float = 0.75, workers: int = 4,
) -> dict[str, float | int | str]:
    """Builds and atomically writes the HippoRAG-2 bipartite CSR index on Drive E:\\."""
    assert_not_c_drive(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    con = sqlite3.connect(f"file:{vault_path}?mode=ro", uri=True)
    lo_id, hi_id, n_tot = con.execute("SELECT MIN(id), MAX(id), COUNT(*) FROM docs_content").fetchone()
    con.close()
    if not n_tot:
        raise RuntimeError(f"Empty vault: {vault_path}")
    n_w = min(max(1, workers), os.cpu_count() or 1) if n_tot >= 20000 else 1
    step = (hi_id - lo_id + n_w) // n_w
    ranges = [(str(vault_path), lo_id + k * step, min(hi_id, lo_id + (k + 1) * step - 1)) for k in range(n_w)]
    if n_w == 1:
        parts = [_extract_shard(ranges[0])]
    else:
        with ProcessPoolExecutor(max_workers=n_w) as ex:
            parts = list(ex.map(_extract_shard, ranges))
    t_scan = time.perf_counter() - t0

    doc_p, flg_p, p_p, h_p, tf_p, th_p, p_off = [], [], [], [], [], [], 0
    for d_b, f_b, p_b, h_b, tf_b, th_b in parts:
        da = np.frombuffer(d_b, dtype=np.uint32).astype(np.int32)
        doc_p.append(da); flg_p.append(np.frombuffer(f_b, dtype=np.uint8))
        p_p.append(np.frombuffer(p_b, dtype=np.uint32).astype(np.int32) + p_off)
        h_p.append(np.frombuffer(h_b, dtype=np.uint64)); tf_p.append(np.frombuffer(tf_b, dtype=np.uint16))
        if th_b:
            th_p.append(np.frombuffer(th_b, dtype=np.uint64))
        p_off += len(da)

    passage_doc_ids, passage_flags = np.concatenate(doc_p), np.concatenate(flg_p)
    all_p, all_h, all_tf = np.concatenate(p_p), np.concatenate(h_p), np.concatenate(tf_p)
    title_hashes = np.unique(np.concatenate(th_p)) if th_p else np.empty(0, dtype=np.uint64)
    n_passages = len(passage_doc_ids)

    uniq_h, inv_e, raw_df = np.unique(all_h, return_inverse=True, return_counts=True)
    is_t = np.isin(uniq_h, title_hashes) if len(title_hashes) else np.zeros(len(uniq_h), dtype=bool)
    keep_u = (raw_df <= max_df) & ((raw_df >= min_df) | is_t)
    phrase_hashes, e_df, e_flags = uniq_h[keep_u], raw_df[keep_u].astype(np.int32), is_t[keep_u].astype(np.uint8)
    n_entities = len(phrase_hashes)

    remap = np.full(len(uniq_h), -1, dtype=np.int32); remap[keep_u] = np.arange(n_entities, dtype=np.int32)
    mapped_e = remap[inv_e]; emask = mapped_e >= 0
    edge_p, edge_e, raw_edge_tf = all_p[emask], mapped_e[emask], all_tf[emask]

    is_te = raw_edge_tf >= _TITLE_TF_SENTINEL
    real_tf = np.where(is_te, raw_edge_tf - _TITLE_TF_SENTINEL + 4, raw_edge_tf).astype(np.float32)
    idf = np.log((float(n_passages) + 1.0) / (e_df.astype(np.float32) + 1.0)) + 1.0
    edge_w = ((1.0 + np.log(real_tf)) * idf[edge_e] * np.where(is_te, 2.8, 1.0)).astype(np.float32)
    nnz = len(edge_p)

    pe_cnt = np.bincount(edge_p, minlength=n_passages).astype(np.int32)
    pe_indptr = np.empty(n_passages + 1, dtype=np.int32); pe_indptr[0] = 0; np.cumsum(pe_cnt, out=pe_indptr[1:])
    ep_ord = np.argsort(edge_e, kind="mergesort")
    ep_cnt = np.bincount(edge_e[ep_ord], minlength=n_entities).astype(np.int32)
    ep_indptr = np.empty(n_entities + 1, dtype=np.int32); ep_indptr[0] = 0; np.cumsum(ep_cnt, out=ep_indptr[1:])

    e_hub = np.power(np.maximum(e_df.astype(np.float32), 1.0), -float(gamma)).astype(np.float32)
    p_hub = np.power(np.maximum(pe_cnt.astype(np.float32), 1.0), -0.25).astype(np.float32)
    p_norm = np.maximum(np.bincount(edge_p, weights=edge_w * e_hub[edge_e], minlength=n_passages).astype(np.float32), 1e-6)
    e_norm = np.maximum(np.bincount(edge_e, weights=edge_w * p_hub[edge_p], minlength=n_entities).astype(np.float32), 1e-6)
    alias_hashes, alias_targets = _load_wsd_aliases(wsd_path, phrase_hashes)

    hdr = struct.pack(HEADER_FMT, MAGIC, FORMAT_VERSION, n_passages, n_entities, nnz,
                      len(alias_hashes), max_df, int(round(gamma * 1000)), 0, 0, 0, 0, 0, 0, 0)
    tmp_path = out_path.with_suffix(".bin.tmp")
    with tmp_path.open("wb") as fh:
        fh.write(hdr)
        for arr in (passage_doc_ids, passage_flags, p_hub, p_norm, phrase_hashes, e_df, e_hub, e_norm, e_flags,
                    pe_indptr, edge_e.astype(np.int32, copy=False), edge_w.astype(np.float16),
                    ep_indptr, edge_p[ep_ord].astype(np.int32, copy=False), edge_w[ep_ord].astype(np.float16),
                    alias_hashes, alias_targets):
            _write_aligned(fh, arr)
    os.replace(tmp_path, out_path)
    return {"index_path": str(out_path), "n_passages": n_passages, "n_entities": n_entities, "nnz": nnz,
            "n_aliases": int(len(alias_hashes)), "scan_sec": round(t_scan, 2),
            "total_sec": round(time.perf_counter() - t0, 2), "file_mb": round(out_path.stat().st_size / 1048576.0, 2)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Genesis-v8 HippoRAG-2 Bipartite CSR Index")
    ap.add_argument("--vault", type=Path, default=DEFAULT_VAULT_DB)
    ap.add_argument("--out", type=Path, default=DEFAULT_INDEX_BIN)
    ap.add_argument("--wsd", type=Path, default=DEFAULT_WSD_DB)
    ap.add_argument("--max-df", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    print(f"[HIPPORAG2-INDEX] {build_bipartite_index(a.vault, a.out, a.wsd, max_df=a.max_df, workers=a.workers)}")


if __name__ == "__main__":
    main()
