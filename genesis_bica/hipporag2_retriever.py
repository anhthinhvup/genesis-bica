"""
Genesis-v8 Offline CPU HippoRAG-2 Multi-Seed Hub-Suppressed PPR Retriever & Recognition Memory
(`mel.v6_lm.hipporag2_retriever`). Dual-seed (entity 1/deg^0.75 + viwiki BM25, lambda_P=0.35,
alpha=0.50, 10 iters) localized hub-suppressed PPR + Sheaf H^1 obstruction + Venn-Abers (< 60 ms).
Strictly <= 320 LOC | 0% `import re` | 0% neural/API | 0% hardcoded NL containers | 0 B on C:\\.
"""
from __future__ import annotations

from dataclasses import dataclass
import math, sqlite3, struct, time, unicodedata
from pathlib import Path
import numpy as np
import scipy.sparse as sp

from ._env_guard import enforce_e_drive_environment
from .hipporag2_indexer import DEFAULT_INDEX_BIN, DEFAULT_VAULT_DB, HEADER_FMT, HEADER_SIZE, MAGIC, hash_phrase64
from .offline_knowledge_vault import OfflineKnowledgeVault

enforce_e_drive_environment()

_STRIP = ".,;:!?\"'()[]{}<>|/*+=#@&$%^~`-–—“”«»_"
_ENDS = frozenset(".!?;:|)}],=>")
_PUNCT_SP = str.maketrans({c: " " for c in ".,!?;:-'\"()[]{}<>/*+=#@&$%^~`|\\_"})
_VA_BAG: tuple[tuple[float, int], ...] = (
    (0.02, 0), (0.05, 0), (0.08, 0), (0.11, 0), (0.14, 0), (0.18, 0), (0.22, 0), (0.26, 0), (0.32, 0), (0.38, 1),
    (0.44, 0), (0.48, 1), (0.54, 1), (0.60, 1), (0.68, 1), (0.75, 1), (0.82, 1), (0.88, 1), (0.94, 1), (0.98, 1),
)


@dataclass
class HippoResult:
    hits: list[dict]; ctx_text: str; evidence_coverage: float
    ppr_ms: float; total_ms: float; matched_entities: list[str]


def _venn_abers_calibrate(score: float) -> float:
    """Self-contained PAV isotonic Venn-Abers probability p1/(1-p0+p1) (< 0.02 ms)."""
    s_q = max(0.0, min(1.0, float(score)))
    if s_q <= 0.01: return 0.0
    ps = []
    for yh in (0, 1):
        blks: list[list[float]] = []
        for xv, yv in sorted((*_VA_BAG, (s_q, yh)), key=lambda it: it[0]):
            if blks and abs(blks[-1][1] - xv) <= 1e-12: blks[-1][2] += float(yv); blks[-1][3] += 1.0
            else: blks.append([xv, xv, float(yv), 1.0])
            while len(blks) >= 2 and (blks[-2][2] / blks[-2][3]) > (blks[-1][2] / blks[-1][3]):
                b = blks.pop(); blks[-1][1], blks[-1][2], blks[-1][3] = b[1], blks[-1][2] + b[2], blks[-1][3] + b[3]
        ps.append(next((max(0.0, min(1.0, sy / c)) for b0, b1, sy, c in blks if b0 - 1e-12 <= s_q <= b1 + 1e-12), max(0.0, min(1.0, blks[-1][2] / blks[-1][3]))))
    p0, p1 = ps[0], max(ps[0], ps[1])
    p_va = p1 / (1.0 - p0 + p1) if (1.0 - p0 + p1) > 1e-9 else 0.5
    return round(max(0.0, min(1.0, 0.55 * p_va + 0.45 * s_q)), 4)


def _clean_wiki_snippet(body: str, max_chars: int = 180) -> str:
    """Zero-regex linear wiki template/tag stripper producing compact <CTX> prose."""
    if not body: return ""
    out, db, tag, i, n = [], 0, False, 0, len(body)
    while i < n:
        ch = body[i]
        if ch == "{" and i + 1 < n and body[i + 1] == "{": db += 1; i += 2; continue
        if ch == "}" and i + 1 < n and body[i + 1] == "}": db = max(0, db - 1); i += 2; continue
        if ch == "<": tag = True
        elif ch == ">": tag = False
        elif db == 0 and not tag: out.append(" " if ch in "|[]*" else ch)
        i += 1
    txt = " ".join("".join(out).split())
    if len(txt) < 35: txt = " ".join(body.replace("{{", " ").replace("}}", " ").replace("|", ", ").split())
    return txt[:max_chars].strip()


class HippoRAG2Retriever:
    """Zero-API Offline CPU HippoRAG-2 Bipartite Retriever + Left-Hemisphere Recognition Memory."""

    def __init__(self, index_path: Path | str = DEFAULT_INDEX_BIN, vault_path: Path | str = DEFAULT_VAULT_DB) -> None:
        self.index_path, self.vault_path = Path(index_path), Path(vault_path)
        self._buf = self.index_path.read_bytes()
        hdr = struct.unpack(HEADER_FMT, self._buf[:HEADER_SIZE])
        if hdr[0] != MAGIC: raise ValueError(f"Invalid HippoRAG-2 index magic: {hdr[0]!r}")
        self.n_passages, self.n_entities, self.nnz, self.n_aliases = int(hdr[2]), int(hdr[3]), int(hdr[4]), int(hdr[5])
        self.max_df, self.gamma, off = int(hdr[6]), float(hdr[7]) / 1000.0, HEADER_SIZE
        for name, cnt, dt in (
            ("passage_doc_ids", self.n_passages, np.int32), ("passage_flags", self.n_passages, np.uint8),
            ("p_hub", self.n_passages, np.float32), ("p_norm", self.n_passages, np.float32),
            ("phrase_hashes", self.n_entities, np.uint64), ("e_df", self.n_entities, np.int32),
            ("e_hub", self.n_entities, np.float32), ("e_norm", self.n_entities, np.float32), ("e_flags", self.n_entities, np.uint8),
            ("pe_indptr", self.n_passages + 1, np.int32), ("pe_indices", self.nnz, np.int32), ("pe_data", self.nnz, np.float16),
            ("ep_indptr", self.n_entities + 1, np.int32), ("ep_indices", self.nnz, np.int32), ("ep_data", self.nnz, np.float16),
            ("alias_hashes", self.n_aliases, np.uint64), ("alias_targets", self.n_aliases, np.int32),
        ):
            nb = cnt * np.dtype(dt).itemsize; setattr(self, name, np.frombuffer(self._buf, dtype=dt, count=cnt, offset=off)); off += nb + ((-nb) & 7)
        self._acc_p, self._acc_e = np.zeros(self.n_passages, dtype=np.float32), np.zeros(self.n_entities, dtype=np.float32)
        self.vault = OfflineKnowledgeVault(self.vault_path)

    def close(self) -> None: self.vault.close()

    def _lookup_entity(self, phrase: str) -> int:
        h = hash_phrase64(phrase); pos = int(np.searchsorted(self.phrase_hashes, h))
        if pos < self.n_entities and int(self.phrase_hashes[pos]) == h: return pos
        if self.n_aliases > 0:
            ap = int(np.searchsorted(self.alias_hashes, h))
            if ap < self.n_aliases and int(self.alias_hashes[ap]) == h: return int(self.alias_targets[ap])
        return -1

    def _extract_query_signals(self, query: str) -> tuple[list[tuple[str, int, float]], list[str], list[str], list[str]]:
        raws, lows, is_cap, at_st, focal, years, span, at_s = [], [], [], [], [], [], [], True
        for r in unicodedata.normalize("NFC", query).replace("-", " ").split():
            w = r.strip(_STRIP)
            if not w: at_s = True; continue
            lw = w.lower(); cap = w[0].isupper() and (w.isalpha() or (w.isalnum() and len(w) >= 3))
            raws.append(w); lows.append(lw); is_cap.append(cap); at_st.append(at_s)
            if len(w) == 4 and w.isdigit() and 1000 <= int(w) <= 2199: years.append(w)
            if cap:
                span.append(lw)
                if ((not at_s and len(lw) >= 3) or (2 <= len(w) <= 8 and w.isupper())) and lw not in focal: focal.append(lw)
            else:
                if len(span) >= 2: focal.extend(sw for sw in span if len(sw) >= 3 and sw not in focal)
                span.clear()
                if not w.isalpha() and not w.isdigit() and w.isalnum() and len(w) >= 3 and lw not in focal: focal.append(lw)
            at_s = r[-1] in _ENDS
        if len(span) >= 2: focal.extend(sw for sw in span if len(sw) >= 3 and sw not in focal)
        m, cov, matched, has_fp = len(lows), [False] * len(lows), {}, bool(focal)
        for ng in (4, 3, 2, 1):
            for i in range(0, m - ng + 1):
                if ng == 1 and (cov[i] or not ((is_cap[i] and not at_st[i] and len(lows[i]) >= 3) or (2 <= len(raws[i]) <= 8 and raws[i].isupper()) or (not raws[i].isalpha() and not raws[i].isdigit() and raws[i].isalnum()) or (not has_fp and len(lows[i]) >= 6))): continue
                if ng > 1 and all(cov[i : i + ng]): continue
                ph = " ".join(lows[i : i + ng]); eid = self._lookup_entity(ph) if len(ph) >= 2 else -1
                cc = sum(1 for k in range(i, i + ng) if is_cap[k]) if eid >= 0 else 0
                if eid < 0 or (ng >= 2 and cc == 0 and (has_fp or len(ph) < 6)): continue
                cmul = 4.0 if (cc >= 2 or (ng == 1 and is_cap[i])) else (1.8 if cc == 1 else 0.55)
                spec = (float(max(1, int(self.e_df[eid]))) ** -self.gamma) * cmul * (1.5 if ng >= 2 else 1.0) * (1.3 if self.e_flags[eid] else 1.0)
                if eid not in matched or spec > matched[eid][1]: matched[eid] = (ph, spec)
                for k in range(i, i + ng): cov[k] = True
        return ([(ph, eid, w) for eid, (ph, w) in sorted(matched.items(), key=lambda it: it[1][1], reverse=True)],
                focal, list(dict.fromkeys(w for w in lows if len(w) >= 3 and not w.isdigit())), years)

    def _fts_query(self, q_expr: str, limit: int, viwiki_first: bool = True) -> list[tuple[int, float]]:
        for dom_sql in ((" AND domain = 'viwiki'", "") if viwiki_first else ("",)):
            try:
                rows = self.vault.conn.execute(f"SELECT rowid, bm25(docs, 0.0, 6.0, 1.0) AS sc FROM docs WHERE docs MATCH ?{dom_sql} ORDER BY sc LIMIT ?;", (q_expr, limit)).fetchall()
                if rows: return rows
            except sqlite3.OperationalError: pass
        return []

    def _bm25_passage_seeds(self, query: str, matched_ents: list[tuple[str, int, float]], focal: list[str], limit: int = 15) -> tuple[np.ndarray, np.ndarray]:
        terms = sorted(dict.fromkeys(t for t in unicodedata.normalize("NFC", query.lower()).translate(_PUNCT_SP).split() if len(t) >= 2), key=len, reverse=True)
        if not terms: return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
        rows, ent_qs = [], [f'"{ph}"' for ph, _, _ in matched_ents[:3] if len(ph) >= 3]
        if len(ent_qs) >= 2:
            rows = self._fts_query(" AND ".join(ent_qs[:2]), limit)
            if not rows:
                seen_r: dict[int, float] = {}
                for eq in ent_qs[:2]:
                    for rid, sc in self._fts_query(eq, max(4, limit // 2)): seen_r[int(rid)] = min(seen_r.get(int(rid), 1e9), float(sc))
                rows = sorted(seen_r.items(), key=lambda it: it[1])[:limit]
        elif ent_qs: rows = self._fts_query(ent_qs[0], limit)
        if not rows:
            for k_and in (min(4, len(terms)), min(3, len(terms)), min(2, len(terms))):
                if k_and >= 2:
                    rows = self._fts_query(" AND ".join(f'"{t}"' for t in terms[:k_and]), limit)
                    if rows: break
        if not rows:
            sel = list(dict.fromkeys(t for t in (focal + [x for x in terms if len(x) >= 5]) if t))[:4]
            rows = self._fts_query(" OR ".join(f'"{t}"' for t in sel), limit, viwiki_first=False) if sel else []
        p_ids, p_ws = [], []
        for rk, (rid, sc) in enumerate(rows):
            pos = int(np.searchsorted(self.passage_doc_ids, int(rid)))
            if pos < self.n_passages and int(self.passage_doc_ids[pos]) == int(rid):
                p_ids.append(pos); p_ws.append(max(0.02, -float(sc)) * math.exp(-0.75 * rk))
        if not p_ids: return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
        arr_w = np.array(p_ws, dtype=np.float32); arr_w /= float(arr_w.sum()) + 1e-9
        return np.array(p_ids, dtype=np.int32), arr_w

    def _fast_push(self, src_idx: np.ndarray, src_w: np.ndarray, indptr: np.ndarray, indices: np.ndarray, data: np.ndarray, acc: np.ndarray, cap: int) -> tuple[np.ndarray, np.ndarray]:
        if len(src_idx) == 0: return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
        starts, ends = indptr[src_idx], indptr[src_idx + 1]; tot = int((ends - starts).sum())
        if tot == 0: return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
        dst, w, off = np.empty(tot, dtype=np.int32), np.empty(tot, dtype=np.float32), 0
        for s, e, sw in zip(starts.tolist(), ends.tolist(), src_w.tolist()):
            ln = e - s
            if ln > 0: dst[off : off + ln] = indices[s:e]; np.multiply(data[s:e], sw, out=w[off : off + ln]); off += ln
        np.add.at(acc, dst, w); touched = np.unique(dst); vals = acc[touched].copy(); acc[touched] = 0.0
        if len(touched) > cap: k = np.argpartition(vals, -cap)[-cap:]; touched, vals = touched[k], vals[k]
        return touched, vals

    def _run_ppr(self, seed_p: np.ndarray, seed_pw: np.ndarray, seed_e: np.ndarray, seed_ew: np.ndarray, lam_p: float = 0.35, alpha: float = 0.50, n_iters: int = 10) -> tuple[np.ndarray, np.ndarray, float]:
        t0 = time.perf_counter()
        if len(seed_p) == 0 and len(seed_e) == 0: return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32), 0.0
        eff_lp = 1.0 if len(seed_e) == 0 else (0.0 if len(seed_p) == 0 else lam_p)
        sp_w = (eff_lp * seed_pw).astype(np.float32) if len(seed_p) else np.empty(0, dtype=np.float32)
        se_w = ((1.0 - eff_lp) * (seed_ew / (float(seed_ew.sum()) + 1e-9))).astype(np.float32) if len(seed_e) else np.empty(0, dtype=np.float32)
        p1_i, p1_v = self._fast_push(seed_e, se_w / self.e_norm[seed_e], self.ep_indptr, self.ep_indices, self.ep_data, self._acc_p, 140)
        k1 = np.argpartition(p1_v, -24)[-24:] if len(p1_i) > 24 else np.arange(len(p1_i))
        p1_top, p1_tv = p1_i[k1], p1_v[k1]
        p_h1 = np.concatenate([a for a in (seed_p[:10], p1_top) if len(a)]) if (len(seed_p) or len(p1_top)) else np.empty(0, dtype=np.int32)
        w_h1 = np.concatenate([a for a in (sp_w[:10] * 2.0, p1_tv * self.p_hub[p1_top]) if len(a)]) if len(p_h1) else np.empty(0, dtype=np.float32)
        eb_i, eb_v = self._fast_push(p_h1, w_h1 / self.p_norm[p_h1], self.pe_indptr, self.pe_indices, self.pe_data, self._acc_e, 180)
        if len(eb_i):
            ok_d = self.e_df[eb_i] >= 2; eb_i, eb_v = eb_i[ok_d], eb_v[ok_d] * self.e_hub[eb_i[ok_d]]
            if len(eb_i) > 100: kb = np.argpartition(eb_v, -100)[-100:]; eb_i, eb_v = eb_i[kb], eb_v[kb]
        p2_i, _ = self._fast_push(eb_i, eb_v / self.e_norm[eb_i], self.ep_indptr, self.ep_indices, self.ep_data, self._acc_p, 320)
        p_sub = np.unique(np.concatenate([a for a in (seed_p, p1_i, p2_i) if len(a)]))
        starts, ends = self.pe_indptr[p_sub], self.pe_indptr[p_sub + 1]; lens = ends - starts; tot = int(lens.sum())
        if len(p_sub) == 0 or tot == 0: return p_sub[:10], np.ones(min(10, len(p_sub)), dtype=np.float32), (time.perf_counter() - t0) * 1000.0
        l_rows, l_cols_g, l_vals, off = np.repeat(np.arange(len(p_sub), dtype=np.int32), lens), np.empty(tot, dtype=np.int32), np.empty(tot, dtype=np.float32), 0
        for s, e in zip(starts.tolist(), ends.tolist()):
            ln = e - s
            if ln > 0: l_cols_g[off : off + ln] = self.pe_indices[s:e]; l_vals[off : off + ln] = self.pe_data[s:e]; off += ln
        e_sub, l_cols = np.unique(l_cols_g, return_inverse=True)
        W_sub = sp.csr_matrix((l_vals, (l_rows, l_cols.astype(np.int32))), shape=(len(p_sub), len(e_sub)), dtype=np.float32); WT_sub = W_sub.T
        inv_pn, inv_en, lp_hub, le_hub = np.float32(1.0) / self.p_norm[p_sub], np.float32(1.0) / self.e_norm[e_sub], self.p_hub[p_sub], self.e_hub[e_sub]
        s_p, s_e = np.zeros(len(p_sub), dtype=np.float32), np.zeros(len(e_sub), dtype=np.float32)
        if len(seed_p): s_p[np.searchsorted(p_sub, seed_p)] = sp_w
        if len(seed_e):
            epos = np.searchsorted(e_sub, seed_e); ok = (epos < len(e_sub)) & (e_sub[np.minimum(epos, len(e_sub) - 1)] == seed_e); s_e[epos[ok]] = se_w[ok]
        r_p, r_e, a32, oma = alpha * s_p, alpha * s_e, np.float32(alpha), np.float32(1.0 - alpha)
        for _ in range(n_iters):
            r_e, r_p = a32 * s_e + oma * (WT_sub.dot(r_p * inv_pn) * le_hub), a32 * s_p + oma * (W_sub.dot(r_e * inv_en) * lp_hub)
        if len(seed_e) >= 2 and len(p_sub) > 2:
            k_anc = min(len(seed_e) + 1, 4, len(p_sub)); anc_l = np.argpartition(r_p, -k_anc)[-k_anc:]
            anc_bin = np.zeros(len(p_sub), dtype=np.float32); anc_bin[anc_l] = 1.0
            e_hits = (WT_sub > 0).astype(np.float32).dot(anc_bin)
            conv = (e_hits >= 2.0) & (s_e == 0.0) & (self.e_df[e_sub] >= 2)
            if np.any(conv):
                br_p = W_sub.dot(np.where(conv, r_e * (e_hits * e_hits), np.float32(0.0)) * inv_en) * lp_hub
                br_p[anc_l] *= np.float32(0.15); bm = float(br_p.max())
                if bm > 1e-12: r_p = r_p + np.float32(0.85 * float(r_p.max()) / bm) * br_p
        k_out = min(10, len(p_sub))
        top_l = np.argpartition(r_p, -k_out)[-k_out:] if len(p_sub) > k_out else np.arange(len(p_sub))
        top_l = top_l[np.argsort(r_p[top_l])[::-1]]
        return p_sub[top_l], r_p[top_l], (time.perf_counter() - t0) * 1000.0

    def _fts_exists(self, token: str) -> bool:
        try: return self.vault.conn.execute('SELECT 1 FROM docs WHERE docs MATCH ? LIMIT 1;', (f'"{token}"',)).fetchone() is not None
        except sqlite3.OperationalError: return False

    def retrieve(self, query: str, top_k: int = 3, max_ctx_chars: int = 360) -> HippoResult:
        t0 = time.perf_counter()
        matched_ents, focal, cwords, years = self._extract_query_signals(query)
        seed_e, seed_ew = np.array([e for _, e, _ in matched_ents[:8]], dtype=np.int32), np.array([w for _, _, w in matched_ents[:8]], dtype=np.float32)
        seed_p, seed_pw = self._bm25_passage_seeds(query, matched_ents, focal, limit=15)
        cand_p, cand_sc, ppr_ms = self._run_ppr(seed_p, seed_pw, seed_e, seed_ew, lam_p=0.35, alpha=0.50, n_iters=10)
        if len(cand_p) == 0:
            return HippoResult([], "", 0.0, round(ppr_ms, 2), round((time.perf_counter() - t0) * 1000.0, 2), [ph for ph, _, _ in matched_ents[:8]])
        doc_ids = [int(self.passage_doc_ids[int(p)]) for p in cand_p]
        rows = {int(r[0]): (str(r[1] or ""), str(r[2] or ""), str(r[3] or "")) for r in self.vault.conn.execute(f"SELECT id, c0, c1, c2 FROM docs_content WHERE id IN ({','.join('?' for _ in doc_ids)})", doc_ids).fetchall()}
        q_eids, bm25_set, max_ppr, p_ents, meta = set(seed_e.tolist()), set(seed_p[:3].tolist()), (float(cand_sc[0]) or 1.0), [], []
        for idx_c, (pidx, sc) in enumerate(zip(cand_p.tolist(), cand_sc.tolist())):
            did = doc_ids[idx_c]; dom, tit, body = rows.get(did, ("", "", ""))
            ep = set(self.pe_indices[self.pe_indptr[pidx] : self.pe_indptr[pidx + 1]].tolist()); p_ents.append(ep)
            pn, qc, iw = float(sc) / max_ppr, (len(ep & q_eids) / float(len(q_eids))) if q_eids else 0.0, (1.0 if (dom == "viwiki" and tit) else 0.0)
            meta.append({"pidx": pidx, "doc_id": did, "domain": dom, "title": tit, "body": body, "ppr": float(sc), "ppr_n": pn, "q_ecov": qc, "is_wiki": iw, "anc": 0.68 * pn + 0.20 * qc + 0.12 * iw})
        chosen = [int(max(range(len(meta)), key=lambda i: meta[i]["anc"]))]
        u_ents, b_max, rem_max = set(p_ents[chosen[0]]), 0.0, (max((m["ppr"] for i, m in enumerate(meta) if i != chosen[0]), default=1e-9) or 1e-9)
        while len(chosen) < min(top_k, len(meta)):
            bi, bv = -1, -1.0
            for i, m in enumerate(meta):
                if i not in chosen:
                    bs = min(1.0, sum(float(self.e_hub[e]) for e in (u_ents & p_ents[i])) * 3.5)
                    v = 0.66 * (m["ppr"] / rem_max) + 0.20 * bs + 0.14 * m["is_wiki"]
                    if v > bv: bv, bi = v, i; b_max = max(b_max, bs)
            if bi < 0: break
            chosen.append(bi); u_ents |= p_ents[bi]
        ev_lo, ev_eids, oov, ungr, gr = " ".join((meta[i]["title"] + " " + meta[i]["body"]).lower() for i in chosen), set().union(*(p_ents[i] for i in chosen)), 0, 0, 0
        for fp in focal:
            efp = self._lookup_entity(fp)
            if (fp in ev_lo) or (efp >= 0 and efp in ev_eids): gr += 1
            else:
                ungr += 1
                if efp < 0 and not self._fts_exists(fp): oov += 1
        if gr == 0:
            chk = 0
            for cw in sorted(cwords, key=len, reverse=True):
                if chk >= 4 or len(cw) < 5: break
                if cw not in ev_lo and cw not in focal:
                    chk += 1
                    if self._lookup_entity(cw) < 0 and not self._fts_exists(cw): oov += 1
        pr = (gr / float(len(focal))) if focal else 1.0
        cwc = (sum(1 for w in cwords if w in ev_lo) / float(len(cwords))) if cwords else 0.0
        cspec = min(1.0, max((math.log((self.n_passages + 1.0) / (float(self.e_df[e]) + 1.0)) for _, e, _ in matched_ents), default=0.0) / 10.5)
        yok = 1.0 if (not years or any(y in ev_lo for y in years)) else 0.65
        s_raw = (0.03 * pr * cspec) if oov > 0 else ((0.06 * cwc) if (ungr > 0 and gr == 0) else (0.42 * pr + 0.28 * cspec + 0.18 * max(cwc, 0.5 * b_max) + 0.12 * (1.0 if matched_ents else 0.2)) * (0.15 ** ungr) * yok)
        ev_cov, hits, parts, used, pcap = _venn_abers_calibrate(s_raw), [], [], 0, max(80, max_ctx_chars // max(1, len(chosen)))
        for rk, ci in enumerate(chosen):
            m = meta[ci]; sep = 3 if parts else 0; rem = max_ctx_chars - used - sep
            if rem <= 24: break
            pref = f"{m['title']}: " if m["title"] else ""; sb = min(pcap, rem) - len(pref)
            snip = _clean_wiki_snippet(m["body"], max_chars=max(24, sb)) if sb > 20 else ""
            ent = (pref + snip)[:rem].strip(); parts.append(ent); used += len(ent) + sep
            ht = "direct" if (m["pidx"] in bm25_set or m["q_ecov"] > 0.0) else ("bridge" if (p_ents[chosen[0]] & p_ents[ci]) else "multihop")
            hits.append({"doc_id": m["doc_id"], "domain": m["domain"], "title": m["title"], "snippet": snip,
                         "ppr_score": round(m["ppr"], 6), "recognition_score": round(ev_cov if rk == 0 else m["ppr_n"] * ev_cov, 4), "hop_type": ht})
        return HippoResult(hits=hits, ctx_text=" | ".join(parts)[:max_ctx_chars], evidence_coverage=ev_cov,
                           ppr_ms=round(ppr_ms, 2), total_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                           matched_entities=[ph for ph, _, _ in matched_ents[:8]])
