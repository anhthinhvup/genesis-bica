"""
Genesis-v8 Tier-1 Zero-Retrain Fast Episodic Memory (`mel.v6_lm.fast_episodic_memory`, Spec #88).
Running-Mean Centering + L2 Whitening (`mean_cos ~ 0.18 -> ~0`), High-beta Modern Hopfield + Kanerva
Sparse-Neighborhood Episodic Slots (`M <= 4,096`, `< 12 MB` RAM, SQLite WAL on Drive E:\\), and ATMS
Non-Monotonic 1-Shot Belief Revision (`ACTIVE` -> `SUPERSEDED` audit ledger).
Strictly <= 350 LOC | 0% `import re`/torch/sklearn/transformers | 0% hardcoded NL templates | Drive E:\\ Only.
"""
from __future__ import annotations

from dataclasses import dataclass
import functools, hashlib, math, sqlite3, sys, threading, time, unicodedata
from pathlib import Path
from typing import Any

import numpy as np

from ._env_guard import assert_not_c_drive, enforce_e_drive_environment

enforce_e_drive_environment()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EPISODIC_DB = PROJECT_ROOT / ".cache_e" / "episodic_slots.sqlite"
STATUS_ACTIVE, STATUS_SUPERSEDED, STATUS_ARCHIVED = "ACTIVE", "SUPERSEDED", "ARCHIVED"
_NULL_TAU, _EPS = 0.30, 1e-8
_PUNCT_SPACE = str.maketrans({c: " " for c in ".,!?;:-'\"()[]{}<>/*+=#@&$%^~`|\\_\t\r\n"})
_ROW_KEYS = ("slot_id", "entity", "attribute", "text", "surprise", "da_reward", "ts", "status")


@dataclass(slots=True)
class EpisodicHit:
    slot_id: int
    entity: str
    attribute: str
    text: str
    score: float
    surprise: float
    da_reward: float
    ts: float
    status: str


def _norm_key(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).strip().lower().split()) if value else ""

def _tokenize(text: str) -> list[str]:
    return [t for t in unicodedata.normalize("NFC", text.lower()).translate(_PUNCT_SPACE).split() if t] if text else []

@functools.lru_cache(maxsize=8192)
def _feat_proj(feat: str, dim: int) -> tuple[int, float, int, float]:
    h = int.from_bytes(hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest(), "little")
    return h % dim, (1.0 if (h >> 16) & 1 else -1.0), (h >> 17) % dim, (1.0 if (h >> 33) & 1 else -1.0)

def _encode_phrase(text: str, dim: int, tag: str = "") -> np.ndarray | None:
    toks = _tokenize(text)
    if not toks and not tag:
        return None
    buf = [0.0] * dim
    def _add(feat: str, w: float) -> None:
        i1, s1, i2, s2 = _feat_proj(feat, dim)
        buf[i1] += s1 * w
        buf[i2] += s2 * w
    if tag:
        _add(tag, 0.65)
    for idx, tok in enumerate(toks):
        lt = len(tok)
        _add(f"W::{tok}", 1.0 + 0.25 * (lt if lt < 8 else 8))
        if idx + 1 < len(toks):
            _add(f"BI::{tok}_{toks[idx + 1]}", 1.35)
        for j in range(lt - 2 if lt >= 4 else 0):
            _add(f"C3::{tok[j : j + 3]}", 0.30)
    arr = np.asarray(buf, dtype=np.float32)
    nrm = math.sqrt(float(np.dot(arr, arr)))
    return (arr * (1.0 / nrm)) if nrm > _EPS else None

def _encode_semantic_vector(text: str, entity: str, attribute: str, dim: int) -> np.ndarray:
    """Length-invariant unit-component CountSketch semantic encoder (0% regex, 0% NL stopwords)."""
    if not entity and not attribute:
        v = _encode_phrase(text, dim)
        return v if v is not None else np.zeros(dim, dtype=np.float32)
    acc = np.zeros(dim, dtype=np.float32)
    if (v_txt := _encode_phrase(text, dim)) is not None:
        acc += v_txt
    if (ent_n := _norm_key(entity)) and (v_ent := _encode_phrase(ent_n, dim, f"ENT::{ent_n}")) is not None:
        acc += 1.15 * v_ent
    if (attr_n := _norm_key(attribute)) and (v_att := _encode_phrase(attr_n, dim, f"ATTR::{attr_n}")) is not None:
        acc += 1.15 * v_att
    nrm = math.sqrt(float(np.dot(acc, acc)))
    return (acc * (1.0 / nrm)) if nrm > _EPS else acc


class FastEpisodicMemory:
    """Zero-Retrain Tier-1 Centered Modern Hopfield + Kanerva SDM + ATMS Belief Revision."""

    def __init__(self, db_path: str | Path = DEFAULT_EPISODIC_DB, dim: int = 384, max_slots: int = 4096, beta: float = 16.0) -> None:
        self.dim, self.max_slots, self.beta = int(dim), max(1, int(max_slots)), float(beta)
        self._lock, self._closed = threading.RLock(), False
        if str(db_path) == ":memory:":
            self.db_path: str | Path = ":memory:"
            self._conn = sqlite3.connect(":memory:", check_same_thread=False, timeout=5.0)
        else:
            p = assert_not_c_drive(Path(db_path))
            p.parent.mkdir(parents=True, exist_ok=True)
            self.db_path = p
            self._conn = sqlite3.connect(str(p), check_same_thread=False, timeout=5.0)
        self._conn.executescript(
            "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA temp_store=MEMORY;"
            "CREATE TABLE IF NOT EXISTS episodic_slots ("
            "slot_id INTEGER PRIMARY KEY AUTOINCREMENT, entity TEXT NOT NULL, entity_norm TEXT NOT NULL, "
            "attribute TEXT NOT NULL, attribute_norm TEXT NOT NULL, text TEXT NOT NULL, "
            "surprise REAL NOT NULL, da_reward REAL NOT NULL, ts REAL NOT NULL, "
            "status TEXT NOT NULL, superseded_by INTEGER DEFAULT 0, key_blob BLOB NOT NULL);"
            "CREATE INDEX IF NOT EXISTS idx_ep_status_ent ON episodic_slots(status, entity_norm, attribute_norm);"
            "CREATE INDEX IF NOT EXISTS idx_ep_da ON episodic_slots(status, da_reward DESC, surprise DESC, ts DESC);"
        )
        self._raw_keys = np.zeros((self.max_slots, self.dim), dtype=np.float32)
        self._inv_norms, self._dot_buf = np.ones(self.max_slots, dtype=np.float32), np.empty(self.max_slots, dtype=np.float32)
        self._key_sum, self._mu_cached = np.zeros(self.dim, dtype=np.float64), np.zeros(self.dim, dtype=np.float32)
        self._key_count = self._n_hot = self._superseded_count = 0
        self._norms_dirty, self._last_ts = False, 0.0
        self._meta: list[tuple[int, str, str, str, str, str, float, float, float]] = []
        self._slot_to_row: dict[int, int] = {}
        self._active_by_entity, self._active_by_ent_attr = {}, {}
        self._load_from_sqlite()

    def _load_from_sqlite(self) -> None:
        cnt = self._conn.execute("SELECT COUNT(*) FROM episodic_slots WHERE status=?;", (STATUS_SUPERSEDED,)).fetchone()
        self._superseded_count = int(cnt[0]) if cnt else 0
        rows = self._conn.execute(
            "SELECT slot_id, entity, entity_norm, attribute, attribute_norm, text, surprise, da_reward, ts, key_blob "
            "FROM episodic_slots WHERE status=? ORDER BY slot_id DESC LIMIT ?;",
            (STATUS_ACTIVE, self.max_slots),
        ).fetchall()
        for sid, ent, ent_n, attr, attr_n, txt, surp, da, ts, blob in reversed(rows):
            vec = np.frombuffer(blob, dtype=np.float32)
            if vec.size == self.dim:
                self._append_hot(int(sid), str(ent), str(ent_n), str(attr), str(attr_n), str(txt), float(surp), float(da), float(ts), vec)
                self._last_ts = max(self._last_ts, float(ts))
        self._sync_centering(force=True)

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._conn.commit()
                self._conn.close()
                self._closed = True

    def __enter__(self) -> FastEpisodicMemory: return self
    def __exit__(self, *_exc: Any) -> None: self.close()
    def encode_text(self, text: str, entity: str = "", attribute: str = "") -> np.ndarray:
        return _encode_semantic_vector(text, entity, attribute, self.dim)

    def _coerce_or_encode(self, text: str, entity: str, attribute: str, key_vec: Any) -> np.ndarray:
        if key_vec is None:
            return _encode_semantic_vector(text, entity, attribute, self.dim)
        arr = np.asarray(key_vec, dtype=np.float32).reshape(-1)
        if arr.size != self.dim:
            raise ValueError(f"Expected key_vec dim={self.dim}, got {arr.size}")
        return arr

    def _sync_centering(self, force: bool = False) -> None:
        if not force and not self._norms_dirty:
            return
        if self._key_count <= 1:
            mu = np.zeros(self.dim, dtype=np.float32)
        else:
            mu = ((((self._key_count - 1.0) / (self._key_count + 2.0)) / float(self._key_count)) * self._key_sum).astype(np.float32)
        diff = mu - self._mu_cached
        if force or float(np.dot(diff, diff)) > 1e-10:
            self._mu_cached[:] = mu
            for s in range(0, self._n_hot, 512):
                sub = self._raw_keys[s : min(self._n_hot, s + 512)] - self._mu_cached
                self._inv_norms[s : min(self._n_hot, s + 512)] = 1.0 / (np.sqrt(np.einsum("ij,ij->i", sub, sub)) + _EPS)
        self._norms_dirty = False

    def _center_unit(self, vec: np.ndarray) -> np.ndarray:
        qc = vec - self._mu_cached
        return qc * (1.0 / (math.sqrt(float(np.dot(qc, qc))) + _EPS))

    def center_vector(self, vec: Any) -> np.ndarray:
        with self._lock:
            self._sync_centering()
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
            if arr.size != self.dim:
                raise ValueError(f"Expected vector dim={self.dim}, got {arr.size}")
            return self._center_unit(arr).astype(np.float32)

    def _append_hot(self, sid: int, entity: str, ent_n: str, attr: str, attr_n: str, text: str, surp: float, da: float, ts: float, vec: np.ndarray) -> None:
        if self._n_hot >= self.max_slots:
            r_ev = min(range(self._n_hot), key=lambda r: (self._meta[r][7] + 0.25 * self._meta[r][6], self._meta[r][8], self._meta[r][0]))
            sid_ev = self._meta[r_ev][0]
            self._remove_hot_row(r_ev)
            self._conn.execute("UPDATE episodic_slots SET status=? WHERE slot_id=?;", (STATUS_ARCHIVED, sid_ev))
        row = self._n_hot
        self._raw_keys[row] = vec
        diff = vec - self._mu_cached
        self._inv_norms[row] = 1.0 / (math.sqrt(float(np.dot(diff, diff))) + _EPS)
        self._meta.append((sid, entity, ent_n, attr, attr_n, text, surp, da, ts))
        self._slot_to_row[sid] = row
        if ent_n:
            self._active_by_entity.setdefault(ent_n, set()).add(sid)
        if ent_n or attr_n:
            self._active_by_ent_attr.setdefault((ent_n, attr_n), set()).add(sid)
        self._key_sum += vec.astype(np.float64)
        self._key_count, self._n_hot, self._norms_dirty = self._key_count + 1, row + 1, True

    def _remove_hot_row(self, row: int) -> None:
        sid, _, ent_n, _, attr_n, *_ = self._meta[row]
        self._key_sum -= self._raw_keys[row].astype(np.float64)
        self._key_count = max(0, self._key_count - 1)
        if self._key_count == 0:
            self._key_sum.fill(0.0)
        self._slot_to_row.pop(sid, None)
        for idx_map, key in ((self._active_by_entity, ent_n), (self._active_by_ent_attr, (ent_n, attr_n))):
            if key in idx_map:
                idx_map[key].discard(sid)
                if not idx_map[key]:
                    del idx_map[key]
        last = self._n_hot - 1
        if row != last:
            self._raw_keys[row], self._inv_norms[row] = self._raw_keys[last], self._inv_norms[last]
            moved = self._meta[last]
            self._meta[row], self._slot_to_row[moved[0]] = moved, row
        self._meta.pop()
        self._n_hot, self._norms_dirty = last, True

    def _insert_slot(self, text: str, entity: str, attribute: str, surprise: float, da_reward: float, vec: np.ndarray) -> int:
        ent_c, attr_c, txt_c = " ".join(entity.split()).strip(), " ".join(attribute.split()).strip(), " ".join(text.split()).strip()
        ent_n, attr_n, now = _norm_key(ent_c), _norm_key(attr_c), time.time()
        self._last_ts = now if now > self._last_ts + 1e-6 else (self._last_ts + 1e-6)
        cur = self._conn.execute(
            "INSERT INTO episodic_slots (entity, entity_norm, attribute, attribute_norm, text, "
            "surprise, da_reward, ts, status, superseded_by, key_blob) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?);",
            (ent_c, ent_n, attr_c, attr_n, txt_c, float(surprise), float(da_reward), self._last_ts, STATUS_ACTIVE, vec.tobytes()),
        )
        sid = int(cur.lastrowid)
        self._append_hot(sid, ent_c, ent_n, attr_c, attr_n, txt_c, float(surprise), float(da_reward), self._last_ts, vec)
        return sid

    def _centered_sims(self, q_hat: np.ndarray, row_arr: np.ndarray | None = None) -> np.ndarray:
        mu_dot = float(np.dot(self._mu_cached, q_hat))
        if row_arr is None:
            sims = self._dot_buf[: self._n_hot]
            np.matmul(self._raw_keys[: self._n_hot], q_hat, out=sims)
            sims -= mu_dot
            sims *= self._inv_norms[: self._n_hot]
        else:
            sims = (self._raw_keys[row_arr] @ q_hat - mu_dot) * self._inv_norms[row_arr]
        np.clip(sims, -1.0, 1.0, out=sims)
        return sims

    def write(self, text: str, entity: str = "", attribute: str = "", surprise: float = 1.0, da_reward: float = 0.5, key_vec: Any = None) -> int:
        with self._lock:
            sid = self._insert_slot(text, entity, attribute, surprise, da_reward, self._coerce_or_encode(text, entity, attribute, key_vec))
            self._conn.commit()
            return sid

    def revise_belief(self, entity: str, attribute: str, new_text: str, surprise: float = 1.0, da_reward: float = 0.8, key_vec: Any = None) -> int:
        """ATMS Non-Monotonic 1-Shot Revision: mark conflicting prior slot(s) SUPERSEDED and write new ACTIVE slot."""
        with self._lock:
            ent_n, attr_n = _norm_key(entity), _norm_key(attribute)
            vec = self._coerce_or_encode(new_text, entity, attribute, key_vec)
            prior_sids: list[int] = []
            if ent_n and attr_n:
                prior_sids = list(self._active_by_ent_attr.get((ent_n, attr_n), ()))
            elif ent_n:
                prior_sids = list(self._active_by_ent_attr.get((ent_n, ""), ()))
                if not prior_sids and ent_n in self._active_by_entity:
                    self._sync_centering()
                    rows = [self._slot_to_row[s] for s in self._active_by_entity[ent_n] if s in self._slot_to_row]
                    if rows:
                        r_arr = np.asarray(rows, dtype=np.int32)
                        sims = self._centered_sims(self._center_unit(vec), r_arr)
                        prior_sids = [self._meta[int(r_arr[i])][0] for i in np.flatnonzero(sims >= 0.55)]
            elif self._n_hot > 0:
                self._sync_centering()
                sims = self._centered_sims(self._center_unit(vec))
                prior_sids = [self._meta[int(r)][0] for r in np.flatnonzero(sims >= 0.80)]
            for sid in prior_sids:
                if (r := self._slot_to_row.get(sid)) is not None:
                    self._remove_hot_row(r)
            new_sid = self._insert_slot(new_text, entity, attribute, surprise, da_reward, vec)
            if prior_sids:
                self._conn.executemany("UPDATE episodic_slots SET status=?, superseded_by=? WHERE slot_id=?;", [(STATUS_SUPERSEDED, new_sid, sid) for sid in prior_sids])
                self._superseded_count += len(prior_sids)
            self._conn.commit()
            return new_sid

    def retrieve(self, query: str, entity: str = "", key_vec: Any = None, top_k: int = 3) -> list[EpisodicHit]:
        """High-beta Centered Modern Hopfield + Kanerva Sparse-Neighborhood active retrieval."""
        k = max(1, int(top_k))
        with self._lock:
            n = self._n_hot
            if n == 0 or (key_vec is None and not (query.strip() or entity.strip())):
                return []
            self._sync_centering()
            ent_sids = self._active_by_entity.get(_norm_key(entity)) if entity else None
            q_ent = "" if (ent_sids and query.strip()) else entity
            q_hat = self._center_unit(self._coerce_or_encode(query, q_ent, "", key_vec))
            pool_cap = max(32, k * 4)
            if ent_sids:
                row_arr = np.asarray([self._slot_to_row[s] for s in ent_sids if s in self._slot_to_row], dtype=np.int32)
                if row_arr.size == 0:
                    return []
                sims = self._centered_sims(q_hat, row_arr)
                loc = np.argpartition(sims, -pool_cap)[-pool_cap:] if row_arr.size > pool_cap else None
                pool_rows, pool_sims = (row_arr[loc], sims[loc]) if loc is not None else (row_arr, sims)
            else:
                sims = self._centered_sims(q_hat)
                pool_rows = np.argpartition(sims, -pool_cap)[-pool_cap:] if n > pool_cap else np.arange(n, dtype=np.int32)
                pool_sims = sims[pool_rows] if n > pool_cap else sims
            mx = max(float(pool_sims.max()), _NULL_TAU)
            ex = np.exp(np.clip(self.beta * (pool_sims - mx), -60.0, 0.0))
            probs = ex / (float(ex.sum()) + math.exp(max(-60.0, self.beta * (_NULL_TAU - mx))) + 1e-12)
            ranked = sorted(
                ((float(probs[i]), self._meta[int(pool_rows[i])][8], self._meta[int(pool_rows[i])][0], int(pool_rows[i])) for i in range(pool_rows.size)),
                key=lambda x: (round(x[0], 9), x[1], x[2]), reverse=True,
            )
            return [EpisodicHit(r[0], r[1], r[3], r[5], sc, r[6], r[7], r[8], STATUS_ACTIVE) for sc, _, _, idx in ranked[:k] for r in (self._meta[idx],)]

    def format_ctx(self, query: str, top_k: int = 2, max_chars: int = 240) -> str:
        parts = [f"[{h.entity}.{h.attribute}] {h.text}" if (h.entity and h.attribute) else (f"[{h.entity}] {h.text}" if h.entity else h.text) for h in self.retrieve(query, top_k=top_k) if h.score >= 0.18]
        joined = " | ".join(parts)
        return joined[:max_chars] if len(joined) > max_chars else joined

    def export_high_da_episodes(self, min_da: float = 0.7, limit: int = 256) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT slot_id, entity, attribute, text, surprise, da_reward, ts, status FROM episodic_slots "
                "WHERE status=? AND da_reward >= ? ORDER BY da_reward DESC, surprise DESC, ts DESC LIMIT ?;",
                (STATUS_ACTIVE, float(min_da), max(1, int(limit))),
            ).fetchall()
        return [dict(zip(_ROW_KEYS, (int(r[0]), str(r[1]), str(r[2]), str(r[3]), float(r[4]), float(r[5]), float(r[6]), str(r[7])))) for r in rows]

    def get_audit_trail(self, entity: str, attribute: str = "", limit: int = 32) -> list[dict[str, Any]]:
        ent_n, attr_n = _norm_key(entity), _norm_key(attribute)
        sql = "SELECT slot_id, entity, attribute, text, surprise, da_reward, ts, status, superseded_by FROM episodic_slots WHERE entity_norm=?" + (" AND attribute_norm=? ORDER BY slot_id ASC LIMIT ?;" if attr_n else " ORDER BY slot_id ASC LIMIT ?;")
        params: tuple[Any, ...] = (ent_n, attr_n, max(1, int(limit))) if attr_n else (ent_n, max(1, int(limit)))
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(zip(_ROW_KEYS + ("superseded_by",), (int(r[0]), str(r[1]), str(r[2]), str(r[3]), float(r[4]), float(r[5]), float(r[6]), str(r[7]), int(r[8])))) for r in rows]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            self._sync_centering()
            arr_b = self._raw_keys.nbytes + self._inv_norms.nbytes + self._dot_buf.nbytes + self._key_sum.nbytes + self._mu_cached.nbytes
            meta_b = sys.getsizeof(self._meta) + sum(sys.getsizeof(r) + len(r[1]) + len(r[3]) + len(r[5]) for r in self._meta)
            idx_b = sys.getsizeof(self._slot_to_row) + sys.getsizeof(self._active_by_entity) + sys.getsizeof(self._active_by_ent_attr) + sum(sys.getsizeof(v) for v in self._active_by_entity.values()) + sum(sys.getsizeof(v) for v in self._active_by_ent_attr.values())
            ram_b = int(arr_b + meta_b + idx_b)
            return {"active_slots": self._n_hot, "superseded_slots": self._superseded_count, "total_slots": self._n_hot + self._superseded_count, "max_slots": self.max_slots, "dim": self.dim, "beta": self.beta, "db_path": str(self.db_path), "mean_norm": round(math.sqrt(float(np.dot(self._mu_cached, self._mu_cached))), 6), "ram_bytes": ram_b, "ram_mb": round(ram_b / (1024.0 * 1024.0), 3)}
