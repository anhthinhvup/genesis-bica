"""
Global Latent Workspace Engineering Blackboard + Grounded 4-Channel Active Inference EFE
(`mel.v6_lm.global_workspace_efe`, Spec #90, <= 380 LOC).
Unifies Left Hemisphere, Hippocampus (HippoRAG-2 + FastEpisodicMemory), Right Hemisphere
(RustV6Bridge / GenesisDeepReasoner), and Damasio Hardware Homeostasis.
100% CPU offline | 0% re/torch/sklearn | 0% hardcoded NL templates | 0 B on C:\\.
"""
from __future__ import annotations

import ast, math, operator, time, unicodedata
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from ._env_guard import enforce_e_drive_environment
from .homeostatic_interoceptor import HomeostaticInteroceptor, HomeostaticState

enforce_e_drive_environment()

STATES: tuple[str, ...] = ("EXACT_FORMAL", "GROUNDED_FACT", "MULTI_HOP_NEEDED", "OPEN_CREATIVE", "OOD_UNKNOWABLE")
POLICIES: tuple[str, ...] = ("LEFT_EXACT", "HIPPO_RAG_R2", "FAST_S1_R1", "DEEP_TREE_SEM_CHECK", "ASK_CLARIFY_ABSTAIN")
OUTCOMES: tuple[str, ...] = ("VERIFIED_ANSWER", "ABSTAIN", "HALLUCINATION")

_POLICY_LATENCY: dict[str, float] = {
    "LEFT_EXACT": 0.05, "FAST_S1_R1": 0.14, "HIPPO_RAG_R2": 0.32,
    "DEEP_TREE_SEM_CHECK": 0.78, "ASK_CLARIFY_ABSTAIN": 0.04,
}

# P(o | s, pi): rows=STATES, cols=(VERIFIED_ANSWER, ABSTAIN, HALLUCINATION)
_LIKELIHOOD: dict[str, tuple[tuple[float, float, float], ...]] = {
    "LEFT_EXACT": ((0.98, 0.01, 0.01), (0.03, 0.27, 0.70), (0.02, 0.28, 0.70), (0.02, 0.28, 0.70), (0.01, 0.24, 0.75)),
    "HIPPO_RAG_R2": ((0.45, 0.05, 0.50), (0.94, 0.03, 0.03), (0.90, 0.04, 0.06), (0.70, 0.10, 0.20), (0.05, 0.07, 0.88)),
    "FAST_S1_R1": ((0.35, 0.05, 0.60), (0.91, 0.04, 0.05), (0.32, 0.06, 0.62), (0.89, 0.05, 0.06), (0.04, 0.04, 0.92)),
    "DEEP_TREE_SEM_CHECK": ((0.65, 0.10, 0.25), (0.88, 0.07, 0.05), (0.80, 0.12, 0.08), (0.75, 0.10, 0.15), (0.04, 0.85, 0.11)),
    "ASK_CLARIFY_ABSTAIN": ((0.02, 0.97, 0.01),) * 5,
}

_PUNCT_TRANS = str.maketrans({c: " " for c in ".,!?;:'\"()[]{}<>/*+=#@&$%^~`|\\_\t\r\n"})
_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
}


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


def _norm_tokens(text: str) -> list[str]:
    s = unicodedata.normalize("NFC", str(text or "")).lower().translate(_PUNCT_TRANS)
    return [w for w in s.split() if len(w) >= 2]


def _extract_numbers(text: str) -> set[str]:
    out, s = set(), unicodedata.normalize("NFC", str(text or ""))
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if ch.isdigit() or (ch == "-" and i + 1 < n and s[i + 1].isdigit() and (i == 0 or not s[i - 1].isalnum())):
            j, dot = i + 1, False
            while j < n and (s[j].isdigit() or (s[j] == "." and not dot and j + 1 < n and s[j + 1].isdigit())):
                dot, j = (dot or s[j] == "."), j + 1
            raw = s[i:j]
            try:
                val = Fraction(raw)
                out.add(str(val.numerator) if val.denominator == 1 else f"{float(val):.6g}")
            except Exception:
                out.add(raw)
            i = j
        else:
            i += 1
    return out


def _extract_anchors(text: str) -> tuple[set[str], set[str]]:
    """Extracts proper/capitalized entity anchors & content tokens (0% regex, 0% NL lists)."""
    words = [rw.strip(".,!?;:'\"()[]{}<>") for rw in unicodedata.normalize("NFC", str(text or "")).strip().split()]
    words = [w for w in words if len(w) >= 2]
    proper, content = set(), set()
    for idx, cl in enumerate(words):
        low = cl.lower()
        has_alpha, has_dig = any(c.isalpha() for c in cl), any(c.isdigit() for c in cl)
        is_cap = cl[0].isupper() and has_alpha
        next_cap = (idx + 1 < len(words)) and words[idx + 1][0].isupper() and any(c.isalpha() for c in words[idx + 1])
        if (has_alpha and has_dig) or (cl.isupper() and len(cl) >= 2) or (idx > 0 and is_cap) or (idx == 0 and is_cap and next_cap):
            proper.add(low)
        if len(low) >= 4 and low.isalpha():
            content.add(low)
    if not proper and words and words[0][0].isupper() and any(c.isalpha() for c in words[0]):
        proper.add(words[0].lower())
    return proper, content


def _eval_exact_arith(prompt: str) -> str | None:
    """Pure AST exact rational evaluator for arithmetic expressions (0% eval, 0% regex)."""
    s = unicodedata.normalize("NFC", str(prompt or "")).strip().rstrip("?.!")
    if not s or not any(c.isdigit() for c in s):
        return None
    rep = s.replace("^", "**").replace("\u00d7", "*").replace("\u00f7", "/").replace("\u2212", "-")
    spans = "".join(ch if ch in "0123456789+-*/().% " else "|" for ch in rep).split("|")
    best = max((sp.strip() for sp in spans), key=len, default="")
    ops_strong = sum(1 for c in best if c in "+*/%")
    if not best or sum(1 for c in best if c.isdigit()) < 2 or (ops_strong + best.count("-")) < 1:
        return None
    if ops_strong == 0 and (len(best.replace(" ", "")) / max(1, len(rep.replace(" ", "")))) < 0.45 and "=" not in rep:
        return None
    try:
        node = ast.parse(best, mode="eval").body
    except SyntaxError:
        return None

    def _ev(nd: ast.AST) -> Fraction:
        if isinstance(nd, ast.Constant) and isinstance(nd.value, (int, float)):
            return Fraction(str(nd.value))
        if isinstance(nd, ast.UnaryOp) and isinstance(nd.op, (ast.UAdd, ast.USub)):
            v = _ev(nd.operand)
            return v if isinstance(nd.op, ast.UAdd) else -v
        if isinstance(nd, ast.BinOp) and type(nd.op) in _BIN_OPS:
            return _BIN_OPS[type(nd.op)](_ev(nd.left), _ev(nd.right))
        if isinstance(nd, ast.BinOp) and isinstance(nd.op, ast.Pow):
            lv, rv = _ev(nd.left), _ev(nd.right)
            if rv.denominator == 1 and abs(rv.numerator) <= 20:
                return lv ** rv.numerator
        raise ValueError("unsupported")

    try:
        res = _ev(node)
    except Exception:
        return None
    return str(res.numerator) if res.denominator == 1 else f"{float(res):.10g}"


def _branches_equivalent(a: str, b: str) -> bool:
    na, nb = " ".join(_norm_tokens(a)), " ".join(_norm_tokens(b))
    if na == nb:
        return True
    nums_a, nums_b = _extract_numbers(a), _extract_numbers(b)
    if (nums_a or nums_b) and nums_a != nums_b:
        return False
    ta, tb = set(na.split()), set(nb.split())
    if not ta or not tb:
        return False
    inter = len(ta & tb)
    return (inter / max(1, min(len(ta), len(tb)))) >= 0.50 and (inter / max(1, len(ta | tb))) >= 0.35


def compute_semantic_entropy(branches: list[str]) -> tuple[float, list[list[str]], str]:
    """Farquhar et al. (Nature 2024) normalized Semantic Entropy in [0.0, 1.0] over K outputs."""
    cleaned = [str(b or "").strip() for b in branches if str(b or "").strip()]
    if not cleaned:
        return 1.0, [], ""
    clusters: list[list[str]] = []
    for txt in cleaned:
        for cl in clusters:
            if _branches_equivalent(txt, cl[0]):
                cl.append(txt)
                break
        else:
            clusters.append([txt])
    k = len(cleaned)
    ent = max(0.0, -sum((len(cl) / k) * math.log(len(cl) / k) for cl in clusters if cl))
    best_cl = max(clusters, key=lambda c: (len(c), -len(c[0])))
    return round(_clamp01(ent / math.log(max(2, k))), 4), clusters, min(best_cl, key=len)


@dataclass
class UncertaintySignals:
    c_vault: float; c_sheaf: float; cert_left: str = "NONE"; h_sem: float | None = None
    episodic_score: float = 0.0; hippo_coverage: float = 0.0
    multi_hop_hint: bool = False; open_creative_hint: bool = False


@dataclass
class ArbitrationResult:
    policy: str; efe_scores: dict[str, float]; belief_q: dict[str, float]; p_pref: dict[str, float]
    lambda_cost: float; think_steps: int; branch_k: int; arbitration_ms: float


@dataclass
class WorkspaceDecision:
    query: str; policy: str; answer: str; abstained: bool; route: str
    signals: UncertaintySignals; arbitration: ArbitrationResult; homeostasis: HomeostaticState
    ctx_text: str = ""; branches: list[str] = field(default_factory=list); total_ms: float = 0.0


def _infer_belief_q(sig: UncertaintySignals, st: HomeostaticState) -> tuple[float, ...]:
    """Maps 4 grounded channels (c_vault, c_sheaf, cert_left, h_sem) into Q(s) over STATES."""
    if sig.cert_left == "PROVED":
        return (0.96, 0.01, 0.01, 0.01, 0.01)
    if sig.cert_left in ("DISPROVED", "CONTRADICTION"):
        return (0.01, 0.02, 0.02, 0.01, 0.94)
    cv, cs = _clamp01(sig.c_vault), _clamp01(sig.c_sheaf)
    if sig.h_sem is not None:
        hs = _clamp01(sig.h_sem)
        if hs >= 0.50:
            p_ood = min(0.94, 0.55 + 0.42 * hs + 0.15 * (1.0 - cv))
            rem = 1.0 - p_ood
            return (0.01 * rem, 0.45 * rem, 0.45 * rem, 0.09 * rem, p_ood)
        p_gr, p_mh = min(0.88, 0.48 + 0.35 * (1.0 - hs) + 0.15 * cv), min(0.32, 0.15 + 0.15 * st.ne)
        p_ood = max(0.03, 0.97 - p_gr - p_mh)
        tot = 0.03 + p_gr + p_mh + p_ood
        return (0.01 / tot, p_gr / tot, p_mh / tot, 0.02 / tot, p_ood / tot)
    if cv < 0.15:
        if sig.open_creative_hint and cs >= 0.45:
            return (0.01, 0.05, 0.04, 0.82, 0.08)
        p_ood = min(0.95, 0.84 + 0.10 * (1.0 - cv / 0.15))
        rem = 1.0 - p_ood
        return (0.01 * rem, 0.35 * rem, 0.35 * rem, 0.29 * rem, p_ood)
    if cv < 0.65:
        g_str = 0.65 * cv + 0.35 * cs
        if g_str >= 0.46 and cs >= 0.45:
            mh = 0.55 if (sig.multi_hop_hint or st.ne >= 0.45) else 0.38
            p_f, p_m, p_o = (1.0 - mh) * 0.82, mh * 0.82, max(0.06, 0.14 - 0.10 * (g_str - 0.46))
            tot = 0.04 + p_f + p_m + p_o
            return (0.01 / tot, p_f / tot, p_m / tot, 0.03 / tot, p_o / tot)
        p_ood = _clamp01(0.50 - 0.35 * (cv - 0.15))
        rem = 0.96 - p_ood
        return (0.01, 0.45 * rem, 0.55 * rem, 0.03, p_ood)
    mh = 0.42 if (sig.multi_hop_hint or st.ne >= 0.55) else (0.03 if (cv >= 0.85 and st.ne < 0.45) else 0.12)
    p_ood = max(0.02, 0.15 * (1.0 - cs))
    rem = 0.97 - p_ood
    return (0.01, (1.0 - mh) * rem, mh * rem, 0.02, p_ood)


def _compute_p_pref(st: HomeostaticState) -> tuple[float, float, float]:
    p_hall = 0.015
    p_abs = max(0.10, min(0.30, 0.19 + 0.06 * st.contradiction_rate + 0.04 * (1.0 - st.serotonin) - 0.04 * st.da))
    p_ver = max(0.62, 1.0 - p_abs - p_hall)
    tot = p_ver + p_abs + p_hall
    return (p_ver / tot, p_abs / tot, p_hall / tot)


def _estimate_sheaf_consistency(query: str, ctx_text: str) -> float:
    if not ctx_text.strip():
        return 0.50
    q_prop, q_cont = _extract_anchors(query)
    anchors = q_prop | q_cont
    if not anchors:
        return 0.60
    ctx_low, ctx_toks = unicodedata.normalize("NFC", ctx_text).lower(), set(_norm_tokens(ctx_text))
    ratio = sum(1 for a in anchors if (a in ctx_toks or a in ctx_low)) / len(anchors)
    q_yrs = {n for n in _extract_numbers(query) if len(n) == 4 and n.isdigit()}
    c_yrs = {n for n in _extract_numbers(ctx_text) if len(n) == 4 and n.isdigit()}
    pen = 0.35 if (q_yrs and c_yrs and not (q_yrs & c_yrs)) else 0.0
    return _clamp01(0.25 + 0.75 * ratio - pen)


class GlobalWorkspaceEFE:
    """Engineering Blackboard uniting Left, Hippocampus, Right & Homeostasis via Active Inference EFE."""

    def __init__(self, brain: Any = None, hippo: Any = None, episodic: Any = None,
                 rust_engine: Any = None, reasoner: Any = None,
                 interoceptor: HomeostaticInteroceptor | None = None) -> None:
        self.reasoner = reasoner
        self.brain = brain or getattr(reasoner, "brain", None)
        self.rust_engine = rust_engine or getattr(reasoner, "rust", None)
        self.hippo, self.episodic = hippo, episodic
        self.interoceptor = interoceptor or HomeostaticInteroceptor()

    def arbitrate(self, signals: UncertaintySignals, state: HomeostaticState | None = None) -> ArbitrationResult:
        """Computes G(pi) = KL(Q(o|pi)||P_pref) + E[H(P(o|s,pi))] + lambda_cost * Latency (< 1 ms)."""
        t0 = time.perf_counter()
        st = state if state is not None else self.interoceptor.sample()
        q_s, p_pref = _infer_belief_q(signals, st), _compute_p_pref(st)
        lam = max(0.08, 0.24 + (0.0 if st.ac_plugged else 0.42) + 0.18 * (1.0 - st.serotonin) - 0.12 * st.ne)
        efe: dict[str, float] = {}
        for pi in POLICIES:
            rows, q_o, amb = _LIKELIHOOD[pi], [0.0, 0.0, 0.0], 0.0
            for si, qs in enumerate(q_s):
                r0, r1, r2 = rows[si]
                q_o[0] += qs * r0; q_o[1] += qs * r1; q_o[2] += qs * r2
                amb += qs * (-sum(p * math.log(max(1e-9, p)) for p in (r0, r1, r2)))
            kl = sum(q_o[j] * math.log(max(1e-9, q_o[j]) / p_pref[j]) for j in range(3) if q_o[j] > 1e-12)
            efe[pi] = round(kl + amb + lam * _POLICY_LATENCY[pi], 5)
        best_pi = min(POLICIES, key=lambda p: efe[p])
        r_steps = 0 if best_pi in ("LEFT_EXACT", "ASK_CLARIFY_ABSTAIN") else (1 if (best_pi == "FAST_S1_R1" and st.ne < 0.60) else 2)
        k_br = self.interoceptor.recommended_branch_k(4) if best_pi == "DEEP_TREE_SEM_CHECK" else 1
        return ArbitrationResult(
            best_pi, efe, {s: round(v, 4) for s, v in zip(STATES, q_s)},
            {o: round(v, 4) for o, v in zip(OUTCOMES, p_pref)}, round(lam, 4),
            r_steps, k_br, round((time.perf_counter() - t0) * 1000.0, 4),
        )

    def _try_left(self, query: str) -> tuple[str, str, str]:
        arith = _eval_exact_arith(query)
        if arith is not None:
            return arith, "LEFT:AST_ARITH", "PROVED"
        if self.brain is not None and hasattr(self.brain, "_try_left_hemisphere"):
            res = self.brain._try_left_hemisphere(query)
            if res is not None:
                ans, tag = res
                return ans, tag, ("DISPROVED" if "ABSTAIN" in tag else "PROVED")
        return "", "", "NONE"

    def _query_hippocampus(self, query: str, st: HomeostaticState) -> tuple[float, float, str, str, bool]:
        epi_score, epi_ctx, q_low = 0.0, "", unicodedata.normalize("NFC", query).lower()
        q_toks = set(_norm_tokens(query))
        if self.episodic is not None and hasattr(self.episodic, "retrieve"):
            try:
                for h in self.episodic.retrieve(query, top_k=3):
                    if getattr(h, "status", "ACTIVE") != "ACTIVE":
                        continue
                    htxt, hent = str(getattr(h, "text", "")).strip(), str(getattr(h, "entity", "")).strip().lower()
                    htoks = set(_norm_tokens(htxt + " " + hent))
                    ov = len(q_toks & htoks) / max(1, min(len(q_toks), len(htoks))) if (q_toks and htoks) else 0.0
                    ent_hit, raw_sc = bool(hent and hent in q_low), float(getattr(h, "score", 0.0))
                    if ent_hit or ov >= 0.34 or (raw_sc >= 0.65 and ov >= 0.20):
                        sc = _clamp01(max(raw_sc, 0.72 if ent_hit else ov) * (0.88 + 0.24 * st.ach))
                        if sc > epi_score:
                            epi_score, epi_ctx = sc, htxt
            except Exception:
                pass
        hippo_cov, hippo_ctx, mh_hint = 0.0, "", False
        if self.hippo is not None and hasattr(self.hippo, "retrieve"):
            try:
                hres = self.hippo.retrieve(query, top_k=3, max_ctx_chars=360)
                hippo_cov = _clamp01(float(getattr(hres, "evidence_coverage", 0.0)))
                hippo_ctx = str(getattr(hres, "ctx_text", "")).strip()
                mh_hint = len(getattr(hres, "matched_entities", [])) >= 2
            except Exception:
                pass
        elif self.brain is not None and getattr(self.brain, "vault", None) is not None:
            q_prop, q_cont = _extract_anchors(query)
            q_ent = " ".join(sorted(q_prop, key=len, reverse=True)) if q_prop else ""
            hits = (self.brain.vault.retrieve(q_ent, top_k=2, max_ctx_chars=360) if q_ent else []) or self.brain.vault.retrieve(query, top_k=2, max_ctx_chars=360)
            if hits:
                hippo_ctx = " | ".join(f"{h['title']}: {h['snippet']}" if h.get("title") else h["snippet"] for h in hits)[:360]
                ctx_low, t_low = unicodedata.normalize("NFC", hippo_ctx).lower(), " ".join(str(h.get("title", "")).lower() for h in hits)
                if q_prop:
                    pc = sum(1 for p in q_prop if p in ctx_low) / len(q_prop)
                    tc = sum(1 for p in q_prop if p in t_low) / len(q_prop)
                    cc = (sum(1 for c in q_cont if c in ctx_low) / len(q_cont)) if q_cont else pc
                    hippo_cov = _clamp01(0.65 * pc + 0.20 * tc + 0.15 * cc) if (pc >= 0.50 and (tc > 0.0 or pc >= 0.99)) else _clamp01(0.30 * pc)
                elif q_cont:
                    hippo_cov = _clamp01(sum(1 for c in q_cont if c in ctx_low) / len(q_cont))
        return epi_score, hippo_cov, epi_ctx, hippo_ctx, mh_hint

    def process(self, query: str, execute_right: bool = True, max_tokens: int = 64, allow_sem_check: bool = True) -> WorkspaceDecision:
        t0, st = time.perf_counter(), self.interoceptor.sample()
        left_ans, left_tag, cert_left = self._try_left(query)
        epi_sc, hippo_cov, epi_ctx, hippo_ctx, mh_hint = self._query_hippocampus(query, st) if cert_left == "NONE" else (0.0, 0.0, "", "", False)
        use_hippo = bool(hippo_ctx) and (epi_sc < 0.75)
        ctx_text = " | ".join(p for p in (epi_ctx, hippo_ctx if use_hippo else "") if p)[:420]
        c_vault = 1.0 if cert_left == "PROVED" else _clamp01(max(hippo_cov, epi_sc) + 0.08 * min(hippo_cov, epi_sc))
        c_sheaf = 1.0 if cert_left == "PROVED" else (0.0 if cert_left == "DISPROVED" else _estimate_sheaf_consistency(query, ctx_text))
        if cert_left == "NONE" and c_vault >= 0.35 and c_sheaf < 0.18:
            cert_left = "CONTRADICTION"
        sig = UncertaintySignals(c_vault, c_sheaf, cert_left, None, epi_sc, hippo_cov, mh_hint, False)
        arb, branches = self.arbitrate(sig, state=st), []
        full_q = f"<CTX> {ctx_text} <USER> {query} <BOT>" if ctx_text else f"<USER> {query} <BOT>"
        if arb.policy == "LEFT_EXACT" and left_ans:
            return WorkspaceDecision(query, arb.policy, left_ans, False, left_tag, sig, arb,
                                     self.interoceptor.update_feedback(True, False, 0.05), ctx_text, [], round((time.perf_counter() - t0) * 1000.0, 2))
        if arb.policy == "DEEP_TREE_SEM_CHECK" and allow_sem_check and self.rust_engine is not None:
            branches, _ = self.rust_engine.expand_branches(full_q, n_branches=arb.branch_k, max_new=min(48, max_tokens), base_temp=0.55, top_p=0.90, think_steps=2, seed=42)
            sig.h_sem, _, consensus = compute_semantic_entropy(branches)
            arb = self.arbitrate(sig, state=st)
            if arb.policy != "ASK_CLARIFY_ABSTAIN" and sig.h_sem < 0.55 and consensus:
                return WorkspaceDecision(query, "DEEP_TREE_SEM_CHECK", consensus, False, f"GWT:DEEP_SEM_K{len(branches)}", sig, arb,
                                         self.interoceptor.update_feedback(sig.h_sem <= 0.25, False, sig.h_sem), ctx_text, branches, round((time.perf_counter() - t0) * 1000.0, 2))
            arb.policy = "ASK_CLARIFY_ABSTAIN"
        unresolved_ambig = (arb.policy == "DEEP_TREE_SEM_CHECK") and (not ctx_text or c_vault < 0.45 or c_sheaf < 0.50)
        if arb.policy == "ASK_CLARIFY_ABSTAIN" or unresolved_ambig:
            tag = cert_left if cert_left != "NONE" else ("UNRESOLVED_AMBIGUITY" if unresolved_ambig else "OOD_LOW_EVIDENCE")
            h_str = f"{sig.h_sem:.2f}" if sig.h_sem is not None else "NA"
            ans = f"[ABSTAIN:{tag}] c_vault={c_vault:.2f} c_sheaf={c_sheaf:.2f} h_sem={h_str}"
            st_n = self.interoceptor.update_feedback(False, cert_left in ("DISPROVED", "CONTRADICTION"), max(0.70, sig.h_sem or 0.70))
            return WorkspaceDecision(query, "ASK_CLARIFY_ABSTAIN", ans, True, "GWT:ASK_CLARIFY_ABSTAIN", sig, arb, st_n, ctx_text, branches, round((time.perf_counter() - t0) * 1000.0, 2))
        if epi_sc >= 0.65 and epi_ctx and (epi_sc >= 0.85 or not execute_right or self.rust_engine is None):
            ans, route = epi_ctx, "GWT:EPISODIC_ACTIVE"
        elif execute_right and self.rust_engine is not None:
            r_use = max(1, arb.think_steps)
            ans, _ = self.rust_engine.generate_one(full_q, max_new=max_tokens, temp=0.35, think_steps=r_use, seed=42)
            route = f"GWT:{arb.policy}:R{r_use}"
        else:
            ans, route = (epi_ctx or hippo_ctx[:260]), f"GWT:{arb.policy}:GROUNDED_CTX"
        st_n = self.interoceptor.update_feedback(c_vault >= 0.60 and c_sheaf >= 0.60, False, max(0.05, 1.0 - c_vault))
        return WorkspaceDecision(query, arb.policy, ans, False, route, sig, arb, st_n, ctx_text, branches, round((time.perf_counter() - t0) * 1000.0, 2))
