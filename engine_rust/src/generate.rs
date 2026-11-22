//! Genesis-v6.5 O(1)-State Streaming Inference with Recurrent Depth (`R=1..16`) & PEER (`generate.rs`).
//! Zero external crates | Strictly <= 260 LOC.

use crate::attn_rope::{make_inv_freq, AttnKvState};
use crate::bpe::{ByteBpeTokenizer, TOK_BOS, TOK_EOS, TOK_USER};
use crate::gv8_kernels::lm_head_softcap_fast;
use crate::model::{rand_uniform, GenesisV6Model, InferenceState, LayerState, ModelConfig, PlanStep};
use crate::simd_avx2::{dot_f32, rmsnorm_fwd};

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[inline]
pub fn relu2(x: f32) -> f32 {
    if x > 0.0 { let c = x.min(6.0); c * c } else { 0.0 }
}

#[inline]
pub fn logit_softcap(z: f32) -> f32 {
    30.0 * (z / 30.0).tanh()
}

#[derive(Clone)]
pub struct StepScratch {
    pub x: Vec<f32>,
    pub s0: Vec<f32>,
    pub xn: Vec<f32>,
    pub xc: Vec<f32>,
    pub z: Vec<f32>,
    pub hc: Vec<f32>,
    pub yo: Vec<f32>,
    pub u: Vec<f32>,
    pub up: Vec<f32>,
    pub dn: Vec<f32>,
    pub r_logits: Vec<f32>,
    pub xf: Vec<f32>,
    // Preallocated zero-alloc Attention scratch (7 buffers incl. precomputed RoPE inv_freq)
    pub attn_xn: Vec<f32>,
    pub attn_q: Vec<f32>,
    pub attn_k: Vec<f32>,
    pub attn_v: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub attn_scores: Vec<f32>,
    pub attn_inv_freq: Vec<f32>,
    // Preallocated zero-alloc PEER scratch (4 buffers)
    pub peer_xn: Vec<f32>,
    pub peer_q: Vec<f32>,
    pub peer_sa: Vec<f32>,
    pub peer_sb: Vec<f32>,
    // Fixed stack trace of routed PEER expert IDs from the most recent token step (0 heap alloc)
    pub last_peer_ids: [[u32; 64]; 2],
    pub last_peer_counts: [usize; 2],
}

impl StepScratch {
    pub fn from_cfg(cfg: &ModelConfig) -> Self {
        let d = cfg.d_model;
        let df = cfg.d_ff;
        let e = cfg.n_experts;
        let has_attn = cfg.attn_window > 0;
        let aw = if has_attn { cfg.attn_window.max(8) } else { 0 };
        let nh = cfg.peer_heads.max(1);
        let hd = d / nh;
        let attn_inv_freq = if has_attn { make_inv_freq(hd) } else { Vec::new() };
        let sk = cfg.peer_sub_k;
        Self {
            x: vec![0.0; d], s0: vec![0.0; d], xn: vec![0.0; d], xc: vec![0.0; d],
            z: vec![0.0; d], hc: vec![0.0; d], yo: vec![0.0; d], u: vec![0.0; d],
            up: vec![0.0; df], dn: vec![0.0; d], r_logits: vec![0.0; e], xf: vec![0.0; d],
            attn_xn: vec![0.0; if has_attn { d } else { 0 }],
            attn_q: vec![0.0; if has_attn { d } else { 0 }],
            attn_k: vec![0.0; if has_attn { d } else { 0 }],
            attn_v: vec![0.0; if has_attn { d } else { 0 }],
            attn_out: vec![0.0; if has_attn { d } else { 0 }],
            attn_scores: vec![0.0; aw],
            attn_inv_freq,
            peer_xn: vec![0.0; if sk > 0 { d } else { 0 }],
            peer_q: vec![0.0; if sk > 0 { d } else { 0 }],
            peer_sa: vec![0.0; sk],
            peer_sb: vec![0.0; sk],
            last_peer_ids: [[0u32; 64]; 2],
            last_peer_counts: [0usize; 2],
        }
    }

    pub fn new(model: &GenesisV6Model) -> Self {
        Self::from_cfg(&model.cfg)
    }
}

pub fn select_topk_experts(r_logits: &[f32], top_k: usize, out_idx: &mut [usize], out_w: &mut [f32]) {
    let n = r_logits.len();
    let k = top_k.min(n);
    let mut used = [false; 32];
    let mut max_v = f32::NEG_INFINITY;
    for j in 0..k {
        let mut best_i = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for i in 0..n {
            if !used[i] && r_logits[i] > best_v {
                best_v = r_logits[i];
                best_i = i;
            }
        }
        used[best_i] = true;
        out_idx[j] = best_i;
        out_w[j] = best_v;
        if best_v > max_v { max_v = best_v; }
    }
    let mut sum_e = 0.0f32;
    for j in 0..k {
        out_w[j] = (out_w[j] - max_v).exp();
        sum_e += out_w[j];
    }
    let inv = 1.0 / sum_e.max(1e-12);
    for j in 0..k { out_w[j] *= inv; }
}

#[inline]
fn step_token_core(
    model: &GenesisV6Model,
    slots: &mut [LayerState],
    attn_kvs: &mut [AttnKvState],
    plan: &[PlanStep],
    sc: &mut StepScratch,
    tok_id: u16,
    logits_out: &mut [f32],
) {
    let d = model.cfg.d_model;
    let df = model.cfg.d_ff;
    let ne = model.cfg.n_experts;
    let tk = model.cfg.top_k;
    let tid = (tok_id as usize).min(model.cfg.vocab - 1);
    sc.x.copy_from_slice(&model.emb[tid * d..(tid + 1) * d]);
    sc.last_peer_counts = [0, 0];

    let mut top_idx = [0usize; 8];
    let mut top_w = [0.0f32; 8];

    for (step_i, ps) in plan.iter().enumerate() {
        if ps.is_inj {
            for i in 0..d { sc.x[i] += model.recur_inj_alpha[i] * sc.s0[i]; }
        }
        let layer = &model.layers[ps.layer_idx];
        let ls = &mut slots[ps.slot_idx];
        let _ = rmsnorm_fwd(&sc.x, &layer.norm1, &mut sc.xn);
        for i in 0..d {
            sc.xc[i] = layer.conv_w[i] * ls.prev2[i]
                + layer.conv_w[d + i] * ls.prev1[i]
                + layer.conv_w[2 * d + i] * sc.xn[i];
        }
        ls.prev2.copy_from_slice(&ls.prev1);
        ls.prev1.copy_from_slice(&sc.xn);

        layer.wz.matvec(d, d, &sc.xc, &mut sc.z);
        layer.wh.matvec(d, d, &sc.xc, &mut sc.hc);
        for i in 0..d {
            let zi = sigmoid(sc.z[i]);
            let hi = sc.hc[i].tanh();
            ls.h[i] = (1.0 - zi) * ls.h[i] + zi * hi;
        }
        layer.wo.matvec(d, d, &ls.h, &mut sc.yo);
        for i in 0..d { sc.x[i] += sc.yo[i]; }

        let _ = rmsnorm_fwd(&sc.x, &layer.norm2, &mut sc.u);
        for e in 0..ne { sc.r_logits[e] = dot_f32(&layer.router[e * d..(e + 1) * d], &sc.u); }
        select_topk_experts(&sc.r_logits, tk, &mut top_idx, &mut top_w);
        for j in 0..tk {
            let e = top_idx[j];
            let p = top_w[j];
            let eu = &layer.exp_up[e];
            let ed = &layer.exp_dn[e];
            eu.matvec(df, d, &sc.u, &mut sc.up);
            for m in 0..df { sc.up[m] = relu2(sc.up[m]); }
            ed.matvec(d, df, &sc.up, &mut sc.dn);
            for i in 0..d { sc.x[i] += p * sc.dn[i]; }
        }
        if let Some(pi) = ps.peer_opt {
            let (r_ids, r_cnt) = if pi < 2 {
                (Some(&mut sc.last_peer_ids[pi]), Some(&mut sc.last_peer_counts[pi]))
            } else {
                (None, None)
            };
            model.peers[pi].forward_step_residual_scratch(
                &mut sc.x, &mut sc.peer_xn, &mut sc.peer_q, &mut sc.peer_sa, &mut sc.peer_sb,
                model.peer_delta.table(pi), r_ids, r_cnt,
            );
        }
        if let (Some(ai), Some(ref attn)) = (ps.attn_opt, &model.attn) {
            attn.forward_step_residual_scratch(
                &mut sc.x, &mut attn_kvs[ai], &mut sc.attn_xn, &mut sc.attn_q, &mut sc.attn_k,
                &mut sc.attn_v, &mut sc.attn_out, &mut sc.attn_scores, &sc.attn_inv_freq,
            );
        }
        if step_i == 0 {
            sc.s0.copy_from_slice(&sc.x);
        }
    }

    let _ = rmsnorm_fwd(&sc.x, &model.norm_f, &mut sc.xf);
    let v = model.cfg.vocab.min(logits_out.len());
    lm_head_softcap_fast(&model.emb, v, d, &sc.xf, logits_out);
}

/// Zero-alloc single-token step using `StepScratch` owned inside `InferenceState`.
pub fn step_token_in_state(
    model: &GenesisV6Model,
    st: &mut InferenceState,
    tok_id: u16,
    logits_out: &mut [f32],
) {
    let InferenceState { slots, attn_kvs, plan, plan_len, scratch, .. } = st;
    step_token_core(model, slots, attn_kvs, &plan[..*plan_len], scratch, tok_id, logits_out);
}

/// Zero-alloc O(1) recurrent step across Prelude -> R x (Core + PEER) -> Coda (with explicit scratch).
pub fn step_token(
    model: &GenesisV6Model,
    st: &mut InferenceState,
    sc: &mut StepScratch,
    tok_id: u16,
    logits_out: &mut [f32],
) {
    step_token_core(model, &mut st.slots, &mut st.attn_kvs, &st.plan[..st.plan_len], sc, tok_id, logits_out);
}

pub fn sample_next_token(
    logits: &mut [f32],
    history: &[u16],
    temp: f32,
    top_k: usize,
    top_p: f32,
    rep_pen: f32,
    rng: &mut u64,
) -> u16 {
    let v = logits.len();
    let start = history.len().saturating_sub(48);
    for &t in &history[start..] {
        let idx = t as usize;
        if idx < v {
            if logits[idx] > 0.0 { logits[idx] /= rep_pen; } else { logits[idx] *= rep_pen; }
        }
    }
    if temp <= 1e-4 {
        let mut best_i = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &lv) in logits.iter().enumerate() {
            if lv > best_v { best_v = lv; best_i = i; }
        }
        return best_i as u16;
    }
    let inv_t = 1.0 / temp;
    let mut pairs: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &x)| (i, x * inv_t)).collect();
    let k = top_k.max(1).min(v);
    pairs.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
    pairs.truncate(k);
    pairs.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

    let max_l = pairs[0].1;
    let mut sum_p = 0.0f32;
    for p in &mut pairs { p.1 = (p.1 - max_l).exp(); sum_p += p.1; }
    let inv_s = 1.0 / sum_p.max(1e-12);
    for p in &mut pairs { p.1 *= inv_s; }
    let mut cum = 0.0f32;
    let mut cut = pairs.len();
    for (idx, p) in pairs.iter().enumerate() {
        cum += p.1;
        if cum >= top_p { cut = idx + 1; break; }
    }
    pairs.truncate(cut.max(1));
    let renorm: f32 = pairs.iter().map(|p| p.1).sum();
    let r = (rand_uniform(rng) * 0.5 + 0.5) * renorm;
    let mut acc = 0.0f32;
    for &(tok, prob) in &pairs {
        acc += prob;
        if acc >= r { return tok as u16; }
    }
    pairs[0].0 as u16
}

pub fn generate_text_with_think(
    model: &GenesisV6Model,
    bpe: &ByteBpeTokenizer,
    prompt: &str,
    max_new: usize,
    temp: f32,
    top_p: f32,
    seed: u64,
    think_steps: usize,
) -> String {
    let mut st = InferenceState::with_think_steps(&model.cfg, think_steps.max(model.cfg.recur_steps.max(1)));
    let mut sc = StepScratch::new(model);
    let mut logits = vec![0.0f32; model.cfg.vocab];
    let mut toks = bpe.encode(prompt);
    if toks.is_empty() { toks.push(TOK_BOS); }
    for &t in &toks { step_token(model, &mut st, &mut sc, t, &mut logits); }
    let mut gen: Vec<u16> = Vec::with_capacity(max_new);
    let mut rng = seed.max(1);
    for _ in 0..max_new {
        let nxt = sample_next_token(&mut logits, &gen, temp, 40, top_p, 1.15, &mut rng);
        if nxt == TOK_EOS || nxt == TOK_USER { break; }
        gen.push(nxt);
        step_token(model, &mut st, &mut sc, nxt, &mut logits);
    }
    bpe.decode(&gen)
}

pub fn generate_text(
    model: &GenesisV6Model,
    bpe: &ByteBpeTokenizer,
    prompt: &str,
    max_new: usize,
    temp: f32,
    top_p: f32,
    seed: u64,
) -> String {
    generate_text_with_think(model, bpe, prompt, max_new, temp, top_p, seed, model.cfg.recur_steps)
}
