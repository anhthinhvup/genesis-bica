//! Genesis-v7 Hybrid Architecture (`model.rs`):
//!   3x BitNet-MinGRU-MoE + 1x Sliding-Window Causal Attention (W=128, RoPE) + 54M DeepMind PEER
//!   + Curriculum Recurrent Depth (`R=1 -> R=2..16`). Supports GV6MOD01, GV6MOD02, and GV6MOD03!
//! Zero external crates | 100% CPU | Strictly <= 375 LOC.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

use crate::attn_rope::{AttnKvState, SlidingWindowAttn};
use crate::generate::StepScratch;
use crate::peer_delta_sidecar::PeerDeltaSidecar;
use crate::peer_pk::PeerPkLayer;
use crate::simd_avx2::quantize_bitnet_absmean;

#[derive(Clone, Copy, Debug)]
pub struct ModelConfig {
    pub vocab: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_experts: usize,
    pub top_k: usize,
    pub d_ff: usize,
    pub peer_sub_k: usize,
    pub peer_heads: usize,
    pub peer_k: usize,
    pub recur_steps: usize,
    pub attn_window: usize, // 0 = pure MinGRU, 128 = Hybrid Griffin/Hymba Causal Attention (RoPE)!
}

impl ModelConfig {
    pub fn num_params(&self) -> usize {
        let emb = self.vocab * self.d_model + 2 * self.d_model;
        let per_layer = 2 * self.d_model + 3 * self.d_model + 3 * self.d_model * self.d_model
            + self.n_experts * self.d_model + self.n_experts * (2 * self.d_ff * self.d_model);
        let peer_per = if self.peer_sub_k > 0 && self.peer_heads > 0 {
            let sd = (self.d_model / self.peer_heads) / 2;
            let ne = self.peer_sub_k * self.peer_sub_k;
            self.d_model + self.d_model * self.d_model + 2 * self.peer_heads * self.peer_sub_k * sd + 2 * ne * self.d_model
        } else { 0 };
        let n_peers = if self.peer_sub_k > 0 { if self.n_layers >= 4 { 2 } else { 1 } } else { 0 };
        let attn_p = if self.attn_window > 0 { self.d_model + 4 * self.d_model * self.d_model } else { 0 };
        emb + self.n_layers * per_layer + n_peers * peer_per + attn_p
    }
}

#[derive(Clone)]
pub struct BitLinear {
    pub w: Vec<f32>,
    pub q: Vec<i8>,
    pub q2: Vec<u8>,
    pub gamma: f32,
}

impl BitLinear {
    pub fn new(rows: usize, cols: usize, rng: &mut u64, scale: f32) -> Self {
        let n = rows * cols;
        let mut w = Vec::with_capacity(n);
        for _ in 0..n { w.push(rand_uniform(rng) * scale); }
        let mut q = vec![0i8; n];
        let gamma = quantize_bitnet_absmean(&w, &mut q);
        Self { w, q, q2: Vec::new(), gamma }
    }
    pub fn from_packed_2bit(q2: Vec<u8>, gamma: f32) -> Self {
        Self { w: Vec::new(), q: Vec::new(), q2, gamma }
    }
    #[inline]
    pub fn refresh_quant(&mut self) { self.gamma = quantize_bitnet_absmean(&self.w, &mut self.q); }
    #[inline]
    pub fn matvec(&self, rows: usize, cols: usize, x: &[f32], y: &mut [f32]) {
        if !self.q2.is_empty() {
            crate::gv8_kernels::matvec_bitnet_2bit(&self.q2, self.gamma, rows, cols, x, y);
        } else {
            crate::simd_avx2::matvec_bitnet(&self.q, self.gamma, rows, cols, x, y);
        }
    }
}

#[derive(Clone)]
pub struct SparseBlock {
    pub norm1: Vec<f32>, pub conv_w: Vec<f32>,
    pub wz: BitLinear, pub wh: BitLinear, pub wo: BitLinear,
    pub norm2: Vec<f32>, pub router: Vec<f32>,
    pub exp_up: Vec<BitLinear>, pub exp_dn: Vec<BitLinear>,
}

#[derive(Clone)]
pub struct GenesisV6Model {
    pub cfg: ModelConfig,
    pub emb: Vec<f32>,
    pub norm_f: Vec<f32>,
    pub recur_inj_alpha: Vec<f32>,
    pub layers: Vec<SparseBlock>,
    pub peers: Vec<PeerPkLayer>,
    pub attn: Option<SlidingWindowAttn>, // Hybrid Griffin/Hymba Causal Attention (W=128/256, RoPE)
    pub peer_delta: PeerDeltaSidecar,    // Live sparse f32 PEER delta overlay (Spec #91)
}

pub const MAX_PLAN_LEN: usize = 64;

#[derive(Clone, Copy, Debug, Default)]
pub struct PlanStep {
    pub layer_idx: usize,
    pub slot_idx: usize,
    pub peer_opt: Option<usize>,
    pub attn_opt: Option<usize>,
    pub is_inj: bool,
}

#[derive(Clone)]
pub struct LayerState {
    pub h: Vec<f32>, pub prev1: Vec<f32>, pub prev2: Vec<f32>,
}

impl LayerState {
    pub fn new(d: usize) -> Self { Self { h: vec![0.0; d], prev1: vec![0.0; d], prev2: vec![0.0; d] } }
    pub fn reset(&mut self) { self.h.fill(0.0); self.prev1.fill(0.0); self.prev2.fill(0.0); }
}

pub fn build_fixed_plan(cfg: &ModelConfig, think_steps: usize, max_think_steps: usize) -> ([PlanStep; MAX_PLAN_LEN], usize) {
    let r = think_steps.clamp(1, max_think_steps.max(1));
    let n_peers = if cfg.peer_sub_k > 0 { if cfg.n_layers >= 4 { 2 } else { 1 } } else { 0 };
    let has_attn = cfg.attn_window > 0;
    let mut plan = [PlanStep::default(); MAX_PLAN_LEN];
    let mut n = 0usize;
    if cfg.n_layers >= 4 {
        plan[n] = PlanStep { layer_idx: 0, slot_idx: 0, peer_opt: None, attn_opt: None, is_inj: false };
        n += 1;
        for step in 0..r {
            let p0 = if n_peers >= 1 { Some(0) } else { None };
            let p1 = if n_peers >= 2 { Some(1) } else { None };
            plan[n] = PlanStep { layer_idx: 1, slot_idx: 1 + 2 * step, peer_opt: p0, attn_opt: if has_attn { Some(step) } else { None }, is_inj: step > 0 };
            n += 1;
            plan[n] = PlanStep { layer_idx: 2, slot_idx: 2 + 2 * step, peer_opt: p1, attn_opt: None, is_inj: false };
            n += 1;
        }
        for li in 3..cfg.n_layers {
            plan[n] = PlanStep { layer_idx: li, slot_idx: 1 + 2 * max_think_steps + (li - 3), peer_opt: None, attn_opt: None, is_inj: false };
            n += 1;
        }
    } else {
        for step in 0..r {
            for li in 0..cfg.n_layers {
                let po = if li == 0 && n_peers >= 1 { Some(0) } else { None };
                let ao = if li == 0 && has_attn { Some(step) } else { None };
                plan[n] = PlanStep { layer_idx: li, slot_idx: step * cfg.n_layers + li, peer_opt: po, attn_opt: ao, is_inj: step > 0 && li == 0 };
                n += 1;
            }
        }
    }
    (plan, n)
}

#[derive(Clone)]
pub struct InferenceState {
    pub slots: Vec<LayerState>,
    pub attn_kvs: Vec<AttnKvState>, // 1 O(1) ring-buffer KV-cache per recurrent loop r=0..max_think_steps
    pub think_steps: usize,
    pub max_think_steps: usize,
    pub plan: [PlanStep; MAX_PLAN_LEN],
    pub plan_len: usize,
    pub scratch: StepScratch,
}

impl InferenceState {
    pub fn new(cfg: &ModelConfig) -> Self { Self::with_think_steps(cfg, cfg.recur_steps.max(1)) }

    pub fn with_think_steps(cfg: &ModelConfig, think_steps: usize) -> Self {
        let r = think_steps.clamp(1, 16);
        let max_r = r.max(cfg.recur_steps).max(2); // Preallocate >= 2 for state-safe R=1 <-> R=2!
        let n_slots = if cfg.n_layers >= 4 { 1 + 2 * max_r + (cfg.n_layers - 3) } else { cfg.n_layers * max_r };
        let slots = (0..n_slots).map(|_| LayerState::new(cfg.d_model)).collect();
        let attn_kvs = if cfg.attn_window > 0 {
            (0..max_r).map(|_| AttnKvState::new(cfg.attn_window, cfg.d_model)).collect()
        } else { Vec::new() };
        let (plan, plan_len) = build_fixed_plan(cfg, r, max_r);
        let scratch = StepScratch::from_cfg(cfg);
        Self { slots, attn_kvs, think_steps: r, max_think_steps: max_r, plan, plan_len, scratch }
    }

    /// State-safe recurrent-depth transition: preserves step=0, Prelude, Coda & KV-cache history!
    /// For `r <= max_think_steps` (e.g. R=1 <-> R=2), performs ZERO allocations and ZERO state wipe.
    pub fn set_think_steps(&mut self, cfg: &ModelConfig, think_steps: usize) {
        let r = think_steps.clamp(1, 16);
        if r > self.max_think_steps {
            let old_max = self.max_think_steps;
            let new_max = r;
            if cfg.n_layers >= 4 {
                let n_coda = cfg.n_layers - 3;
                let mut new_slots = Vec::with_capacity(1 + 2 * new_max + n_coda);
                let mut old = std::mem::take(&mut self.slots).into_iter();
                new_slots.push(old.next().unwrap_or_else(|| LayerState::new(cfg.d_model)));
                for _ in 0..(2 * old_max) { new_slots.push(old.next().unwrap_or_else(|| LayerState::new(cfg.d_model))); }
                for _ in 0..(2 * (new_max - old_max)) { new_slots.push(LayerState::new(cfg.d_model)); }
                for _ in 0..n_coda { new_slots.push(old.next().unwrap_or_else(|| LayerState::new(cfg.d_model))); }
                self.slots = new_slots;
            } else {
                for _ in 0..((new_max - old_max) * cfg.n_layers) { self.slots.push(LayerState::new(cfg.d_model)); }
            }
            if cfg.attn_window > 0 {
                for _ in old_max..new_max { self.attn_kvs.push(AttnKvState::new(cfg.attn_window, cfg.d_model)); }
            }
            self.max_think_steps = new_max;
        }
        self.think_steps = r;
        let (plan, plan_len) = build_fixed_plan(cfg, r, self.max_think_steps);
        self.plan = plan;
        self.plan_len = plan_len;
    }

    pub fn reset(&mut self) {
        for ls in &mut self.slots { ls.reset(); }
        for kv in &mut self.attn_kvs { kv.reset(); }
    }
}

#[inline]
pub fn rand_uniform(state: &mut u64) -> f32 {
    let mut x = *state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *state = x;
    2.0 * (((x >> 40) as f32) / ((1u32 << 24) as f32)) - 1.0
}

impl GenesisV6Model {
    pub fn new(cfg: ModelConfig, seed: u64) -> Self {
        let mut rng = seed.max(1);
        let d = cfg.d_model;
        let emb: Vec<f32> = (0..cfg.vocab * d).map(|_| rand_uniform(&mut rng) * 0.04).collect();
        let norm_f = vec![1.0f32; d];
        let recur_inj_alpha = vec![0.25f32; d];
        let eff_layers = (cfg.n_layers + 2 * cfg.recur_steps.saturating_sub(1)).max(1);
        let lin_sc = (1.0 / (d as f32)).sqrt();
        let out_sc = lin_sc / ((2 * eff_layers) as f32).sqrt();
        let ff_sc = (1.0 / (cfg.d_ff as f32)).sqrt() / ((2 * eff_layers) as f32).sqrt();

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            let mut conv_w = vec![0.0f32; 3 * d];
            for i in 0..d { conv_w[i] = 0.10; conv_w[d + i] = 0.20; conv_w[2 * d + i] = 0.70; }
            let router: Vec<f32> = (0..cfg.n_experts * d).map(|_| rand_uniform(&mut rng) * 0.05).collect();
            let mut exp_up = Vec::with_capacity(cfg.n_experts);
            let mut exp_dn = Vec::with_capacity(cfg.n_experts);
            for _ in 0..cfg.n_experts {
                exp_up.push(BitLinear::new(cfg.d_ff, d, &mut rng, lin_sc));
                exp_dn.push(BitLinear::new(d, cfg.d_ff, &mut rng, ff_sc));
            }
            layers.push(SparseBlock {
                norm1: vec![1.0; d], conv_w,
                wz: BitLinear::new(d, d, &mut rng, lin_sc),
                wh: BitLinear::new(d, d, &mut rng, lin_sc),
                wo: BitLinear::new(d, d, &mut rng, out_sc),
                norm2: vec![1.0; d], router, exp_up, exp_dn,
            });
        }
        let n_peers = if cfg.peer_sub_k > 0 { if cfg.n_layers >= 4 { 2 } else { 1 } } else { 0 };
        let mut peers = Vec::with_capacity(n_peers);
        for _ in 0..n_peers {
            peers.push(PeerPkLayer::new(d, cfg.peer_heads.max(1), cfg.peer_sub_k, cfg.peer_k.max(1), &mut rng));
        }
        let attn = if cfg.attn_window > 0 {
            Some(SlidingWindowAttn::new(d, cfg.peer_heads.max(1), cfg.attn_window, &mut rng))
        } else { None };
        Self { cfg, emb, norm_f, recur_inj_alpha, layers, peers, attn, peer_delta: PeerDeltaSidecar::empty() }
    }

    /// Execution schedule `(layer_idx, peer_opt, attn_loop_opt, is_recur_inj)` for `think_steps` loops.
    pub fn exec_plan(&self, think_steps: usize) -> Vec<(usize, Option<usize>, Option<usize>, bool)> {
        let r = think_steps.max(1);
        let has_attn = self.attn.is_some();
        let mut plan = Vec::new();
        if self.cfg.n_layers >= 4 {
            plan.push((0, None, None, false));
            for step in 0..r {
                let p0 = if !self.peers.is_empty() { Some(0) } else { None };
                let p1 = if self.peers.len() >= 2 { Some(1) } else { None };
                plan.push((1, p0, if has_attn { Some(step) } else { None }, step > 0));
                plan.push((2, p1, None, false));
            }
            for li in 3..self.cfg.n_layers { plan.push((li, None, None, false)); }
        } else {
            for step in 0..r {
                for li in 0..self.cfg.n_layers {
                    let po = if li == 0 && !self.peers.is_empty() { Some(0) } else { None };
                    let ao = if li == 0 && has_attn { Some(step) } else { None };
                    plan.push((li, po, ao, step > 0 && li == 0));
                }
            }
        }
        plan
    }

    pub fn refresh_all_quant(&mut self) {
        for l in &mut self.layers {
            l.wz.refresh_quant(); l.wh.refresh_quant(); l.wo.refresh_quant();
            for e in &mut l.exp_up { e.refresh_quant(); }
            for e in &mut l.exp_dn { e.refresh_quant(); }
        }
    }

    pub fn flat_param_slices_mut(&mut self) -> Vec<&mut [f32]> {
        let mut out: Vec<&mut [f32]> = Vec::new();
        out.push(&mut self.emb);
        out.push(&mut self.norm_f);
        out.push(&mut self.recur_inj_alpha);
        for l in &mut self.layers {
            out.push(&mut l.norm1); out.push(&mut l.conv_w);
            out.push(&mut l.wz.w); out.push(&mut l.wh.w); out.push(&mut l.wo.w);
            out.push(&mut l.norm2); out.push(&mut l.router);
            for e in &mut l.exp_up { out.push(&mut e.w); }
            for e in &mut l.exp_dn { out.push(&mut e.w); }
        }
        for p in &mut self.peers {
            out.push(&mut p.norm); out.push(&mut p.wq); out.push(&mut p.keys_a); out.push(&mut p.keys_b);
        }
        if let Some(ref mut a) = self.attn {
            out.push(&mut a.norm); out.push(&mut a.wq); out.push(&mut a.wk); out.push(&mut a.wv); out.push(&mut a.wo);
        }
        out
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut w = BufWriter::with_capacity(1 << 20, File::create(path)?);
        w.write_all(b"GV6MOD03")?;
        for &v in &[
            self.cfg.vocab, self.cfg.d_model, self.cfg.n_layers, self.cfg.n_experts,
            self.cfg.top_k, self.cfg.d_ff, self.cfg.peer_sub_k, self.cfg.peer_heads,
            self.cfg.peer_k, self.cfg.recur_steps, self.cfg.attn_window,
        ] { w.write_all(&(v as u32).to_le_bytes())?; }
        let wf = |w: &mut BufWriter<File>, s: &[f32]| -> std::io::Result<()> {
            for &x in s { w.write_all(&x.to_le_bytes())?; }
            Ok(())
        };
        wf(&mut w, &self.emb)?; wf(&mut w, &self.norm_f)?; wf(&mut w, &self.recur_inj_alpha)?;
        for l in &self.layers {
            wf(&mut w, &l.norm1)?; wf(&mut w, &l.conv_w)?;
            wf(&mut w, &l.wz.w)?; wf(&mut w, &l.wh.w)?; wf(&mut w, &l.wo.w)?;
            wf(&mut w, &l.norm2)?; wf(&mut w, &l.router)?;
            for e in &l.exp_up { wf(&mut w, &e.w)?; }
            for e in &l.exp_dn { wf(&mut w, &e.w)?; }
        }
        for p in &self.peers { p.write_to(&mut w)?; }
        if let Some(ref a) = self.attn { a.write_to(&mut w)?; }
        w.flush()
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let mut r = BufReader::with_capacity(1 << 20, File::open(path)?);
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic == crate::gv8_codec::GV8_MAGIC {
            drop(r);
            return crate::gv8_codec::load_gv8(path);
        }
        let ver: u8 = match &magic {
            b"GV6MOD01" => 1,
            b"GV6MOD02" => 2,
            b"GV6MOD03" => 3,
            _ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad model magic")),
        };
        let n_hdr = match ver { 1 => 6, 2 => 10, _ => 11 };
        let mut u = [0u32; 11];
        let mut b4 = [0u8; 4];
        for i in 0..n_hdr { r.read_exact(&mut b4)?; u[i] = u32::from_le_bytes(b4); }
        let cfg = ModelConfig {
            vocab: u[0] as usize, d_model: u[1] as usize, n_layers: u[2] as usize,
            n_experts: u[3] as usize, top_k: u[4] as usize, d_ff: u[5] as usize,
            peer_sub_k: if ver >= 2 { u[6] as usize } else { 0 },
            peer_heads: if ver >= 2 { u[7] as usize } else { 4 },
            peer_k: if ver >= 2 { u[8] as usize } else { 8 },
            recur_steps: if ver >= 2 { (u[9] as usize).max(1) } else { 1 },
            attn_window: if ver >= 3 { u[10] as usize } else { 0 },
        };
        let mut model = Self::new(cfg, 42);
        let mut rf = |dst: &mut [f32]| -> std::io::Result<()> {
            let mut raw = vec![0u8; dst.len() * 4]; r.read_exact(&mut raw)?;
            for (i, ch) in raw.chunks_exact(4).enumerate() { dst[i] = f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]); }
            Ok(())
        };
        rf(&mut model.emb)?; rf(&mut model.norm_f)?;
        if ver >= 2 { rf(&mut model.recur_inj_alpha)?; }
        for l in &mut model.layers {
            rf(&mut l.norm1)?; rf(&mut l.conv_w)?;
            rf(&mut l.wz.w)?; rf(&mut l.wh.w)?; rf(&mut l.wo.w)?;
            rf(&mut l.norm2)?; rf(&mut l.router)?;
            for e in &mut l.exp_up { rf(&mut e.w)?; }
            for e in &mut l.exp_dn { rf(&mut e.w)?; }
        }
        for p in &mut model.peers { p.read_from(&mut r)?; }
        if let Some(ref mut a) = model.attn { a.read_from(&mut r)?; }
        model.refresh_all_quant();
        Ok(model)
    }
}
