//! Genesis-v6.5 Gradient Buffers, Dense AdamW + Sparse PEER Row-Wise Adafactor (`optim.rs`).
//! Zero external crates | Strictly <= 220 LOC.

use crate::attn_rope::AttnGrads;
use crate::model::GenesisV6Model;
use crate::peer_pk::PeerSparseGrad;

pub struct LayerGrads {
    pub norm1: Vec<f32>,
    pub conv_w: Vec<f32>,
    pub wz: Vec<f32>,
    pub wh: Vec<f32>,
    pub wo: Vec<f32>,
    pub norm2: Vec<f32>,
    pub router: Vec<f32>,
    pub exp_up: Vec<Vec<f32>>,
    pub exp_dn: Vec<Vec<f32>>,
}

pub struct GradBuf {
    pub emb: Vec<f32>,
    pub norm_f: Vec<f32>,
    pub recur_inj_alpha: Vec<f32>,
    pub layers: Vec<LayerGrads>,
    pub peers: Vec<PeerSparseGrad>,
    pub attn: Option<AttnGrads>,
}

impl GradBuf {
    pub fn new(model: &GenesisV6Model) -> Self {
        let cfg = &model.cfg;
        let d = cfg.d_model;
        let df = cfg.d_ff;
        let ne = cfg.n_experts;
        let layers = (0..cfg.n_layers)
            .map(|_| LayerGrads {
                norm1: vec![0.0; d],
                conv_w: vec![0.0; 3 * d],
                wz: vec![0.0; d * d],
                wh: vec![0.0; d * d],
                wo: vec![0.0; d * d],
                norm2: vec![0.0; d],
                router: vec![0.0; ne * d],
                exp_up: (0..ne).map(|_| vec![0.0; df * d]).collect(),
                exp_dn: (0..ne).map(|_| vec![0.0; d * df]).collect(),
            })
            .collect();
        let peers = model.peers.iter().map(PeerSparseGrad::new).collect();
        let attn = model.attn.as_ref().map(|_| AttnGrads::new(d));
        Self { emb: vec![0.0; cfg.vocab * d], norm_f: vec![0.0; d], recur_inj_alpha: vec![0.0; d], layers, peers, attn }
    }

    pub fn zero(&mut self) {
        self.emb.fill(0.0);
        self.norm_f.fill(0.0);
        self.recur_inj_alpha.fill(0.0);
        for l in &mut self.layers {
            l.norm1.fill(0.0); l.conv_w.fill(0.0);
            l.wz.fill(0.0); l.wh.fill(0.0); l.wo.fill(0.0);
            l.norm2.fill(0.0); l.router.fill(0.0);
            for e in &mut l.exp_up { e.fill(0.0); }
            for e in &mut l.exp_dn { e.fill(0.0); }
        }
        for p in &mut self.peers { p.zero(); }
        if let Some(ref mut a) = self.attn { a.zero(); }
    }

    pub fn flat_slices(&self) -> Vec<&[f32]> {
        let mut out: Vec<&[f32]> = Vec::new();
        out.push(&self.emb); out.push(&self.norm_f); out.push(&self.recur_inj_alpha);
        for l in &self.layers {
            out.push(&l.norm1); out.push(&l.conv_w);
            out.push(&l.wz); out.push(&l.wh); out.push(&l.wo);
            out.push(&l.norm2); out.push(&l.router);
            for e in &l.exp_up { out.push(e); }
            for e in &l.exp_dn { out.push(e); }
        }
        for p in &self.peers {
            out.push(&p.norm); out.push(&p.wq); out.push(&p.keys_a); out.push(&p.keys_b);
        }
        if let Some(ref a) = self.attn {
            out.push(&a.norm); out.push(&a.wq); out.push(&a.wk); out.push(&a.wv); out.push(&a.wo);
        }
        out
    }

    pub fn flat_slices_mut(&mut self) -> Vec<&mut [f32]> {
        let mut out: Vec<&mut [f32]> = Vec::new();
        out.push(&mut self.emb); out.push(&mut self.norm_f); out.push(&mut self.recur_inj_alpha);
        for l in &mut self.layers {
            out.push(&mut l.norm1); out.push(&mut l.conv_w);
            out.push(&mut l.wz); out.push(&mut l.wh); out.push(&mut l.wo);
            out.push(&mut l.norm2); out.push(&mut l.router);
            for e in &mut l.exp_up { out.push(e); }
            for e in &mut l.exp_dn { out.push(e); }
        }
        for p in &mut self.peers {
            out.push(&mut p.norm); out.push(&mut p.wq); out.push(&mut p.keys_a); out.push(&mut p.keys_b);
        }
        if let Some(ref mut a) = self.attn {
            out.push(&mut a.norm); out.push(&mut a.wq); out.push(&mut a.wk); out.push(&mut a.wv); out.push(&mut a.wo);
        }
        out
    }

    pub fn add_from(&mut self, other: &GradBuf) {
        {
            let mut dst = self.flat_slices_mut();
            let src = other.flat_slices();
            for (d, s) in dst.iter_mut().zip(src.iter()) {
                for (a, &b) in d.iter_mut().zip(s.iter()) { *a += b; }
            }
        }
        for (dp, sp) in self.peers.iter_mut().zip(other.peers.iter()) {
            dp.add_from(sp);
        }
    }

    pub fn grad_norm(&self, scale: f32) -> f32 {
        let mut sq = 0.0f64;
        for g in self.flat_slices() {
            for &x in g {
                let xs = (x * scale) as f64;
                sq += xs * xs;
            }
        }
        sq.sqrt() as f32
    }
}

pub struct AdamWState {
    pub m: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    pub step: u64,
    pub stoch_rng: u64,
}

impl AdamWState {
    pub fn new(model: &mut GenesisV6Model) -> Self {
        let slices = model.flat_param_slices_mut();
        let m: Vec<Vec<f32>> = slices.iter().map(|s| vec![0.0f32; s.len()]).collect();
        let v: Vec<Vec<f32>> = slices.iter().map(|s| vec![0.0f32; s.len()]).collect();
        Self { m, v, step: 0, stoch_rng: 0x9E37_79B9_7F4A_7C15 }
    }

    /// Runs one decoupled AdamW step (dense backbone + PEER keys) + Sparse Adafactor (Q14 experts).
    pub fn step_update(
        &mut self,
        model: &mut GenesisV6Model,
        grads: &GradBuf,
        scale: f32,
        lr: f32,
        weight_decay: f32,
        grad_clip: f32,
    ) -> f32 {
        self.step += 1;
        let g_slices = grads.flat_slices();
        let mut sq_norm = 0.0f64;
        for g in &g_slices {
            for &x in *g {
                let xs = (x * scale) as f64;
                sq_norm += xs * xs;
            }
        }
        let gnorm = sq_norm.sqrt() as f32;
        let clip_mult = if gnorm > grad_clip && gnorm > 1e-12 {
            (grad_clip / gnorm) * scale
        } else {
            scale
        };

        let beta1 = 0.9f32;
        let beta2 = 0.95f32;
        let eps = 1e-8f32;
        let bc1 = 1.0 - beta1.powi(self.step as i32);
        let bc2 = 1.0 - beta2.powi(self.step as i32);
        let step_lr = lr * (bc2.sqrt() / bc1);

        {
            let mut p_slices = model.flat_param_slices_mut();
            for (idx, ((p, g), (m, v))) in p_slices
                .iter_mut()
                .zip(g_slices.iter())
                .zip(self.m.iter_mut().zip(self.v.iter_mut()))
                .enumerate()
            {
                let wd = if idx == 0 || p.len() > 4096 { weight_decay } else { 0.0 };
                for i in 0..p.len() {
                    let gi = g[i] * clip_mult;
                    if gi == 0.0 && m[i] == 0.0 { continue; }
                    let mi = beta1 * m[i] + (1.0 - beta1) * gi;
                    let vi = beta2 * v[i] + (1.0 - beta2) * gi * gi;
                    m[i] = mi;
                    v[i] = vi;
                    let mut w = p[i];
                    if wd > 0.0 { w *= 1.0 - lr * wd; }
                    p[i] = w - step_lr * mi / (vi.sqrt() + eps);
                }
            }
        }
        for (peer, pg) in model.peers.iter_mut().zip(grads.peers.iter()) {
            peer.apply_sparse_updates(pg, clip_mult, lr * 0.8, weight_decay, &mut self.stoch_rng);
        }
        model.refresh_all_quant();
        gnorm
    }
}
