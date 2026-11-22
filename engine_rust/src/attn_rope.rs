//! Genesis-v7 Sliding-Window Causal Multi-Head Self-Attention with RoPE (`attn_rope.rs`).
//! Google Griffin / NVIDIA Hymba hybrid block: lets `<BOT>` attend directly back to every token
//! in `<USER>` within window `W = 128`, with O(1) ring-buffer KV-cache (`196 KB`) at inference!
//! Zero external crates | Strictly <= 295 LOC.

use std::io::{Read, Write};

use crate::gemm_f32::{gemm_f32_bwd, gemm_f32_fwd};
use crate::gv8_kernels::matvec_f32_fast;
use crate::model::rand_uniform;
use crate::simd_avx2::{axpy_f32, dot_f32, rmsnorm_bwd, rmsnorm_fwd};

#[derive(Clone)]
pub struct SlidingWindowAttn {
    pub d_model: usize,
    pub n_heads: usize,
    pub window: usize,
    pub norm: Vec<f32>, // [d]
    pub wq: Vec<f32>,   // [d * d]
    pub wk: Vec<f32>,   // [d * d]
    pub wv: Vec<f32>,   // [d * d]
    pub wo: Vec<f32>,   // [d * d]
}

pub struct AttnGrads {
    pub norm: Vec<f32>,
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
}

/// O(1) fixed-size ring-buffer KV-cache (`W * d * 8 bytes = 196 KB` for W=128, d=192!).
#[derive(Clone)]
pub struct AttnKvState {
    pub k_ring: Vec<f32>,
    pub v_ring: Vec<f32>,
    pub pos: usize,
}

impl AttnKvState {
    pub fn new(window: usize, d_model: usize) -> Self {
        Self { k_ring: vec![0.0; window * d_model], v_ring: vec![0.0; window * d_model], pos: 0 }
    }
    pub fn reset(&mut self) {
        self.k_ring.fill(0.0);
        self.v_ring.fill(0.0);
        self.pos = 0;
    }
}

pub struct AttnBatchCache {
    pub x_in: Vec<f32>,
    pub xn: Vec<f32>,
    pub inv_r: Vec<f32>,
    pub q_rot: Vec<f32>,
    pub k_rot: Vec<f32>,
    pub v_all: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub probs: Vec<f32>, // [n_heads * t_len * window]
}

impl AttnGrads {
    pub fn new(d: usize) -> Self {
        Self { norm: vec![0.0; d], wq: vec![0.0; d * d], wk: vec![0.0; d * d], wv: vec![0.0; d * d], wo: vec![0.0; d * d] }
    }
    pub fn zero(&mut self) {
        self.norm.fill(0.0); self.wq.fill(0.0); self.wk.fill(0.0); self.wv.fill(0.0); self.wo.fill(0.0);
    }
}

/// Applies Rotary Position Embeddings (RoPE, Su et al. 2021) in-place using precomputed `inv_freq`.
#[inline]
fn apply_rope_head(vec: &mut [f32], pos: usize, inv_freq: &[f32], inverse: bool) {
    let p = pos as f32;
    for (i, &freq) in inv_freq.iter().enumerate() {
        let (sn, cs) = (p * freq).sin_cos();
        let s = if inverse { -sn } else { sn };
        let x0 = vec[2 * i];
        let x1 = vec[2 * i + 1];
        vec[2 * i] = x0 * cs - x1 * s;
        vec[2 * i + 1] = x0 * s + x1 * cs;
    }
}

#[inline]
pub fn make_inv_freq(hd: usize) -> Vec<f32> {
    (0..hd / 2).map(|i| 10000.0f32.powf(-(2.0 * (i as f32)) / (hd as f32))).collect()
}

impl SlidingWindowAttn {
    pub fn new(d_model: usize, n_heads: usize, window: usize, rng: &mut u64) -> Self {
        let sc = (1.0 / (d_model as f32)).sqrt();
        let out_sc = sc * 0.35;
        let mut mk = |s: f32| -> Vec<f32> { (0..d_model * d_model).map(|_| rand_uniform(rng) * s).collect() };
        Self {
            d_model, n_heads, window: window.max(8),
            norm: vec![1.0; d_model], wq: mk(sc), wk: mk(sc), wv: mk(sc), wo: mk(out_sc),
        }
    }

    /// Zero-alloc O(1)-memory single-token causal sliding-window residual forward (`x += Attn(RMSNorm(x))`).
    pub fn forward_step_residual_scratch(
        &self,
        x: &mut [f32],
        kv: &mut AttnKvState,
        xn: &mut [f32],
        q: &mut [f32],
        k: &mut [f32],
        v: &mut [f32],
        out: &mut [f32],
        scores: &mut [f32],
        inv_f: &[f32],
    ) {
        let d = self.d_model;
        let nh = self.n_heads;
        let hd = d / nh;
        let w = self.window;
        let scale = 1.0 / (hd as f32).sqrt();
        let _ = rmsnorm_fwd(x, &self.norm, xn);

        matvec_f32_fast(&self.wq, d, d, xn, q);
        matvec_f32_fast(&self.wk, d, d, xn, k);
        matvec_f32_fast(&self.wv, d, d, xn, v);

        let pos = kv.pos;
        for h in 0..nh {
            apply_rope_head(&mut q[h * hd..(h + 1) * hd], pos, inv_f, false);
            apply_rope_head(&mut k[h * hd..(h + 1) * hd], pos, inv_f, false);
        }
        let slot = pos % w;
        kv.k_ring[slot * d..(slot + 1) * d].copy_from_slice(&k[..d]);
        kv.v_ring[slot * d..(slot + 1) * d].copy_from_slice(&v[..d]);
        let start_pos = pos.saturating_sub(w - 1);
        let len = pos - start_pos + 1;
        kv.pos = pos + 1;

        out[..d].fill(0.0);
        for h in 0..nh {
            let qh = &q[h * hd..(h + 1) * hd];
            let mut mx = f32::NEG_INFINITY;
            for j in 0..len {
                let rs = (start_pos + j) % w;
                let kh = &kv.k_ring[rs * d + h * hd..rs * d + (h + 1) * hd];
                let sc = dot_f32(qh, kh) * scale;
                scores[j] = sc;
                if sc > mx { mx = sc; }
            }
            let mut se = 0.0f32;
            for j in 0..len { let e = (scores[j] - mx).exp(); scores[j] = e; se += e; }
            let inv = 1.0 / se.max(1e-12);
            let oh = &mut out[h * hd..(h + 1) * hd];
            for j in 0..len {
                let p = scores[j] * inv;
                let rs = (start_pos + j) % w;
                let vh = &kv.v_ring[rs * d + h * hd..rs * d + (h + 1) * hd];
                axpy_f32(p, vh, oh);
            }
        }
        matvec_f32_fast(&self.wo, d, d, &out[..d], &mut xn[..d]);
        for r in 0..d { x[r] += xn[r]; }
    }

    /// O(1)-memory single-token causal sliding-window residual forward (`x += Attn(RMSNorm(x))`).
    pub fn forward_step_residual(&self, x: &mut [f32], kv: &mut AttnKvState) {
        let d = self.d_model;
        let hd = d / self.n_heads;
        let mut xn = vec![0.0f32; d];
        let mut q = vec![0.0f32; d];
        let mut k = vec![0.0f32; d];
        let mut v = vec![0.0f32; d];
        let mut out = vec![0.0f32; d];
        let mut scores = vec![0.0f32; self.window];
        let inv_f = make_inv_freq(hd);
        self.forward_step_residual_scratch(x, kv, &mut xn, &mut q, &mut k, &mut v, &mut out, &mut scores, &inv_f);
    }

    /// Batched causal sliding-window forward across `t_len` tokens using tiled AVX2 `gemm_f32_fwd`!
    pub fn forward_batch_residual(&self, x_batch: &mut [f32], t_len: usize) -> AttnBatchCache {
        let d = self.d_model; let nh = self.n_heads; let hd = d / nh; let w = self.window;
        let scale = 1.0 / (hd as f32).sqrt();
        let x_in = x_batch.to_vec();
        let mut xn = vec![0.0f32; t_len * d];
        let mut inv_r = vec![0.0f32; t_len];
        for t in 0..t_len {
            let td = t * d;
            inv_r[t] = rmsnorm_fwd(&x_in[td..td + d], &self.norm, &mut xn[td..td + d]);
        }
        let mut q_rot = vec![0.0f32; t_len * d];
        let mut k_rot = vec![0.0f32; t_len * d];
        let mut v_all = vec![0.0f32; t_len * d];
        gemm_f32_fwd(&self.wq, d, d, &xn, &mut q_rot, t_len);
        gemm_f32_fwd(&self.wk, d, d, &xn, &mut k_rot, t_len);
        gemm_f32_fwd(&self.wv, d, d, &xn, &mut v_all, t_len);
        let inv_f = make_inv_freq(hd);
        for t in 0..t_len {
            for h in 0..nh {
                let off = t * d + h * hd;
                apply_rope_head(&mut q_rot[off..off + hd], t, &inv_f, false);
                apply_rope_head(&mut k_rot[off..off + hd], t, &inv_f, false);
            }
        }
        let mut attn_out = vec![0.0f32; t_len * d];
        let mut probs = vec![0.0f32; nh * t_len * w];
        for h in 0..nh {
            for t in 0..t_len {
                let s0 = t.saturating_sub(w - 1);
                let len = t - s0 + 1;
                let qh = &q_rot[t * d + h * hd..t * d + (h + 1) * hd];
                let p_base = (h * t_len + t) * w;
                let mut mx = f32::NEG_INFINITY;
                for j in 0..len {
                    let s = s0 + j;
                    let kh = &k_rot[s * d + h * hd..s * d + (h + 1) * hd];
                    let sc = dot_f32(qh, kh) * scale;
                    probs[p_base + j] = sc;
                    if sc > mx { mx = sc; }
                }
                let mut se = 0.0f32;
                for j in 0..len { let e = (probs[p_base + j] - mx).exp(); probs[p_base + j] = e; se += e; }
                let inv = 1.0 / se.max(1e-12);
                let oh = &mut attn_out[t * d + h * hd..t * d + (h + 1) * hd];
                for j in 0..len {
                    let p = probs[p_base + j] * inv;
                    probs[p_base + j] = p;
                    let s = s0 + j;
                    axpy_f32(p, &v_all[s * d + h * hd..s * d + (h + 1) * hd], oh);
                }
            }
        }
        let mut yo = vec![0.0f32; t_len * d];
        gemm_f32_fwd(&self.wo, d, d, &attn_out, &mut yo, t_len);
        for i in 0..t_len * d { x_batch[i] += yo[i]; }
        AttnBatchCache { x_in, xn, inv_r, q_rot, k_rot, v_all, attn_out, probs }
    }

    /// Exact backward pass through output projection, causal softmax attention, inverse RoPE, QKV & RMSNorm.
    pub fn backward_batch_residual(
        &self, c: &AttnBatchCache, dy_batch: &[f32], dx_batch: &mut [f32], g: &mut AttnGrads, t_len: usize,
    ) {
        let d = self.d_model; let nh = self.n_heads; let hd = d / nh; let w = self.window;
        let scale = 1.0 / (hd as f32).sqrt();
        dx_batch.copy_from_slice(dy_batch); // residual path
        let mut d_out = vec![0.0f32; t_len * d];
        gemm_f32_bwd(&self.wo, d, d, &c.attn_out, dy_batch, &mut d_out, &mut g.wo, t_len);

        let mut dq = vec![0.0f32; t_len * d];
        let mut dk = vec![0.0f32; t_len * d];
        let mut dv = vec![0.0f32; t_len * d];
        let mut dp = vec![0.0f32; w];
        for h in 0..nh {
            for t in 0..t_len {
                let s0 = t.saturating_sub(w - 1);
                let len = t - s0 + 1;
                let doh = &d_out[t * d + h * hd..t * d + (h + 1) * hd];
                let p_base = (h * t_len + t) * w;
                let mut sum_p_dp = 0.0f32;
                for j in 0..len {
                    let s = s0 + j;
                    let p = c.probs[p_base + j];
                    let vh = &c.v_all[s * d + h * hd..s * d + (h + 1) * hd];
                    let dpj = dot_f32(doh, vh);
                    dp[j] = dpj;
                    sum_p_dp += p * dpj;
                    axpy_f32(p, doh, &mut dv[s * d + h * hd..s * d + (h + 1) * hd]);
                }
                let qh = &c.q_rot[t * d + h * hd..t * d + (h + 1) * hd];
                for j in 0..len {
                    let s = s0 + j;
                    let ds = c.probs[p_base + j] * (dp[j] - sum_p_dp) * scale;
                    axpy_f32(ds, &c.k_rot[s * d + h * hd..s * d + (h + 1) * hd], &mut dq[t * d + h * hd..t * d + (h + 1) * hd]);
                    axpy_f32(ds, qh, &mut dk[s * d + h * hd..s * d + (h + 1) * hd]);
                }
            }
        }
        let inv_f = make_inv_freq(hd);
        for t in 0..t_len {
            for h in 0..nh {
                let off = t * d + h * hd;
                apply_rope_head(&mut dq[off..off + hd], t, &inv_f, true);
                apply_rope_head(&mut dk[off..off + hd], t, &inv_f, true);
            }
        }
        let mut dxn = vec![0.0f32; t_len * d];
        gemm_f32_bwd(&self.wq, d, d, &c.xn, &dq, &mut dxn, &mut g.wq, t_len);
        gemm_f32_bwd(&self.wk, d, d, &c.xn, &dk, &mut dxn, &mut g.wk, t_len);
        gemm_f32_bwd(&self.wv, d, d, &c.xn, &dv, &mut dxn, &mut g.wv, t_len);
        for t in 0..t_len {
            let td = t * d;
            rmsnorm_bwd(&c.x_in[td..td + d], &self.norm, c.inv_r[t], &dxn[td..td + d], &mut dx_batch[td..td + d], &mut g.norm);
        }
    }

    pub fn write_to<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        for &s in &[&self.norm[..], &self.wq[..], &self.wk[..], &self.wv[..], &self.wo[..]] {
            for &x in s { w.write_all(&x.to_le_bytes())?; }
        }
        Ok(())
    }

    pub fn read_from<R: Read>(&mut self, r: &mut R) -> std::io::Result<()> {
        let mut rf = |dst: &mut [f32]| -> std::io::Result<()> {
            let mut raw = vec![0u8; dst.len() * 4]; r.read_exact(&mut raw)?;
            for (i, ch) in raw.chunks_exact(4).enumerate() { dst[i] = f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]); }
            Ok(())
        };
        rf(&mut self.norm)?; rf(&mut self.wq)?; rf(&mut self.wk)?; rf(&mut self.wv)?; rf(&mut self.wo)?;
        Ok(())
    }
}
