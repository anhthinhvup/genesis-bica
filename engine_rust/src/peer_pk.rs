//! Genesis-v6.5 DeepMind PEER Product-Key Ultra-Sparse Memory (`peer_pk.rs`).
//! `E = sub_k^2` micro-experts routed in O(sqrt(E)) via Cartesian Product Keys, stored in Q14 `i16`
//! with AVX2+FMA kernels & Sparse Row-Wise Adafactor + Stochastic Rounding. Strictly <= 375 LOC.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::collections::HashMap;
use std::io::{Read, Write};

use crate::gemm_f32::{gemm_f32_bwd, gemm_f32_fwd};
use crate::generate::sigmoid;
use crate::gv8_kernels::{f16_to_f32, matvec_f32_fast, q8_axpy, q8_dot};
use crate::model::rand_uniform;
use crate::peer_delta_sidecar::PeerDeltaTable;
use crate::simd_avx2::{axpy_f32, dot_f32, rmsnorm_bwd, rmsnorm_fwd};

pub const Q14_SCALE: f32 = 16384.0;
pub const Q14_INV: f32 = 1.0 / 16384.0;

#[inline] fn silu(x: f32) -> f32 { x * sigmoid(x) }
#[inline] fn silu_bwd(x: f32) -> f32 { let s = sigmoid(x); s * (1.0 + x * (1.0 - s)) }

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn q14_dot_avx2(q: &[i16], x: &[f32]) -> f32 {
    let n = q.len().min(x.len());
    let mut i = 0usize;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let pq = q.as_ptr();
    let px = x.as_ptr();
    while i + 16 <= n {
        let q0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_loadu_si128(pq.add(i) as *const __m128i)));
        let q1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_loadu_si128(pq.add(i + 8) as *const __m128i)));
        acc0 = _mm256_fmadd_ps(q0, _mm256_loadu_ps(px.add(i)), acc0);
        acc1 = _mm256_fmadd_ps(q1, _mm256_loadu_ps(px.add(i + 8)), acc1);
        i += 16;
    }
    let sum_v = _mm256_add_ps(acc0, acc1);
    let hi = _mm256_extractf128_ps(sum_v, 1);
    let lo = _mm256_castps256_ps128(sum_v);
    let s128 = _mm_add_ps(lo, hi);
    let sh = _mm_movehdup_ps(s128);
    let ss = _mm_add_ps(s128, sh);
    let mut s = _mm_cvtss_f32(_mm_add_ss(ss, _mm_movehl_ps(sh, ss)));
    while i < n { s += (*pq.add(i) as f32) * *px.add(i); i += 1; }
    s * Q14_INV
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn q14_axpy_avx2(scale: f32, q: &[i16], y: &mut [f32]) {
    let n = q.len().min(y.len());
    let sc = scale * Q14_INV;
    let vs = _mm256_set1_ps(sc);
    let pq = q.as_ptr();
    let py = y.as_mut_ptr();
    let mut i = 0usize;
    while i + 16 <= n {
        let q0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_loadu_si128(pq.add(i) as *const __m128i)));
        let q1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_loadu_si128(pq.add(i + 8) as *const __m128i)));
        _mm256_storeu_ps(py.add(i), _mm256_fmadd_ps(vs, q0, _mm256_loadu_ps(py.add(i))));
        _mm256_storeu_ps(py.add(i + 8), _mm256_fmadd_ps(vs, q1, _mm256_loadu_ps(py.add(i + 8))));
        i += 16;
    }
    while i < n { *py.add(i) += sc * (*pq.add(i) as f32); i += 1; }
}

#[inline]
fn q14_dot(q: &[i16], x: &[f32]) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    { return unsafe { q14_dot_avx2(q, x) }; }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
    { q.iter().zip(x.iter()).map(|(&qi, &xi)| (qi as f32) * xi).sum::<f32>() * Q14_INV }
}

#[inline]
fn q14_axpy(scale: f32, q: &[i16], y: &mut [f32]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    { unsafe { q14_axpy_avx2(scale, q, y); } return; }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
    { let sc = scale * Q14_INV; for (yi, &qi) in y.iter_mut().zip(q.iter()) { *yi += sc * (qi as f32); } }
}

#[inline]
fn quant_q14_stoch(val: f32, rng: &mut u64) -> i16 {
    let scaled = (val * Q14_SCALE).clamp(-32766.0, 32766.0);
    let fl = scaled.floor();
    let u = rand_uniform(rng) * 0.5 + 0.5;
    (if u < (scaled - fl) { fl + 1.0 } else { fl }).clamp(-32767.0, 32767.0) as i16
}

#[derive(Clone)]
pub struct PeerQ8Storage {
    pub u_q8: Vec<i8>,
    pub v_q8: Vec<i8>,
    pub u_sc_f16: Vec<u16>,
    pub v_sc_f16: Vec<u16>,
    pub u_sc_f32: Vec<f32>,
    pub v_sc_f32: Vec<f32>,
}

#[derive(Clone)]
pub struct PeerPkLayer {
    pub d_model: usize, pub n_heads: usize, pub sub_k: usize, pub k_head: usize,
    pub norm: Vec<f32>, pub wq: Vec<f32>, pub keys_a: Vec<f32>, pub keys_b: Vec<f32>,
    pub u_exp: Vec<i16>, pub v_exp: Vec<i16>, pub vu: Vec<f32>, pub vv: Vec<f32>,
    pub q8: Option<PeerQ8Storage>,
}

pub struct PeerSparseGrad {
    pub norm: Vec<f32>, pub wq: Vec<f32>, pub keys_a: Vec<f32>, pub keys_b: Vec<f32>,
    pub exp_grads: HashMap<u32, (Vec<f32>, Vec<f32>)>,
}

pub struct PeerBatchCache {
    pub x_in: Vec<f32>, pub xn: Vec<f32>, pub inv_r: Vec<f32>, pub q_all: Vec<f32>,
    pub top_a: Vec<u16>, pub top_b: Vec<u16>, pub top_w: Vec<f32>, pub pre_act: Vec<f32>, pub act: Vec<f32>,
}

impl PeerSparseGrad {
    pub fn new(layer: &PeerPkLayer) -> Self {
        Self {
            norm: vec![0.0; layer.norm.len()], wq: vec![0.0; layer.wq.len()],
            keys_a: vec![0.0; layer.keys_a.len()], keys_b: vec![0.0; layer.keys_b.len()],
            exp_grads: HashMap::with_capacity(1024),
        }
    }
    pub fn zero(&mut self) {
        self.norm.fill(0.0); self.wq.fill(0.0); self.keys_a.fill(0.0); self.keys_b.fill(0.0);
        self.exp_grads.clear();
    }
    pub fn add_from(&mut self, other: &PeerSparseGrad) {
        for (a, &b) in self.norm.iter_mut().zip(other.norm.iter()) { *a += b; }
        for (a, &b) in self.wq.iter_mut().zip(other.wq.iter()) { *a += b; }
        for (a, &b) in self.keys_a.iter_mut().zip(other.keys_a.iter()) { *a += b; }
        for (a, &b) in self.keys_b.iter_mut().zip(other.keys_b.iter()) { *a += b; }
        for (&eid, (gu, gv)) in &other.exp_grads {
            let e = self.exp_grads.entry(eid).or_insert_with(|| (vec![0.0; gu.len()], vec![0.0; gv.len()]));
            for (a, &b) in e.0.iter_mut().zip(gu.iter()) { *a += b; }
            for (a, &b) in e.1.iter_mut().zip(gv.iter()) { *a += b; }
        }
    }
}

fn select_top_indices(scores: &[f32], m: usize, out_idx: &mut [usize], out_val: &mut [f32]) {
    let k = m.min(scores.len());
    for j in 0..k { out_idx[j] = 0; out_val[j] = f32::NEG_INFINITY; }
    for (i, &v) in scores.iter().enumerate() {
        if v <= out_val[k - 1] { continue; }
        let mut pos = k - 1;
        while pos > 0 && v > out_val[pos - 1] {
            out_val[pos] = out_val[pos - 1]; out_idx[pos] = out_idx[pos - 1]; pos -= 1;
        }
        out_val[pos] = v; out_idx[pos] = i;
    }
}

impl PeerPkLayer {
    pub fn new(d_model: usize, n_heads: usize, sub_k: usize, k_head: usize, rng: &mut u64) -> Self {
        let sub_dim = (d_model / n_heads) / 2;
        let n_exp = sub_k * sub_k;
        let q_sc = (1.0 / (d_model as f32)).sqrt();
        let k_sc = (1.0 / (sub_dim as f32)).sqrt();
        let u_sc = (1.0 / (d_model as f32)).sqrt();
        let v_sc = 0.08 / ((n_heads * k_head) as f32).sqrt();
        let wq: Vec<f32> = (0..d_model * d_model).map(|_| rand_uniform(rng) * q_sc).collect();
        let keys_a: Vec<f32> = (0..n_heads * sub_k * sub_dim).map(|_| rand_uniform(rng) * k_sc).collect();
        let keys_b: Vec<f32> = (0..n_heads * sub_k * sub_dim).map(|_| rand_uniform(rng) * k_sc).collect();
        let u_exp: Vec<i16> = (0..n_exp * d_model)
            .map(|_| ((rand_uniform(rng) * u_sc) * Q14_SCALE).clamp(-32767.0, 32767.0) as i16).collect();
        let v_exp: Vec<i16> = (0..n_exp * d_model)
            .map(|_| ((rand_uniform(rng) * v_sc) * Q14_SCALE).clamp(-32767.0, 32767.0) as i16).collect();
        Self {
            d_model, n_heads, sub_k, k_head, norm: vec![1.0; d_model], wq, keys_a, keys_b,
            u_exp, v_exp, vu: vec![1e-4; n_exp], vv: vec![1e-4; n_exp], q8: None,
        }
    }

    fn route_head(
        &self, h: usize, qa: &[f32], qb: &[f32], sa_buf: &mut [f32], sb_buf: &mut [f32],
        out_a: &mut [u16], out_b: &mut [u16], out_w: &mut [f32],
    ) {
        let sd = (self.d_model / self.n_heads) / 2;
        let sk = self.sub_k; let kh = self.k_head;
        let base = h * sk * sd;
        for i in 0..sk {
            sa_buf[i] = dot_f32(&self.keys_a[base + i * sd..base + (i + 1) * sd], qa);
            sb_buf[i] = dot_f32(&self.keys_b[base + i * sd..base + (i + 1) * sd], qb);
        }
        let top_m = kh.min(sk).min(10);
        let mut ia = [0usize; 10]; let mut va = [0.0f32; 10];
        let mut ib = [0usize; 10]; let mut vb = [0.0f32; 10];
        select_top_indices(&sa_buf[..sk], top_m, &mut ia, &mut va);
        select_top_indices(&sb_buf[..sk], top_m, &mut ib, &mut vb);
        let mut pair_sc = [f32::NEG_INFINITY; 100];
        let mut pair_ab = [(0u16, 0u16); 100];
        let mut cnt = 0usize;
        for p in 0..top_m {
            for q in 0..top_m {
                pair_sc[cnt] = va[p] + vb[q]; pair_ab[cnt] = (ia[p] as u16, ib[q] as u16); cnt += 1;
            }
        }
        let mut bi = [0usize; 16]; let mut bv = [0.0f32; 16];
        select_top_indices(&pair_sc[..cnt], kh, &mut bi, &mut bv);
        let mx = bv[0]; let mut sum_e = 0.0f32;
        for m in 0..kh {
            let (a, b) = pair_ab[bi[m]]; out_a[m] = a; out_b[m] = b;
            let ex = (bv[m] - mx).exp(); out_w[m] = ex; sum_e += ex;
        }
        let inv = 1.0 / sum_e.max(1e-12);
        for m in 0..kh { out_w[m] *= inv; }
    }

    #[inline]
    fn dot_u_expert(&self, eid: usize, xn: &[f32]) -> f32 {
        let d = self.d_model;
        if let Some(ref q8) = self.q8 {
            let sc = if !q8.u_sc_f16.is_empty() { f16_to_f32(q8.u_sc_f16[eid]) } else { q8.u_sc_f32[eid] };
            q8_dot(&q8.u_q8[eid * d..(eid + 1) * d], xn, sc)
        } else {
            q14_dot(&self.u_exp[eid * d..(eid + 1) * d], xn)
        }
    }

    #[inline]
    fn axpy_v_expert(&self, scale: f32, eid: usize, x: &mut [f32]) {
        let d = self.d_model;
        if let Some(ref q8) = self.q8 {
            let sc = if !q8.v_sc_f16.is_empty() { f16_to_f32(q8.v_sc_f16[eid]) } else { q8.v_sc_f32[eid] };
            q8_axpy(scale * sc, &q8.v_q8[eid * d..(eid + 1) * d], x);
        } else {
            q14_axpy(scale, &self.v_exp[eid * d..(eid + 1) * d], x);
        }
    }

    /// Zero-alloc single-token PEER forward supporting both Q8/Q14 base weights + sparse f32 delta sidecar.
    pub fn forward_step_residual_scratch(
        &self,
        x: &mut [f32],
        xn: &mut [f32],
        q: &mut [f32],
        sa: &mut [f32],
        sb: &mut [f32],
        delta: Option<&PeerDeltaTable>,
        routed_ids_out: Option<&mut [u32; 64]>,
        routed_count_out: Option<&mut usize>,
    ) {
        let d = self.d_model; let nh = self.n_heads; let hd = d / nh; let sd = hd / 2; let kh = self.k_head;
        let _ = rmsnorm_fwd(x, &self.norm, xn);
        matvec_f32_fast(&self.wq, d, d, xn, q);
        let mut oa = [0u16; 16]; let mut ob = [0u16; 16]; let mut ow = [0.0f32; 16];
        let mut r_cnt = 0usize;
        let mut r_ids_opt = routed_ids_out;
        for h in 0..nh {
            self.route_head(h, &q[h * hd..h * hd + sd], &q[h * hd + sd..(h + 1) * hd], sa, sb, &mut oa, &mut ob, &mut ow);
            for m in 0..kh {
                let eid = (oa[m] as usize) * self.sub_k + (ob[m] as usize);
                if let Some(ref mut buf) = r_ids_opt {
                    if r_cnt < buf.len() { buf[r_cnt] = eid as u32; r_cnt += 1; }
                }
                let mut pre = self.dot_u_expert(eid, xn);
                let d_opt = delta.and_then(|t| t.lookup(eid as u32));
                if let Some((du, _)) = d_opt { pre += dot_f32(du, xn); }
                let sc = ow[m] * silu(pre);
                self.axpy_v_expert(sc, eid, x);
                if let Some((_, dv)) = d_opt { axpy_f32(sc, dv, x); }
            }
        }
        if let Some(c_out) = routed_count_out { *c_out = r_cnt; }
    }

    pub fn forward_step_residual(&self, x: &mut [f32]) {
        let mut xn = vec![0.0f32; self.d_model];
        let mut q = vec![0.0f32; self.d_model];
        let mut sa = vec![0.0f32; self.sub_k];
        let mut sb = vec![0.0f32; self.sub_k];
        self.forward_step_residual_scratch(x, &mut xn, &mut q, &mut sa, &mut sb, None, None, None);
    }

    pub fn forward_batch_residual(&self, x_batch: &mut [f32], t_len: usize) -> PeerBatchCache {
        let d = self.d_model; let nh = self.n_heads; let hd = d / nh; let sd = hd / 2; let kh = self.k_head;
        let n_slots = t_len * nh * kh;
        let mut c = PeerBatchCache {
            x_in: x_batch.to_vec(), xn: vec![0.0; t_len * d], inv_r: vec![0.0; t_len], q_all: vec![0.0; t_len * d],
            top_a: vec![0; n_slots], top_b: vec![0; n_slots], top_w: vec![0.0; n_slots],
            pre_act: vec![0.0; n_slots], act: vec![0.0; n_slots],
        };
        let mut sa = vec![0.0f32; self.sub_k]; let mut sb = vec![0.0f32; self.sub_k];
        for t in 0..t_len {
            let td = t * d;
            c.inv_r[t] = rmsnorm_fwd(&c.x_in[td..td + d], &self.norm, &mut c.xn[td..td + d]);
        }
        // Register-tiled AVX2 Batched GEMM for `wq` across all T tokens!
        gemm_f32_fwd(&self.wq, d, d, &c.xn, &mut c.q_all, t_len);
        for t in 0..t_len {
            let td = t * d;
            for h in 0..nh {
                let slot0 = (t * nh + h) * kh;
                self.route_head(
                    h, &c.q_all[td + h * hd..td + h * hd + sd], &c.q_all[td + h * hd + sd..td + (h + 1) * hd],
                    &mut sa, &mut sb, &mut c.top_a[slot0..slot0 + kh], &mut c.top_b[slot0..slot0 + kh], &mut c.top_w[slot0..slot0 + kh],
                );
                for m in 0..kh {
                    let s = slot0 + m;
                    let eid = (c.top_a[s] as usize) * self.sub_k + (c.top_b[s] as usize);
                    let pre = q14_dot(&self.u_exp[eid * d..(eid + 1) * d], &c.xn[td..td + d]);
                    let act = silu(pre); c.pre_act[s] = pre; c.act[s] = act;
                    q14_axpy(c.top_w[s] * act, &self.v_exp[eid * d..(eid + 1) * d], &mut x_batch[td..td + d]);
                }
            }
        }
        c
    }

    pub fn backward_batch_residual(
        &self, c: &PeerBatchCache, dy_batch: &[f32], dx_batch: &mut [f32], g: &mut PeerSparseGrad, t_len: usize,
    ) {
        let d = self.d_model; let nh = self.n_heads; let hd = d / nh; let sd = hd / 2; let kh = self.k_head; let sk = self.sub_k;
        dx_batch.copy_from_slice(dy_batch);
        let mut dxn_all = vec![0.0f32; t_len * d];
        let mut dq_all = vec![0.0f32; t_len * d];
        let mut d_w = [0.0f32; 16];
        for t in 0..t_len {
            let td = t * d; let xn = &c.xn[td..td + d]; let dy = &dy_batch[td..td + d];
            for h in 0..nh {
                let slot0 = (t * nh + h) * kh; let mut sum_wd = 0.0f32;
                for m in 0..kh {
                    let s = slot0 + m; let eid = (c.top_a[s] as usize) * sk + (c.top_b[s] as usize);
                    let w = c.top_w[s]; let act = c.act[s];
                    let gv = q14_dot(&self.v_exp[eid * d..(eid + 1) * d], dy);
                    d_w[m] = gv * act; sum_wd += w * d_w[m];
                    let d_pre = gv * w * silu_bwd(c.pre_act[s]);
                    q14_axpy(d_pre, &self.u_exp[eid * d..(eid + 1) * d], &mut dxn_all[td..td + d]);
                    let e = g.exp_grads.entry(eid as u32).or_insert_with(|| (vec![0.0; d], vec![0.0; d]));
                    axpy_f32(d_pre, xn, &mut e.0); axpy_f32(w * act, dy, &mut e.1);
                }
                let qa = &c.q_all[td + h * hd..td + h * hd + sd];
                let qb = &c.q_all[td + h * hd + sd..td + (h + 1) * hd];
                let base = h * sk * sd;
                for m in 0..kh {
                    let s = slot0 + m; let ds = c.top_w[s] * (d_w[m] - sum_wd);
                    let ao = base + (c.top_a[s] as usize) * sd; let bo = base + (c.top_b[s] as usize) * sd;
                    axpy_f32(ds, qa, &mut g.keys_a[ao..ao + sd]);
                    axpy_f32(ds, &self.keys_a[ao..ao + sd], &mut dq_all[td + h * hd..td + h * hd + sd]);
                    axpy_f32(ds, qb, &mut g.keys_b[bo..bo + sd]);
                    axpy_f32(ds, &self.keys_b[bo..bo + sd], &mut dq_all[td + h * hd + sd..td + (h + 1) * hd]);
                }
            }
        }
        // Register-tiled AVX2 Batched GEMM backward for `wq` (128x fewer RAM writes on `g.wq`!)
        gemm_f32_bwd(&self.wq, d, d, &c.xn, &dq_all, &mut dxn_all, &mut g.wq, t_len);
        for t in 0..t_len {
            let td = t * d;
            rmsnorm_bwd(&c.x_in[td..td + d], &self.norm, c.inv_r[t], &dxn_all[td..td + d], &mut dx_batch[td..td + d], &mut g.norm);
        }
    }

    pub fn apply_sparse_updates(&mut self, g: &PeerSparseGrad, scale: f32, lr: f32, wd: f32, rng: &mut u64) {
        let d = self.d_model; let inv_d = 1.0 / (d as f32);
        for (&eid_u32, (gu, gv)) in &g.exp_grads {
            let eid = eid_u32 as usize;
            let (mut su, mut sv) = (0.0f32, 0.0f32);
            for i in 0..d { let a = gu[i] * scale; let b = gv[i] * scale; su += a * a; sv += b * b; }
            self.vu[eid] = 0.98 * self.vu[eid] + 0.02 * (su * inv_d);
            self.vv[eid] = 0.98 * self.vv[eid] + 0.02 * (sv * inv_d);
            let step_u = lr / (self.vu[eid].sqrt() + 1e-5);
            let step_v = lr / (self.vv[eid].sqrt() + 1e-5);
            let base = eid * d;
            for i in 0..d {
                let u_old = (self.u_exp[base + i] as f32) * Q14_INV * (1.0 - lr * wd);
                let v_old = (self.v_exp[base + i] as f32) * Q14_INV * (1.0 - lr * wd);
                let gu_c = (gu[i] * scale).clamp(-2.0, 2.0);
                let gv_c = (gv[i] * scale).clamp(-2.0, 2.0);
                self.u_exp[base + i] = quant_q14_stoch(u_old - step_u * gu_c, rng);
                self.v_exp[base + i] = quant_q14_stoch(v_old - step_v * gv_c, rng);
            }
        }
    }

    pub fn write_to<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        for &s in &[&self.norm[..], &self.wq[..], &self.keys_a[..], &self.keys_b[..], &self.vu[..], &self.vv[..]] {
            for &x in s { w.write_all(&x.to_le_bytes())?; }
        }
        for &q in &self.u_exp { w.write_all(&q.to_le_bytes())?; }
        for &q in &self.v_exp { w.write_all(&q.to_le_bytes())?; }
        Ok(())
    }

    pub fn read_from<R: Read>(&mut self, r: &mut R) -> std::io::Result<()> {
        let mut rf = |dst: &mut [f32]| -> std::io::Result<()> {
            let mut raw = vec![0u8; dst.len() * 4]; r.read_exact(&mut raw)?;
            for (i, ch) in raw.chunks_exact(4).enumerate() { dst[i] = f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]); }
            Ok(())
        };
        rf(&mut self.norm)?; rf(&mut self.wq)?; rf(&mut self.keys_a)?; rf(&mut self.keys_b)?; rf(&mut self.vu)?; rf(&mut self.vv)?;
        let mut ri = |dst: &mut [i16]| -> std::io::Result<()> {
            let mut raw = vec![0u8; dst.len() * 2]; r.read_exact(&mut raw)?;
            for (i, ch) in raw.chunks_exact(2).enumerate() { dst[i] = i16::from_le_bytes([ch[0], ch[1]]); }
            Ok(())
        };
        ri(&mut self.u_exp)?; ri(&mut self.v_exp)?;
        Ok(())
    }
}
