//! Genesis-v7 Register-Tiled AVX2+FMA Dense `f32` Batched GEMM Kernels (`gemm_f32.rs`).
//! Shared by Sliding-Window Causal Attention (`wq, wk, wv, wo`) & PEER Query (`peer.wq`).
//! 4-token tiled forward/dx + register-pinned `dw` across all `T` tokens (`128x` fewer RAM writes!).
//! Zero external crates | Strictly <= 195 LOC.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::simd_avx2::{axpy_f32, dot_f32, rmsnorm_bwd, rmsnorm_fwd};

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum256_ps(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let s128 = _mm_add_ps(lo, hi);
    let sh = _mm_movehdup_ps(s128);
    let sums = _mm_add_ps(s128, sh);
    _mm_cvtss_f32(_mm_add_ss(sums, _mm_movehl_ps(sh, sums)))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_f32_fwd_x4_avx2(
    w: &[f32], rows: usize, cols: usize,
    x0: &[f32], x1: &[f32], x2: &[f32], x3: &[f32],
    y0: &mut [f32], y1: &mut [f32], y2: &mut [f32], y3: &mut [f32],
) {
    let (p0, p1, p2, p3) = (x0.as_ptr(), x1.as_ptr(), x2.as_ptr(), x3.as_ptr());
    for r in 0..rows {
        let pw = w.as_ptr().add(r * cols);
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();
        let mut c = 0usize;
        while c + 8 <= cols {
            let vw = _mm256_loadu_ps(pw.add(c));
            a0 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(p0.add(c)), a0);
            a1 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(p1.add(c)), a1);
            a2 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(p2.add(c)), a2);
            a3 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(p3.add(c)), a3);
            c += 8;
        }
        let (mut s0, mut s1, mut s2, mut s3) = (hsum256_ps(a0), hsum256_ps(a1), hsum256_ps(a2), hsum256_ps(a3));
        while c < cols {
            let wv = *pw.add(c);
            s0 += wv * *p0.add(c); s1 += wv * *p1.add(c);
            s2 += wv * *p2.add(c); s3 += wv * *p3.add(c);
            c += 1;
        }
        y0[r] = s0; y1[r] = s1; y2[r] = s2; y3[r] = s3;
    }
}

/// Batched `Y[t] = W @ X[t]` across `t_len` tokens (4-token register-tiled AVX2+FMA!).
pub fn gemm_f32_fwd(w: &[f32], rows: usize, cols: usize, x_batch: &[f32], y_batch: &mut [f32], t_len: usize) {
    let mut t = 0usize;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    {
        while t + 4 <= t_len {
            let x0 = &x_batch[t * cols..(t + 1) * cols];
            let x1 = &x_batch[(t + 1) * cols..(t + 2) * cols];
            let x2 = &x_batch[(t + 2) * cols..(t + 3) * cols];
            let x3 = &x_batch[(t + 3) * cols..(t + 4) * cols];
            let (left, right) = y_batch[t * rows..(t + 4) * rows].split_at_mut(2 * rows);
            let (y0, y1) = left.split_at_mut(rows);
            let (y2, y3) = right.split_at_mut(rows);
            unsafe { gemm_f32_fwd_x4_avx2(w, rows, cols, x0, x1, x2, x3, y0, y1, y2, y3); }
            t += 4;
        }
    }
    while t < t_len {
        let xt = &x_batch[t * cols..(t + 1) * cols];
        let yt = &mut y_batch[t * rows..(t + 1) * rows];
        for r in 0..rows { yt[r] = dot_f32(&w[r * cols..(r + 1) * cols], xt); }
        t += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_f32_dw_avx2(rows: usize, cols: usize, x_batch: &[f32], dy_batch: &[f32], dw: &mut [f32], t_len: usize) {
    let (px, pdy, pdw) = (x_batch.as_ptr(), dy_batch.as_ptr(), dw.as_mut_ptr());
    for r in 0..rows {
        let row_dw = pdw.add(r * cols);
        let mut c = 0usize;
        while c + 16 <= cols {
            let mut acc0 = _mm256_loadu_ps(row_dw.add(c));
            let mut acc1 = _mm256_loadu_ps(row_dw.add(c + 8));
            for t in 0..t_len {
                let vd = _mm256_set1_ps(*pdy.add(t * rows + r));
                let xt = px.add(t * cols + c);
                acc0 = _mm256_fmadd_ps(vd, _mm256_loadu_ps(xt), acc0);
                acc1 = _mm256_fmadd_ps(vd, _mm256_loadu_ps(xt.add(8)), acc1);
            }
            _mm256_storeu_ps(row_dw.add(c), acc0);
            _mm256_storeu_ps(row_dw.add(c + 8), acc1);
            c += 16;
        }
        while c + 8 <= cols {
            let mut acc = _mm256_loadu_ps(row_dw.add(c));
            for t in 0..t_len {
                let vd = _mm256_set1_ps(*pdy.add(t * rows + r));
                acc = _mm256_fmadd_ps(vd, _mm256_loadu_ps(px.add(t * cols + c)), acc);
            }
            _mm256_storeu_ps(row_dw.add(c), acc);
            c += 8;
        }
        while c < cols {
            let mut s = *row_dw.add(c);
            for t in 0..t_len { s += *pdy.add(t * rows + r) * *px.add(t * cols + c); }
            *row_dw.add(c) = s;
            c += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_f32_dx_x4_avx2(
    w: &[f32], rows: usize, cols: usize,
    dy0: &[f32], dy1: &[f32], dy2: &[f32], dy3: &[f32],
    dx0: &mut [f32], dx1: &mut [f32], dx2: &mut [f32], dx3: &mut [f32],
) {
    let (p0, p1, p2, p3) = (dx0.as_mut_ptr(), dx1.as_mut_ptr(), dx2.as_mut_ptr(), dx3.as_mut_ptr());
    for r in 0..rows {
        let (vg0, vg1, vg2, vg3) = (_mm256_set1_ps(dy0[r]), _mm256_set1_ps(dy1[r]), _mm256_set1_ps(dy2[r]), _mm256_set1_ps(dy3[r]));
        let pw = w.as_ptr().add(r * cols);
        let mut c = 0usize;
        while c + 8 <= cols {
            let vw = _mm256_loadu_ps(pw.add(c));
            _mm256_storeu_ps(p0.add(c), _mm256_fmadd_ps(vg0, vw, _mm256_loadu_ps(p0.add(c))));
            _mm256_storeu_ps(p1.add(c), _mm256_fmadd_ps(vg1, vw, _mm256_loadu_ps(p1.add(c))));
            _mm256_storeu_ps(p2.add(c), _mm256_fmadd_ps(vg2, vw, _mm256_loadu_ps(p2.add(c))));
            _mm256_storeu_ps(p3.add(c), _mm256_fmadd_ps(vg3, vw, _mm256_loadu_ps(p3.add(c))));
            c += 8;
        }
        while c < cols {
            let wv = *pw.add(c);
            *p0.add(c) += dy0[r] * wv; *p1.add(c) += dy1[r] * wv;
            *p2.add(c) += dy2[r] * wv; *p3.add(c) += dy3[r] * wv;
            c += 1;
        }
    }
}

/// Batched `dw += dy^T @ x` (register-pinned!) AND `dx += W^T @ dy` (4-token L1-tiled!).
pub fn gemm_f32_bwd(
    w: &[f32], rows: usize, cols: usize,
    x_batch: &[f32], dy_batch: &[f32], dx_batch: &mut [f32], dw: &mut [f32], t_len: usize,
) {
    if t_len == 0 { return; }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    {
        unsafe { gemm_f32_dw_avx2(rows, cols, x_batch, dy_batch, dw, t_len); }
        let mut t = 0usize;
        while t + 4 <= t_len {
            let dy0 = &dy_batch[t * rows..(t + 1) * rows];
            let dy1 = &dy_batch[(t + 1) * rows..(t + 2) * rows];
            let dy2 = &dy_batch[(t + 2) * rows..(t + 3) * rows];
            let dy3 = &dy_batch[(t + 3) * rows..(t + 4) * rows];
            let (left, right) = dx_batch[t * cols..(t + 4) * cols].split_at_mut(2 * cols);
            let (dx0, dx1) = left.split_at_mut(cols);
            let (dx2, dx3) = right.split_at_mut(cols);
            unsafe { gemm_f32_dx_x4_avx2(w, rows, cols, dy0, dy1, dy2, dy3, dx0, dx1, dx2, dx3); }
            t += 4;
        }
        while t < t_len {
            let dy = &dy_batch[t * rows..(t + 1) * rows];
            let dx = &mut dx_batch[t * cols..(t + 1) * cols];
            for r in 0..rows { axpy_f32(dy[r], &w[r * cols..(r + 1) * cols], dx); }
            t += 1;
        }
        return;
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
    {
        for t in 0..t_len {
            let xt = &x_batch[t * cols..(t + 1) * cols];
            let dy = &dy_batch[t * rows..(t + 1) * rows];
            let dx = &mut dx_batch[t * cols..(t + 1) * cols];
            for r in 0..rows {
                axpy_f32(dy[r], xt, &mut dw[r * cols..(r + 1) * cols]);
                axpy_f32(dy[r], &w[r * cols..(r + 1) * cols], dx);
            }
        }
    }
}

/// Register-tiled AVX2 Batched LM Head (Final RMSNorm + Gemma-2 SoftCap + Exact or Sampled Softmax).
/// Replaces per-token L3 random gathers/scatters with 1 contiguous `gemm_f32_fwd` + `gemm_f32_bwd`!
pub fn lm_head_fwd_bwd(
    emb: &[f32],
    norm_f: &[f32],
    x_cur: &[f32],
    tokens: &[u16],
    loss_mask: &[bool],
    v: usize,
    d: usize,
    t_len: usize,
    sampled_k: usize,
    rng: &mut u64,
    dx_cur: &mut [f32],
    grad_norm_f: &mut [f32],
    grad_emb: &mut [f32],
) -> (f64, usize) {
    let mut act_ts: Vec<usize> = Vec::with_capacity(t_len);
    for t in 0..t_len {
        if loss_mask[t] { act_ts.push(t); }
    }
    let m = act_ts.len();
    if m == 0 { return (0.0, 1); }

    let mut xf = vec![0.0f32; m * d];
    let mut inv_f = vec![0.0f32; m];
    for (a, &t) in act_ts.iter().enumerate() {
        inv_f[a] = rmsnorm_fwd(&x_cur[t * d..(t + 1) * d], norm_f, &mut xf[a * d..(a + 1) * d]);
    }
    let mut dxf_all = vec![0.0f32; m * d];
    let mut total_loss = 0.0f64;
    let use_sampled = sampled_k > 0 && sampled_k + 8 < v;

    if !use_sampled {
        let mut logits = vec![0.0f32; m * v];
        gemm_f32_fwd(emb, v, d, &xf, &mut logits, m);
        let mut cap_d = vec![0.0f32; v];
        for (a, &t) in act_ts.iter().enumerate() {
            let y_id = (tokens[t + 1] as usize).min(v - 1);
            let row = &mut logits[a * v..(a + 1) * v];
            let mut max_l = f32::NEG_INFINITY;
            for tok in 0..v {
                let th = ((row[tok] as f64) / 30.0).tanh();
                let lv = (30.0 * th) as f32;
                row[tok] = lv;
                cap_d[tok] = (1.0 - th * th) as f32;
                if lv > max_l { max_l = lv; }
            }
            let mut sum_e = 0.0f64;
            for tok in 0..v {
                let e = ((row[tok] - max_l) as f64).exp();
                row[tok] = e as f32;
                sum_e += e;
            }
            let inv_s = 1.0 / sum_e.max(1e-18);
            total_loss -= ((row[y_id] as f64) * inv_s).max(1e-18).ln();
            let inv_sf = inv_s as f32;
            for tok in 0..v {
                row[tok] = (row[tok] * inv_sf - if tok == y_id { 1.0 } else { 0.0 }) * cap_d[tok];
            }
        }
        gemm_f32_bwd(emb, v, d, &xf, &logits, &mut dxf_all, grad_emb, m);
    } else {
        let mut seen = vec![false; v];
        let mut negs: Vec<usize> = Vec::with_capacity(sampled_k);
        while negs.len() < sampled_k {
            let mut x = *rng;
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            *rng = x;
            let cand = (x as usize) % v;
            if !seen[cand] { seen[cand] = true; negs.push(cand); }
        }
        let k_len = negs.len();
        let mut w_neg = vec![0.0f32; k_len * d];
        for (k, &uid) in negs.iter().enumerate() {
            w_neg[k * d..(k + 1) * d].copy_from_slice(&emb[uid * d..(uid + 1) * d]);
        }
        let mut neg_logits = vec![0.0f32; m * k_len];
        gemm_f32_fwd(&w_neg, k_len, d, &xf, &mut neg_logits, m);
        let mut d_neg = vec![0.0f32; m * k_len];
        let mut cap_d = vec![0.0f32; k_len];
        let mut exp_k = vec![0.0f32; k_len];

        for (a, &t) in act_ts.iter().enumerate() {
            let y_id = (tokens[t + 1] as usize).min(v - 1);
            let xfa = &xf[a * d..(a + 1) * d];
            let raw0 = dot_f32(&emb[y_id * d..(y_id + 1) * d], xfa);
            let th0 = (raw0 / 30.0).tanh();
            let l0 = 30.0 * th0;
            let cd0 = 1.0 - th0 * th0;
            let neg_cnt = if seen[y_id] { k_len - 1 } else { k_len };
            let log_corr = (((v - 1) as f32) / (neg_cnt.max(1) as f32)).ln();

            let base = a * k_len;
            let mut max_l = l0;
            for k in 0..k_len {
                if negs[k] == y_id { continue; }
                let th = (neg_logits[base + k] / 30.0).tanh();
                let lv = 30.0 * th + log_corr;
                exp_k[k] = lv;
                cap_d[k] = 1.0 - th * th;
                if lv > max_l { max_l = lv; }
            }
            let e0 = ((l0 - max_l) as f64).exp();
            let mut sum_e = e0;
            for k in 0..k_len {
                if negs[k] == y_id { continue; }
                let e = ((exp_k[k] - max_l) as f64).exp();
                exp_k[k] = e as f32;
                sum_e += e;
            }
            let inv_s = 1.0 / sum_e.max(1e-18);
            total_loss -= (e0 * inv_s).max(1e-18).ln();
            let inv_sf = inv_s as f32;
            let dl0 = ((e0 as f32) * inv_sf - 1.0) * cd0;
            axpy_f32(dl0, &emb[y_id * d..(y_id + 1) * d], &mut dxf_all[a * d..(a + 1) * d]);
            axpy_f32(dl0, xfa, &mut grad_emb[y_id * d..(y_id + 1) * d]);
            for k in 0..k_len {
                if negs[k] == y_id { continue; }
                d_neg[base + k] = exp_k[k] * inv_sf * cap_d[k];
            }
        }
        let mut dw_neg = vec![0.0f32; k_len * d];
        gemm_f32_bwd(&w_neg, k_len, d, &xf, &d_neg, &mut dxf_all, &mut dw_neg, m);
        for (k, &uid) in negs.iter().enumerate() {
            let dst = &mut grad_emb[uid * d..(uid + 1) * d];
            let src = &dw_neg[k * d..(k + 1) * d];
            for i in 0..d { dst[i] += src[i]; }
        }
    }

    for (a, &t) in act_ts.iter().enumerate() {
        rmsnorm_bwd(&x_cur[t * d..(t + 1) * d], norm_f, inv_f[a], &dxf_all[a * d..(a + 1) * d], &mut dx_cur[t * d..(t + 1) * d], grad_norm_f);
    }
    (total_loss, m)
}
