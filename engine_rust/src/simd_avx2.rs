//! Genesis-v6 Pure-Rust AVX2 + FMA & BitNet b1.58 Ternary SIMD Kernels (`simd_avx2.rs`).
//! Branchless 16-wide AVX2+FMA ternary forward & backward STE kernels.
//! Zero external crates | Zero C:\ writes | 100% CPU | Strictly <= 265 LOC.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Fast horizontal sum of `__m256`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum256_ps(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn dot_f32_avx2_inner(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut i = 0usize;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    while i + 16 <= n {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + 8)), _mm256_loadu_ps(pb.add(i + 8)), acc1);
        i += 16;
    }
    while i + 8 <= n {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
        i += 8;
    }
    let mut sum = hsum256_ps(_mm256_add_ps(acc0, acc1));
    while i < n {
        sum += *pa.add(i) * *pb.add(i);
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn axpy_f32_avx2_inner(scale: f32, x: &[f32], y: &mut [f32]) {
    let n = x.len().min(y.len());
    let mut i = 0usize;
    let vs = _mm256_set1_ps(scale);
    let px = x.as_ptr();
    let py = y.as_mut_ptr();
    while i + 16 <= n {
        let vy0 = _mm256_loadu_ps(py.add(i));
        let vy1 = _mm256_loadu_ps(py.add(i + 8));
        _mm256_storeu_ps(py.add(i), _mm256_fmadd_ps(vs, _mm256_loadu_ps(px.add(i)), vy0));
        _mm256_storeu_ps(py.add(i + 8), _mm256_fmadd_ps(vs, _mm256_loadu_ps(px.add(i + 8)), vy1));
        i += 16;
    }
    while i + 8 <= n {
        let vy = _mm256_loadu_ps(py.add(i));
        _mm256_storeu_ps(py.add(i), _mm256_fmadd_ps(vs, _mm256_loadu_ps(px.add(i)), vy));
        i += 8;
    }
    while i < n {
        *py.add(i) += scale * *px.add(i);
        i += 1;
    }
}

#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    {
        return unsafe { dot_f32_avx2_inner(a, b) };
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
    {
        let mut s = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) { s += *x * *y; }
        s
    }
}

#[inline]
pub fn axpy_f32(scale: f32, x: &[f32], y: &mut [f32]) {
    if scale == 0.0 {
        return;
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    {
        unsafe { axpy_f32_avx2_inner(scale, x, y) };
        return;
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
    {
        for (yi, xi) in y.iter_mut().zip(x.iter()) { *yi += scale * *xi; }
    }
}

/// BitNet b1.58 AbsMean ternary quantization: `gamma = mean(|W|)`, `Q = clamp(round(W / gamma), -1, 1)`.
pub fn quantize_bitnet_absmean(w: &[f32], q_out: &mut [i8]) -> f32 {
    let n = w.len();
    if n == 0 {
        return 1.0;
    }
    let mut sum_abs = 0.0f32;
    for &v in w {
        sum_abs += v.abs();
    }
    let gamma = (sum_abs / (n as f32)).max(1e-6);
    let inv = 1.0 / gamma;
    for (dst, &v) in q_out.iter_mut().zip(w.iter()) {
        let s = v * inv;
        *dst = if s >= 0.5 { 1 } else if s <= -0.5 { -1 } else { 0 };
    }
    gamma
}

/// 16-wide branchless AVX2+FMA ternary dot product: `sum_i q[i] * x[i]` (`q[i] in {-1, 0, +1}`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn dot_ternary_avx2_inner(q: &[i8], x: &[f32]) -> f32 {
    let n = q.len().min(x.len());
    let mut i = 0usize;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let pq = q.as_ptr();
    let px = x.as_ptr();
    while i + 16 <= n {
        let q128 = _mm_loadu_si128(pq.add(i) as *const __m128i);
        let qf0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q128));
        let qf1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q128, 8)));
        acc0 = _mm256_fmadd_ps(qf0, _mm256_loadu_ps(px.add(i)), acc0);
        acc1 = _mm256_fmadd_ps(qf1, _mm256_loadu_ps(px.add(i + 8)), acc1);
        i += 16;
    }
    while i + 8 <= n {
        let q64 = std::ptr::read_unaligned(pq.add(i) as *const i64);
        let qf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_cvtsi64_si128(q64)));
        acc0 = _mm256_fmadd_ps(qf, _mm256_loadu_ps(px.add(i)), acc0);
        i += 8;
    }
    let mut sum = hsum256_ps(_mm256_add_ps(acc0, acc1));
    while i < n {
        sum += (*pq.add(i) as f32) * *px.add(i);
        i += 1;
    }
    sum
}

/// Branchless 16-wide AVX2+FMA fused BitNet backward row kernel:
/// `dw_row[c] += d * x[c]` AND `dx[c] += g_d * q_row[c]` simultaneously with 0 branch mispredictions!
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn bitnet_bwd_row_avx2(d: f32, g_d: f32, q_row: &[i8], x: &[f32], dx: &mut [f32], dw_row: &mut [f32]) {
    let n = q_row.len();
    let mut i = 0usize;
    let vd = _mm256_set1_ps(d);
    let vgd = _mm256_set1_ps(g_d);
    let pq = q_row.as_ptr();
    let px = x.as_ptr();
    let pdx = dx.as_mut_ptr();
    let pdw = dw_row.as_mut_ptr();
    while i + 16 <= n {
        let q128 = _mm_loadu_si128(pq.add(i) as *const __m128i);
        let qf0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q128));
        let qf1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q128, 8)));

        let vx0 = _mm256_loadu_ps(px.add(i));
        let vx1 = _mm256_loadu_ps(px.add(i + 8));
        let vdw0 = _mm256_loadu_ps(pdw.add(i));
        let vdw1 = _mm256_loadu_ps(pdw.add(i + 8));
        _mm256_storeu_ps(pdw.add(i), _mm256_fmadd_ps(vd, vx0, vdw0));
        _mm256_storeu_ps(pdw.add(i + 8), _mm256_fmadd_ps(vd, vx1, vdw1));

        let vdx0 = _mm256_loadu_ps(pdx.add(i));
        let vdx1 = _mm256_loadu_ps(pdx.add(i + 8));
        _mm256_storeu_ps(pdx.add(i), _mm256_fmadd_ps(vgd, qf0, vdx0));
        _mm256_storeu_ps(pdx.add(i + 8), _mm256_fmadd_ps(vgd, qf1, vdx1));
        i += 16;
    }
    while i + 8 <= n {
        let q64 = std::ptr::read_unaligned(pq.add(i) as *const i64);
        let qf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_cvtsi64_si128(q64)));
        let vx = _mm256_loadu_ps(px.add(i));
        _mm256_storeu_ps(pdw.add(i), _mm256_fmadd_ps(vd, vx, _mm256_loadu_ps(pdw.add(i))));
        _mm256_storeu_ps(pdx.add(i), _mm256_fmadd_ps(vgd, qf, _mm256_loadu_ps(pdx.add(i))));
        i += 8;
    }
    while i < n {
        *pdw.add(i) += d * *px.add(i);
        *pdx.add(i) += g_d * (*pq.add(i) as f32);
        i += 1;
    }
}

/// `y[r] = gamma * sum_c q[r, c] * x[c]` (BitNet b1.58 ternary matrix-vector product).
#[inline]
pub fn matvec_bitnet(q: &[i8], gamma: f32, rows: usize, cols: usize, x: &[f32], y: &mut [f32]) {
    for r in 0..rows {
        let row = &q[r * cols..(r + 1) * cols];
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
        let s = unsafe { dot_ternary_avx2_inner(row, x) };
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
        let s: f32 = row.iter().zip(x.iter()).map(|(&qi, &xi)| (qi as f32) * xi).sum();
        y[r] = s * gamma;
    }
}

/// Backward STE for BitNet linear `y = (gamma * Q) @ x`:
/// accumulates `dx += (gamma * Q)^T @ dy` and `dw += dy (x) x^T` via fused branchless AVX2+FMA!
#[inline]
pub fn matvec_bitnet_bwd(
    q: &[i8],
    gamma: f32,
    rows: usize,
    cols: usize,
    x: &[f32],
    dy: &[f32],
    dx: &mut [f32],
    dw: &mut [f32],
) {
    for r in 0..rows {
        let d = dy[r];
        if d.abs() < 1e-12 {
            continue;
        }
        let g_d = d * gamma;
        let q_row = &q[r * cols..(r + 1) * cols];
        let dw_row = &mut dw[r * cols..(r + 1) * cols];
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
        unsafe { bitnet_bwd_row_avx2(d, g_d, q_row, x, dx, dw_row); }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
        {
            for c in 0..cols {
                dw_row[c] += d * x[c];
                dx[c] += g_d * (q_row[c] as f32);
            }
        }
    }
}

/// RMSNorm forward: `y = (x / rms) * scale`, returns `inv_rms`.
#[inline]
pub fn rmsnorm_fwd(x: &[f32], scale: &[f32], y: &mut [f32]) -> f32 {
    let n = x.len() as f32;
    let ss = dot_f32(x, x) / n + 1e-5;
    let inv = 1.0 / ss.sqrt();
    for i in 0..x.len() {
        y[i] = x[i] * inv * scale[i];
    }
    inv
}

/// RMSNorm backward accumulating into `dx` and `dscale`.
#[inline]
pub fn rmsnorm_bwd(x: &[f32], scale: &[f32], inv: f32, dy: &[f32], dx: &mut [f32], dscale: &mut [f32]) {
    let n = x.len();
    let mut dot_term = 0.0f32;
    for i in 0..n {
        let xhat = x[i] * inv;
        dscale[i] += dy[i] * xhat;
        dot_term += dy[i] * scale[i] * xhat;
    }
    let coeff = dot_term / (n as f32);
    for i in 0..n {
        let xhat = x[i] * inv;
        dx[i] += inv * (dy[i] * scale[i] - xhat * coeff);
    }
}
