//! Genesis-v8 AVX2+FMA Quantized Inference Kernels (`gv8_kernels.rs`, Spec #87).
//! Contains: IEEE-754 `f16` codec, bit-exact 2-bit block-32 BitNet 4-row AVX2 matvec (`rel_err == 0.0`),
//! bit-exact 4-row AVX2 `f32` matvec / `lm_head`, and PEER `Q8 per-row` AVX2 `dot` / `axpy`.
//! Zero external crates | Strictly <= 340 LOC.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::generate::logit_softcap;
use crate::simd_avx2::dot_f32;

/// Exact IEEE-754 binary16 -> f32 conversion (normals, subnormals, zero, inf/nan).
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x03FF) as u32;
    if exp == 0 {
        if mant == 0 { return f32::from_bits(sign); }
        let v = (mant as f32) * 5.9604645e-8f32;
        return if sign != 0 { -v } else { v };
    }
    if exp == 31 { return f32::from_bits(sign | (0xFF << 23) | (mant << 13)); }
    f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
}

/// Exact IEEE-754 f32 -> binary16 round-to-nearest-even (clamps overflow to max finite 65504.0).
#[inline]
pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;
    if exp == 255 { return sign | 0x7C00 | if mant != 0 { 0x0200 } else { 0 }; }
    let mut new_exp = exp - 112;
    if new_exp >= 31 { return sign | 0x7BFF; }
    if new_exp <= 0 {
        if new_exp < -10 { return sign; }
        let m = mant | 0x0080_0000;
        let shift = (14 - new_exp) as u32;
        let half = 1u32 << (shift - 1);
        let rem = m & ((1u32 << shift) - 1);
        let mut q = m >> shift;
        if rem > half || (rem == half && (q & 1) != 0) { q += 1; }
        return sign | (q as u16);
    }
    let rem = mant & 0x1FFF;
    let mut m10 = mant >> 13;
    if rem > 0x1000 || (rem == 0x1000 && (m10 & 1) != 0) { m10 += 1; }
    if m10 == 0x0400 {
        m10 = 0; new_exp += 1;
        if new_exp >= 31 { return sign | 0x7BFF; }
    }
    sign | ((new_exp as u16) << 10) | (m10 as u16)
}

/// Packs ternary `q: &[i8]` (`{-1, 0, +1}`) into 2-bit block-32 format (`{00:0, 01:+1, 10:-1}`).
pub fn pack_2bit_block32(q: &[i8], rows: usize, cols: usize) -> Vec<u8> {
    let row_bytes = (cols + 3) / 4;
    let mut out = vec![0u8; rows * row_bytes];
    let enc = |v: i8| -> u8 { if v > 0 { 1 } else if v < 0 { 2 } else { 0 } };
    for r in 0..rows {
        let src = &q[r * cols..(r + 1) * cols];
        let dst = &mut out[r * row_bytes..(r + 1) * row_bytes];
        let (mut c, mut bi) = (0usize, 0usize);
        while c + 32 <= cols {
            for j in 0..8 {
                dst[bi + j] = enc(src[c + j])
                    | (enc(src[c + j + 8]) << 2)
                    | (enc(src[c + j + 16]) << 4)
                    | (enc(src[c + j + 24]) << 6);
            }
            c += 32; bi += 8;
        }
        while c < cols {
            let mut b = 0u8;
            for k in 0..4 { if c + k < cols { b |= enc(src[c + k]) << (2 * k); } }
            dst[bi] = b; bi += 1; c += 4;
        }
    }
    out
}

#[inline]
fn decode_2bit(code: u8) -> f32 { match code & 3 { 1 => 1.0, 2 => -1.0, _ => 0.0 } }

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

#[inline]
fn dot_2bit_row_tail(row: &[u8], cols: usize, x: &[f32], c_start: usize, bi_start: usize) -> f32 {
    let (mut s, mut c, mut bi) = (0.0f32, c_start, bi_start);
    while c < cols {
        let b = row[bi];
        for k in 0..4 { if c + k < cols { s += decode_2bit(b >> (2 * k)) * x[c + k]; } }
        bi += 1; c += 4;
    }
    s
}

/// Bit-exact 4-row AVX2+FMA 2-bit ternary matvec (`max_abs_diff == 0.0` vs `matvec_bitnet` `i8`!).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matvec_bitnet_2bit_avx2(q2: &[u8], gamma: f32, rows: usize, cols: usize, x: &[f32], y: &mut [f32]) {
    let row_bytes = (cols + 3) / 4;
    let px = x.as_ptr();
    let mask3 = _mm256_set1_epi8(0x03);
    let lut = _mm256_setr_epi8(
        0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    );
    let mut r = 0usize;
    while r + 4 <= rows {
        let p0 = q2.as_ptr().add(r * row_bytes);
        let p1 = q2.as_ptr().add((r + 1) * row_bytes);
        let p2 = q2.as_ptr().add((r + 2) * row_bytes);
        let p3 = q2.as_ptr().add((r + 3) * row_bytes);
        let (mut a00, mut a01) = (_mm256_setzero_ps(), _mm256_setzero_ps());
        let (mut a10, mut a11) = (_mm256_setzero_ps(), _mm256_setzero_ps());
        let (mut a20, mut a21) = (_mm256_setzero_ps(), _mm256_setzero_ps());
        let (mut a30, mut a31) = (_mm256_setzero_ps(), _mm256_setzero_ps());
        let (mut c, mut bi) = (0usize, 0usize);
        while c + 32 <= cols {
            let v = _mm256_set_epi64x(
                std::ptr::read_unaligned(p3.add(bi) as *const i64),
                std::ptr::read_unaligned(p2.add(bi) as *const i64),
                std::ptr::read_unaligned(p1.add(bi) as *const i64),
                std::ptr::read_unaligned(p0.add(bi) as *const i64),
            );
            let q0 = _mm256_shuffle_epi8(lut, _mm256_and_si256(v, mask3));
            let q1 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(v, 2), mask3));
            let q2v = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(v, 4), mask3));
            let q3 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(v, 6), mask3));
            macro_rules! fma4 {
                ($qv:expr, $xv:expr, $b0:expr, $b1:expr, $b2:expr, $b3:expr) => {{
                    let lo = _mm256_castsi256_si128($qv);
                    let hi = _mm256_extracti128_si256($qv, 1);
                    $b0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)), $xv, $b0);
                    $b1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8))), $xv, $b1);
                    $b2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)), $xv, $b2);
                    $b3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8))), $xv, $b3);
                }};
            }
            fma4!(q0, _mm256_loadu_ps(px.add(c)), a00, a10, a20, a30);
            fma4!(q1, _mm256_loadu_ps(px.add(c + 8)), a01, a11, a21, a31);
            fma4!(q2v, _mm256_loadu_ps(px.add(c + 16)), a00, a10, a20, a30);
            fma4!(q3, _mm256_loadu_ps(px.add(c + 24)), a01, a11, a21, a31);
            c += 32; bi += 8;
        }
        let mut s = [
            hsum256_ps(_mm256_add_ps(a00, a01)), hsum256_ps(_mm256_add_ps(a10, a11)),
            hsum256_ps(_mm256_add_ps(a20, a21)), hsum256_ps(_mm256_add_ps(a30, a31)),
        ];
        if c < cols {
            for k in 0..4 { s[k] += dot_2bit_row_tail(&q2[(r + k) * row_bytes..(r + k + 1) * row_bytes], cols, x, c, bi); }
        }
        for k in 0..4 { y[r + k] = s[k] * gamma; }
        r += 4;
    }
    let mask3_128 = _mm_set1_epi8(0x03);
    let lut128 = _mm_setr_epi8(0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    while r < rows {
        let pq = q2.as_ptr().add(r * row_bytes);
        let (mut a0, mut a1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
        let (mut c, mut bi) = (0usize, 0usize);
        while c + 32 <= cols {
            let v = _mm_cvtsi64_si128(std::ptr::read_unaligned(pq.add(bi) as *const i64));
            let q0 = _mm_shuffle_epi8(lut128, _mm_and_si128(v, mask3_128));
            let q1 = _mm_shuffle_epi8(lut128, _mm_and_si128(_mm_srli_epi16(v, 2), mask3_128));
            let q2v = _mm_shuffle_epi8(lut128, _mm_and_si128(_mm_srli_epi16(v, 4), mask3_128));
            let q3 = _mm_shuffle_epi8(lut128, _mm_and_si128(_mm_srli_epi16(v, 6), mask3_128));
            a0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q0)), _mm256_loadu_ps(px.add(c)), a0);
            a1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q1)), _mm256_loadu_ps(px.add(c + 8)), a1);
            a0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q2v)), _mm256_loadu_ps(px.add(c + 16)), a0);
            a1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q3)), _mm256_loadu_ps(px.add(c + 24)), a1);
            c += 32; bi += 8;
        }
        let mut s = hsum256_ps(_mm256_add_ps(a0, a1));
        if c < cols { s += dot_2bit_row_tail(&q2[r * row_bytes..(r + 1) * row_bytes], cols, x, c, bi); }
        y[r] = s * gamma;
        r += 1;
    }
}

#[inline]
pub fn matvec_bitnet_2bit(q2: &[u8], gamma: f32, rows: usize, cols: usize, x: &[f32], y: &mut [f32]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    { unsafe { matvec_bitnet_2bit_avx2(q2, gamma, rows, cols, x, y); } return; }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
    {
        let row_bytes = (cols + 3) / 4;
        for r in 0..rows {
            let row = &q2[r * row_bytes..(r + 1) * row_bytes];
            let (mut s, mut c, mut bi) = (0.0f32, 0usize, 0usize);
            while c + 32 <= cols {
                for j in 0..8 {
                    let b = row[bi + j];
                    s += decode_2bit(b) * x[c + j] + decode_2bit(b >> 2) * x[c + j + 8]
                        + decode_2bit(b >> 4) * x[c + j + 16] + decode_2bit(b >> 6) * x[c + j + 24];
                }
                c += 32; bi += 8;
            }
            y[r] = (s + dot_2bit_row_tail(row, cols, x, c, bi)) * gamma;
        }
    }
}

/// Bit-exact 4-row AVX2+FMA `f32` matvec (`y = W @ x`, identical FMA order to `dot_f32`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matvec_f32_4row_avx2(w: &[f32], rows: usize, cols: usize, x: &[f32], y: &mut [f32], softcap: bool) {
    let px = x.as_ptr();
    let mut r = 0usize;
    while r + 4 <= rows {
        let p0 = w.as_ptr().add(r * cols); let p1 = w.as_ptr().add((r + 1) * cols);
        let p2 = w.as_ptr().add((r + 2) * cols); let p3 = w.as_ptr().add((r + 3) * cols);
        let (mut a00, mut a01) = (_mm256_setzero_ps(), _mm256_setzero_ps());
        let (mut a10, mut a11) = (_mm256_setzero_ps(), _mm256_setzero_ps());
        let (mut a20, mut a21) = (_mm256_setzero_ps(), _mm256_setzero_ps());
        let (mut a30, mut a31) = (_mm256_setzero_ps(), _mm256_setzero_ps());
        let mut i = 0usize;
        while i + 16 <= cols {
            let x0 = _mm256_loadu_ps(px.add(i));
            let x1 = _mm256_loadu_ps(px.add(i + 8));
            a00 = _mm256_fmadd_ps(_mm256_loadu_ps(p0.add(i)), x0, a00);
            a01 = _mm256_fmadd_ps(_mm256_loadu_ps(p0.add(i + 8)), x1, a01);
            a10 = _mm256_fmadd_ps(_mm256_loadu_ps(p1.add(i)), x0, a10);
            a11 = _mm256_fmadd_ps(_mm256_loadu_ps(p1.add(i + 8)), x1, a11);
            a20 = _mm256_fmadd_ps(_mm256_loadu_ps(p2.add(i)), x0, a20);
            a21 = _mm256_fmadd_ps(_mm256_loadu_ps(p2.add(i + 8)), x1, a21);
            a30 = _mm256_fmadd_ps(_mm256_loadu_ps(p3.add(i)), x0, a30);
            a31 = _mm256_fmadd_ps(_mm256_loadu_ps(p3.add(i + 8)), x1, a31);
            i += 16;
        }
        while i + 8 <= cols {
            let x0 = _mm256_loadu_ps(px.add(i));
            a00 = _mm256_fmadd_ps(_mm256_loadu_ps(p0.add(i)), x0, a00);
            a10 = _mm256_fmadd_ps(_mm256_loadu_ps(p1.add(i)), x0, a10);
            a20 = _mm256_fmadd_ps(_mm256_loadu_ps(p2.add(i)), x0, a20);
            a30 = _mm256_fmadd_ps(_mm256_loadu_ps(p3.add(i)), x0, a30);
            i += 8;
        }
        let mut s = [
            hsum256_ps(_mm256_add_ps(a00, a01)), hsum256_ps(_mm256_add_ps(a10, a11)),
            hsum256_ps(_mm256_add_ps(a20, a21)), hsum256_ps(_mm256_add_ps(a30, a31)),
        ];
        while i < cols {
            let xi = *px.add(i);
            s[0] += *p0.add(i) * xi; s[1] += *p1.add(i) * xi;
            s[2] += *p2.add(i) * xi; s[3] += *p3.add(i) * xi;
            i += 1;
        }
        for k in 0..4 { y[r + k] = if softcap { logit_softcap(s[k]) } else { s[k] }; }
        r += 4;
    }
    while r < rows {
        let s = dot_f32(&w[r * cols..(r + 1) * cols], x);
        y[r] = if softcap { logit_softcap(s) } else { s };
        r += 1;
    }
}

#[inline]
pub fn matvec_f32_fast(w: &[f32], rows: usize, cols: usize, x: &[f32], y: &mut [f32]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    { unsafe { matvec_f32_4row_avx2(w, rows, cols, x, y, false); } return; }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
    for r in 0..rows { y[r] = dot_f32(&w[r * cols..(r + 1) * cols], x); }
}

#[inline]
pub fn lm_head_softcap_fast(emb: &[f32], v: usize, d: usize, xf: &[f32], out: &mut [f32]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    { unsafe { matvec_f32_4row_avx2(emb, v, d, xf, out, true); } return; }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
    for r in 0..v { out[r] = logit_softcap(dot_f32(&emb[r * d..(r + 1) * d], xf)); }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn q8_dot_avx2(q: &[i8], x: &[f32], scale: f32) -> f32 {
    let n = q.len().min(x.len());
    let (mut i, mut acc0, mut acc1) = (0usize, _mm256_setzero_ps(), _mm256_setzero_ps());
    let (pq, px) = (q.as_ptr(), x.as_ptr());
    while i + 16 <= n {
        let q128 = _mm_loadu_si128(pq.add(i) as *const __m128i);
        let q0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q128));
        let q1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q128, 8)));
        acc0 = _mm256_fmadd_ps(q0, _mm256_loadu_ps(px.add(i)), acc0);
        acc1 = _mm256_fmadd_ps(q1, _mm256_loadu_ps(px.add(i + 8)), acc1);
        i += 16;
    }
    let mut s = hsum256_ps(_mm256_add_ps(acc0, acc1));
    while i < n { s += (*pq.add(i) as f32) * *px.add(i); i += 1; }
    s * scale
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn q8_axpy_avx2(scale: f32, q: &[i8], y: &mut [f32]) {
    let n = q.len().min(y.len());
    let vs = _mm256_set1_ps(scale);
    let (pq, py, mut i) = (q.as_ptr(), y.as_mut_ptr(), 0usize);
    while i + 16 <= n {
        let q128 = _mm_loadu_si128(pq.add(i) as *const __m128i);
        let q0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q128));
        let q1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q128, 8)));
        _mm256_storeu_ps(py.add(i), _mm256_fmadd_ps(vs, q0, _mm256_loadu_ps(py.add(i))));
        _mm256_storeu_ps(py.add(i + 8), _mm256_fmadd_ps(vs, q1, _mm256_loadu_ps(py.add(i + 8))));
        i += 16;
    }
    while i < n { *py.add(i) += scale * (*pq.add(i) as f32); i += 1; }
}

#[inline]
pub fn q8_dot(q: &[i8], x: &[f32], scale: f32) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    { return unsafe { q8_dot_avx2(q, x, scale) }; }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
    { q.iter().zip(x.iter()).map(|(&qi, &xi)| (qi as f32) * xi).sum::<f32>() * scale }
}

#[inline]
pub fn q8_axpy(scale: f32, q: &[i8], y: &mut [f32]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    { unsafe { q8_axpy_avx2(scale, q, y); } return; }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
    { for (yi, &qi) in y.iter_mut().zip(q.iter()) { *yi += scale * (qi as f32); } }
}
