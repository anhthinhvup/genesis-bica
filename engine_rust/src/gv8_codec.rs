//! Genesis-v8 Packed Full-RAM `GV8MOD01` Binary Loader & Streaming Converter (`gv8_codec.rs`).
//! Drops inference-unused `BitLinear.w:f32` & `PEER vu/vv`, stores BitNet ternary 2-bit block-32
//! bit-exact (`rel_err == 0.0`) and PEER `Q8 per-row` Full-RAM (~407 MiB for `ultra_445m`, 0 SSD miss).
//! Zero external crates | Strictly <= 240 LOC.

use std::fs::File;
use std::io::{BufReader, BufWriter, Error, ErrorKind, Read, Result, Write};

use crate::attn_rope::SlidingWindowAttn;
use crate::gv8_kernels::{f16_to_f32, f32_to_f16, pack_2bit_block32};
use crate::model::{BitLinear, GenesisV6Model, ModelConfig, SparseBlock};
use crate::peer_delta_sidecar::PeerDeltaSidecar;
use crate::peer_pk::{PeerPkLayer, PeerQ8Storage, Q14_INV};
use crate::simd_avx2::quantize_bitnet_absmean;

pub const GV8_MAGIC: &[u8; 8] = b"GV8MOD01";
pub const BITNET_FMT_I2_B32: u32 = 1;
pub const PEER_FMT_Q8_F16: u32 = 1;
pub const PEER_FMT_Q8_F32: u32 = 2;
pub const PEER_FMT_Q14_I16: u32 = 3;
pub const DENSE_FMT_F32: u32 = 0;

fn read_pod_vec<R: Read, T: Copy + Default>(r: &mut R, len: usize) -> Result<Vec<T>> {
    let mut v = vec![T::default(); len];
    #[cfg(target_endian = "little")]
    unsafe {
        let bytes = std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, len * std::mem::size_of::<T>());
        r.read_exact(bytes)?;
    }
    Ok(v)
}

fn write_pod_slice<W: Write, T: Copy>(w: &mut W, s: &[T]) -> Result<()> {
    #[cfg(target_endian = "little")]
    unsafe {
        let bytes = std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * std::mem::size_of::<T>());
        w.write_all(bytes)?;
    }
    Ok(())
}

fn read_u32_le<R: Read>(r: &mut R) -> Result<u32> {
    let mut b = [0u8; 4]; r.read_exact(&mut b)?; Ok(u32::from_le_bytes(b))
}

fn read_f32_le<R: Read>(r: &mut R) -> Result<f32> {
    let mut b = [0u8; 4]; r.read_exact(&mut b)?; Ok(f32::from_le_bytes(b))
}

fn read_bitlinear_2bit<R: Read>(r: &mut R, rows: usize, cols: usize) -> Result<BitLinear> {
    let gamma = read_f32_le(r)?;
    let q2 = read_pod_vec::<R, u8>(r, rows * ((cols + 3) / 4))?;
    Ok(BitLinear::from_packed_2bit(q2, gamma))
}

/// Loads `GV8MOD01` directly in-place with 0 giant temp buffers (`~407 MiB` RAM for `ultra_445m`).
pub fn load_gv8(path: &str) -> Result<GenesisV6Model> {
    let mut r = BufReader::with_capacity(1 << 18, File::open(path)?);
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != GV8_MAGIC {
        return Err(Error::new(ErrorKind::InvalidData, "not GV8MOD01"));
    }
    let mut u = [0u32; 14];
    for slot in &mut u { *slot = read_u32_le(&mut r)?; }
    let cfg = ModelConfig {
        vocab: u[0] as usize, d_model: u[1] as usize, n_layers: u[2] as usize,
        n_experts: u[3] as usize, top_k: u[4] as usize, d_ff: u[5] as usize,
        peer_sub_k: u[6] as usize, peer_heads: u[7] as usize, peer_k: u[8] as usize,
        recur_steps: (u[9] as usize).max(1), attn_window: u[10] as usize,
    };
    let (bitnet_fmt, peer_fmt, dense_fmt) = (u[11], u[12], u[13]);
    if bitnet_fmt != BITNET_FMT_I2_B32 || dense_fmt != DENSE_FMT_F32
        || (peer_fmt != PEER_FMT_Q8_F16 && peer_fmt != PEER_FMT_Q8_F32 && peer_fmt != PEER_FMT_Q14_I16)
    {
        return Err(Error::new(ErrorKind::InvalidData, "unsupported GV8 format flags"));
    }
    let (d, df, ne) = (cfg.d_model, cfg.d_ff, cfg.n_experts);
    let emb = read_pod_vec::<_, f32>(&mut r, cfg.vocab * d)?;
    let norm_f = read_pod_vec::<_, f32>(&mut r, d)?;
    let recur_inj_alpha = read_pod_vec::<_, f32>(&mut r, d)?;

    let mut layers = Vec::with_capacity(cfg.n_layers);
    for _ in 0..cfg.n_layers {
        let norm1 = read_pod_vec::<_, f32>(&mut r, d)?;
        let conv_w = read_pod_vec::<_, f32>(&mut r, 3 * d)?;
        let wz = read_bitlinear_2bit(&mut r, d, d)?;
        let wh = read_bitlinear_2bit(&mut r, d, d)?;
        let wo = read_bitlinear_2bit(&mut r, d, d)?;
        let norm2 = read_pod_vec::<_, f32>(&mut r, d)?;
        let router = read_pod_vec::<_, f32>(&mut r, ne * d)?;
        let mut exp_up = Vec::with_capacity(ne);
        let mut exp_dn = Vec::with_capacity(ne);
        for _ in 0..ne { exp_up.push(read_bitlinear_2bit(&mut r, df, d)?); }
        for _ in 0..ne { exp_dn.push(read_bitlinear_2bit(&mut r, d, df)?); }
        layers.push(SparseBlock { norm1, conv_w, wz, wh, wo, norm2, router, exp_up, exp_dn });
    }

    let n_peers = if cfg.peer_sub_k > 0 { if cfg.n_layers >= 4 { 2 } else { 1 } } else { 0 };
    let mut peers = Vec::with_capacity(n_peers);
    for _ in 0..n_peers {
        let nh = cfg.peer_heads.max(1);
        let sk = cfg.peer_sub_k;
        let sd = (d / nh) / 2;
        let n_exp = sk * sk;
        let norm = read_pod_vec::<_, f32>(&mut r, d)?;
        let wq = read_pod_vec::<_, f32>(&mut r, d * d)?;
        let keys_a = read_pod_vec::<_, f32>(&mut r, nh * sk * sd)?;
        let keys_b = read_pod_vec::<_, f32>(&mut r, nh * sk * sd)?;
        let (u_exp, v_exp, q8) = match peer_fmt {
            PEER_FMT_Q8_F16 => {
                let u_sc_f16 = read_pod_vec::<_, u16>(&mut r, n_exp)?;
                let v_sc_f16 = read_pod_vec::<_, u16>(&mut r, n_exp)?;
                let u_q8 = read_pod_vec::<_, i8>(&mut r, n_exp * d)?;
                let v_q8 = read_pod_vec::<_, i8>(&mut r, n_exp * d)?;
                (Vec::new(), Vec::new(), Some(PeerQ8Storage {
                    u_q8, v_q8, u_sc_f16, v_sc_f16, u_sc_f32: Vec::new(), v_sc_f32: Vec::new(),
                }))
            }
            PEER_FMT_Q8_F32 => {
                let u_sc_f32 = read_pod_vec::<_, f32>(&mut r, n_exp)?;
                let v_sc_f32 = read_pod_vec::<_, f32>(&mut r, n_exp)?;
                let u_q8 = read_pod_vec::<_, i8>(&mut r, n_exp * d)?;
                let v_q8 = read_pod_vec::<_, i8>(&mut r, n_exp * d)?;
                (Vec::new(), Vec::new(), Some(PeerQ8Storage {
                    u_q8, v_q8, u_sc_f16: Vec::new(), v_sc_f16: Vec::new(), u_sc_f32, v_sc_f32,
                }))
            }
            _ => {
                let u_exp = read_pod_vec::<_, i16>(&mut r, n_exp * d)?;
                let v_exp = read_pod_vec::<_, i16>(&mut r, n_exp * d)?;
                (u_exp, v_exp, None)
            }
        };
        peers.push(PeerPkLayer {
            d_model: d, n_heads: nh, sub_k: sk, k_head: cfg.peer_k.max(1),
            norm, wq, keys_a, keys_b, u_exp, v_exp, vu: Vec::new(), vv: Vec::new(), q8,
        });
    }

    let attn = if cfg.attn_window > 0 {
        let norm = read_pod_vec::<_, f32>(&mut r, d)?;
        let wq = read_pod_vec::<_, f32>(&mut r, d * d)?;
        let wk = read_pod_vec::<_, f32>(&mut r, d * d)?;
        let wv = read_pod_vec::<_, f32>(&mut r, d * d)?;
        let wo = read_pod_vec::<_, f32>(&mut r, d * d)?;
        Some(SlidingWindowAttn { d_model: d, n_heads: cfg.peer_heads.max(1), window: cfg.attn_window.max(8), norm, wq, wk, wv, wo })
    } else { None };

    Ok(GenesisV6Model { cfg, emb, norm_f, recur_inj_alpha, layers, peers, attn, peer_delta: PeerDeltaSidecar::empty() })
}

fn quantize_peer_q14_to_q8(src: &[i16], n_exp: usize, d: usize, use_f16: bool) -> (Vec<u16>, Vec<f32>, Vec<i8>) {
    let mut sc16 = if use_f16 { vec![0u16; n_exp] } else { Vec::new() };
    let mut sc32 = if !use_f16 { vec![0.0f32; n_exp] } else { Vec::new() };
    let mut out = vec![0i8; n_exp * d];
    for e in 0..n_exp {
        let row = &src[e * d..(e + 1) * d];
        let mut mx = 0i32;
        for &q in row { let a = (q as i32).abs(); if a > mx { mx = a; } }
        if mx == 0 { continue; }
        let raw_sc = ((mx as f32) * Q14_INV) / 127.0;
        let act_sc = if use_f16 {
            let h = f32_to_f16(raw_sc); sc16[e] = h; f16_to_f32(h)
        } else {
            sc32[e] = raw_sc; raw_sc
        };
        if act_sc <= 0.0 { continue; }
        let inv = Q14_INV / act_sc;
        let dst = &mut out[e * d..(e + 1) * d];
        for c in 0..d { dst[c] = ((row[c] as f32) * inv).round().clamp(-127.0, 127.0) as i8; }
    }
    (sc16, sc32, out)
}

/// Streams `GV6MOD01..03` -> `GV8MOD01` bit-exact BitNet 2-bit + Full-RAM PEER Q8 (low peak RAM!).
pub fn convert_gv6_to_gv8(in_path: &str, out_path: &str, peer_fmt: u32) -> Result<()> {
    if peer_fmt != PEER_FMT_Q8_F16 && peer_fmt != PEER_FMT_Q8_F32 && peer_fmt != PEER_FMT_Q14_I16 {
        return Err(Error::new(ErrorKind::InvalidInput, "invalid peer_fmt"));
    }
    let mut r = BufReader::with_capacity(1 << 20, File::open(in_path)?);
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    let ver: u8 = match &magic { b"GV6MOD01" => 1, b"GV6MOD02" => 2, b"GV6MOD03" => 3, _ => return Err(Error::new(ErrorKind::InvalidData, "bad GV6 magic")) };
    let n_hdr = match ver { 1 => 6, 2 => 10, _ => 11 };
    let mut u = [0u32; 11];
    for i in 0..n_hdr { u[i] = read_u32_le(&mut r)?; }
    let cfg = ModelConfig {
        vocab: u[0] as usize, d_model: u[1] as usize, n_layers: u[2] as usize,
        n_experts: u[3] as usize, top_k: u[4] as usize, d_ff: u[5] as usize,
        peer_sub_k: if ver >= 2 { u[6] as usize } else { 0 },
        peer_heads: if ver >= 2 { u[7] as usize } else { 4 },
        peer_k: if ver >= 2 { u[8] as usize } else { 8 },
        recur_steps: if ver >= 2 { (u[9] as usize).max(1) } else { 1 },
        attn_window: if ver >= 3 { u[10] as usize } else { 0 },
    };

    let tmp_path = format!("{}.tmp", out_path);
    let mut w = BufWriter::with_capacity(1 << 20, File::create(&tmp_path)?);
    w.write_all(GV8_MAGIC)?;
    for &v in &[
        cfg.vocab as u32, cfg.d_model as u32, cfg.n_layers as u32, cfg.n_experts as u32,
        cfg.top_k as u32, cfg.d_ff as u32, cfg.peer_sub_k as u32, cfg.peer_heads as u32,
        cfg.peer_k as u32, cfg.recur_steps as u32, cfg.attn_window as u32,
        BITNET_FMT_I2_B32, peer_fmt, DENSE_FMT_F32,
    ] { w.write_all(&v.to_le_bytes())?; }

    let (d, df, ne) = (cfg.d_model, cfg.d_ff, cfg.n_experts);
    let copy_f32 = |r: &mut BufReader<File>, w: &mut BufWriter<File>, n: usize| -> Result<()> {
        let buf = read_pod_vec::<_, f32>(r, n)?; write_pod_slice(w, &buf)
    };
    copy_f32(&mut r, &mut w, cfg.vocab * d)?;
    copy_f32(&mut r, &mut w, d)?;
    if ver >= 2 { copy_f32(&mut r, &mut w, d)?; } else { write_pod_slice(&mut w, &vec![0.25f32; d])?; }

    let conv_bl = |r: &mut BufReader<File>, w: &mut BufWriter<File>, rows: usize, cols: usize| -> Result<()> {
        let wf = read_pod_vec::<_, f32>(r, rows * cols)?;
        let mut qi = vec![0i8; rows * cols];
        let gamma = quantize_bitnet_absmean(&wf, &mut qi);
        let q2 = pack_2bit_block32(&qi, rows, cols);
        w.write_all(&gamma.to_le_bytes())?;
        w.write_all(&q2)
    };
    for _ in 0..cfg.n_layers {
        copy_f32(&mut r, &mut w, d)?; copy_f32(&mut r, &mut w, 3 * d)?;
        conv_bl(&mut r, &mut w, d, d)?; conv_bl(&mut r, &mut w, d, d)?; conv_bl(&mut r, &mut w, d, d)?;
        copy_f32(&mut r, &mut w, d)?; copy_f32(&mut r, &mut w, ne * d)?;
        for _ in 0..ne { conv_bl(&mut r, &mut w, df, d)?; }
        for _ in 0..ne { conv_bl(&mut r, &mut w, d, df)?; }
    }

    let n_peers = if cfg.peer_sub_k > 0 { if cfg.n_layers >= 4 { 2 } else { 1 } } else { 0 };
    for _ in 0..n_peers {
        let nh = cfg.peer_heads.max(1); let sk = cfg.peer_sub_k; let sd = (d / nh) / 2; let n_exp = sk * sk;
        copy_f32(&mut r, &mut w, d)?; copy_f32(&mut r, &mut w, d * d)?;
        copy_f32(&mut r, &mut w, nh * sk * sd)?; copy_f32(&mut r, &mut w, nh * sk * sd)?;
        let _vu = read_pod_vec::<_, f32>(&mut r, n_exp)?;
        let _vv = read_pod_vec::<_, f32>(&mut r, n_exp)?;
        let u16_src = read_pod_vec::<_, i16>(&mut r, n_exp * d)?;
        let v16_src = read_pod_vec::<_, i16>(&mut r, n_exp * d)?;
        match peer_fmt {
            PEER_FMT_Q8_F16 => {
                let (us16, _, uq8) = quantize_peer_q14_to_q8(&u16_src, n_exp, d, true);
                let (vs16, _, vq8) = quantize_peer_q14_to_q8(&v16_src, n_exp, d, true);
                write_pod_slice(&mut w, &us16)?; write_pod_slice(&mut w, &vs16)?;
                write_pod_slice(&mut w, &uq8)?; write_pod_slice(&mut w, &vq8)?;
            }
            PEER_FMT_Q8_F32 => {
                let (_, us32, uq8) = quantize_peer_q14_to_q8(&u16_src, n_exp, d, false);
                let (_, vs32, vq8) = quantize_peer_q14_to_q8(&v16_src, n_exp, d, false);
                write_pod_slice(&mut w, &us32)?; write_pod_slice(&mut w, &vs32)?;
                write_pod_slice(&mut w, &uq8)?; write_pod_slice(&mut w, &vq8)?;
            }
            _ => { write_pod_slice(&mut w, &u16_src)?; write_pod_slice(&mut w, &v16_src)?; }
        }
    }
    if cfg.attn_window > 0 { copy_f32(&mut r, &mut w, d + 4 * d * d)?; }
    w.flush()?;
    drop(w);
    let _ = std::fs::remove_file(out_path);
    std::fs::rename(&tmp_path, out_path)
}
