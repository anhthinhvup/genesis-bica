//! Genesis-v8 Sparse f32 PEER Delta Sidecar (`peer_delta_sidecar.rs`, Spec #91 Pillar 1).
//! Loads `sleep_peer_delta.bin` (`< 20 MiB`, magic `GV8DLT01`) and applies sparse high-precision
//! `f32` deltas on top of read-only `GV8MOD01` Q8 (or `GV6MOD03` Q14) PEER experts at Wake time
//! with O(1) open-addressing lookup, ZERO heap allocation per token, and ZERO base-model reload!
//! Zero external crates | Strictly <= 240 LOC.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

pub const DELTA_MAGIC: &[u8; 8] = b"GV8DLT01";
pub const DELTA_VERSION: u32 = 1;
pub const DELTA_HEADER_BYTES: usize = 32;
pub const MAX_SIDECAR_BYTES: u64 = 20 * 1024 * 1024; // 20 MiB strict cap
const EMPTY_KEY: u32 = u32::MAX;

/// Single sparse PEER expert delta row (matches `sleep_peer_delta.bin` on-disk record).
#[allow(dead_code)]
#[derive(Clone)]
pub struct PeerDeltaRecord {
    pub peer_idx: u32,
    pub expert_id: u32,
    pub vu_local: f32,
    pub vv_local: f32,
    pub delta_u: Vec<f32>,
    pub delta_v: Vec<f32>,
}

/// Zero-alloc O(1) flat open-addressing lookup table for one PEER layer's sparse `f32` deltas.
#[derive(Clone, Default)]
pub struct PeerDeltaTable {
    d_model: usize,
    n_rows: usize,
    keys: Vec<u32>,
    vals: Vec<u32>,
    mask: usize,
    delta_u: Vec<f32>,
    delta_v: Vec<f32>,
}

impl PeerDeltaTable {
    pub fn empty(d_model: usize) -> Self {
        Self { d_model, n_rows: 0, keys: Vec::new(), vals: Vec::new(), mask: 0, delta_u: Vec::new(), delta_v: Vec::new() }
    }

    fn with_capacity(d_model: usize, cap_rows: usize) -> Self {
        if cap_rows == 0 { return Self::empty(d_model); }
        let mut tbl_len = 16usize;
        while tbl_len < cap_rows.saturating_mul(2) { tbl_len <<= 1; }
        Self {
            d_model, n_rows: 0,
            keys: vec![EMPTY_KEY; tbl_len], vals: vec![0u32; tbl_len], mask: tbl_len - 1,
            delta_u: Vec::with_capacity(cap_rows * d_model),
            delta_v: Vec::with_capacity(cap_rows * d_model),
        }
    }

    fn insert_or_replace(&mut self, expert_id: u32, du: &[f32], dv: &[f32]) {
        if self.keys.is_empty() { return; }
        let mut slot = (expert_id.wrapping_mul(0x9E37_79B1)) as usize & self.mask;
        loop {
            let k = self.keys[slot];
            if k == EMPTY_KEY {
                self.keys[slot] = expert_id;
                self.vals[slot] = self.n_rows as u32;
                self.delta_u.extend_from_slice(du);
                self.delta_v.extend_from_slice(dv);
                self.n_rows += 1;
                return;
            }
            if k == expert_id {
                let off = (self.vals[slot] as usize) * self.d_model;
                self.delta_u[off..off + self.d_model].copy_from_slice(du);
                self.delta_v[off..off + self.d_model].copy_from_slice(dv);
                return;
            }
            slot = (slot + 1) & self.mask;
        }
    }

    #[inline] pub fn len(&self) -> usize { self.n_rows }
    #[inline] pub fn is_empty(&self) -> bool { self.n_rows == 0 }

    /// O(1), 0-alloc lookup returning `(delta_u, delta_v)` slices of length `d_model`.
    #[inline]
    pub fn lookup(&self, expert_id: u32) -> Option<(&[f32], &[f32])> {
        if self.n_rows == 0 { return None; }
        let mut slot = (expert_id.wrapping_mul(0x9E37_79B1)) as usize & self.mask;
        loop {
            let k = self.keys[slot];
            if k == expert_id {
                let off = (self.vals[slot] as usize) * self.d_model;
                return Some((&self.delta_u[off..off + self.d_model], &self.delta_v[off..off + self.d_model]));
            }
            if k == EMPTY_KEY { return None; }
            slot = (slot + 1) & self.mask;
        }
    }
}

/// Multi-PEER sparse `f32` delta overlay loaded live on top of read-only base weights.
#[derive(Clone, Default)]
pub struct PeerDeltaSidecar {
    #[allow(dead_code)]
    pub d_model: usize,
    pub total_rows: usize,
    pub tables: Vec<PeerDeltaTable>,
}

fn read_u32_le<R: Read>(r: &mut R) -> std::io::Result<u32> {
    let mut b = [0u8; 4]; r.read_exact(&mut b)?; Ok(u32::from_le_bytes(b))
}

fn read_f32_slice_checked<R: Read>(r: &mut R, dst: &mut [f32], raw: &mut [u8]) -> Result<(), i32> {
    r.read_exact(raw).map_err(|_| -3)?;
    for (i, ch) in raw.chunks_exact(4).enumerate() {
        let v = f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]);
        if !v.is_finite() { return Err(-6); }
        dst[i] = v;
    }
    Ok(())
}

impl PeerDeltaSidecar {
    pub fn empty() -> Self { Self { d_model: 0, total_rows: 0, tables: Vec::new() } }
    #[allow(dead_code)]
    #[inline] pub fn is_empty(&self) -> bool { self.total_rows == 0 }

    #[inline]
    pub fn table(&self, peer_idx: usize) -> Option<&PeerDeltaTable> {
        if self.total_rows == 0 { return None; }
        self.tables.get(peer_idx).filter(|t| !t.is_empty())
    }

    /// Loads `sleep_peer_delta.bin` (`GV8DLT01`), validating header, exact length, <= 20 MiB cap,
    /// expert bounds, and finiteness. Error codes: `-3` IO/format, `-4` shape/bounds, `-5` >20 MiB, `-6` NaN/Inf.
    pub fn load(
        path: &str, expected_d_model: usize, expected_n_peers: usize, max_expert_id: usize,
    ) -> Result<Self, i32> {
        let file = File::open(path).map_err(|_| -3)?;
        let file_len = file.metadata().map_err(|_| -3)?.len();
        if file_len < DELTA_HEADER_BYTES as u64 { return Err(-3); }
        if file_len > MAX_SIDECAR_BYTES { return Err(-5); }
        let mut r = BufReader::with_capacity(1 << 18, file);
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic).map_err(|_| -3)?;
        if &magic != DELTA_MAGIC { return Err(-3); }
        let version = read_u32_le(&mut r).map_err(|_| -3)?;
        let d_model = read_u32_le(&mut r).map_err(|_| -3)? as usize;
        let n_peers_hdr = read_u32_le(&mut r).map_err(|_| -3)? as usize;
        let n_rows = read_u32_le(&mut r).map_err(|_| -3)? as usize;
        let _flags = read_u32_le(&mut r).map_err(|_| -3)?;
        let _reserved = read_u32_le(&mut r).map_err(|_| -3)?;

        if version != DELTA_VERSION || d_model != expected_d_model || n_peers_hdr == 0 || n_peers_hdr > expected_n_peers {
            return Err(-4);
        }
        let rec_bytes = 16u64 + 8u64 * (d_model as u64);
        if (DELTA_HEADER_BYTES as u64) + (n_rows as u64).saturating_mul(rec_bytes) != file_len {
            return Err(-3);
        }
        if n_rows == 0 { return Ok(Self::empty()); }

        let mut counts = vec![0usize; expected_n_peers];
        let mut rec_meta: Vec<(usize, u32)> = Vec::with_capacity(n_rows);
        let mut all_du = vec![0.0f32; n_rows * d_model];
        let mut all_dv = vec![0.0f32; n_rows * d_model];
        let mut raw = vec![0u8; d_model * 4];
        let mut hdr16 = [0u8; 16];
        for i in 0..n_rows {
            r.read_exact(&mut hdr16).map_err(|_| -3)?;
            let peer_idx = u32::from_le_bytes([hdr16[0], hdr16[1], hdr16[2], hdr16[3]]) as usize;
            let expert_id = u32::from_le_bytes([hdr16[4], hdr16[5], hdr16[6], hdr16[7]]);
            let vu = f32::from_le_bytes([hdr16[8], hdr16[9], hdr16[10], hdr16[11]]);
            let vv = f32::from_le_bytes([hdr16[12], hdr16[13], hdr16[14], hdr16[15]]);
            if peer_idx >= expected_n_peers || (expert_id as usize) >= max_expert_id { return Err(-4); }
            if !vu.is_finite() || !vv.is_finite() { return Err(-6); }
            let off = i * d_model;
            read_f32_slice_checked(&mut r, &mut all_du[off..off + d_model], &mut raw)?;
            read_f32_slice_checked(&mut r, &mut all_dv[off..off + d_model], &mut raw)?;
            counts[peer_idx] += 1;
            rec_meta.push((peer_idx, expert_id));
        }

        let mut tables: Vec<PeerDeltaTable> = counts.iter().map(|&c| PeerDeltaTable::with_capacity(d_model, c)).collect();
        for (i, &(pi, eid)) in rec_meta.iter().enumerate() {
            let off = i * d_model;
            tables[pi].insert_or_replace(eid, &all_du[off..off + d_model], &all_dv[off..off + d_model]);
        }
        let total_rows: usize = tables.iter().map(|t| t.len()).sum();
        Ok(Self { d_model, total_rows, tables })
    }

    /// Writes `GV8DLT01` sparse sidecar file (for Rust tests & round-trip checks).
    #[allow(dead_code)]
    pub fn save_records(path: &str, d_model: usize, n_peers: usize, records: &[PeerDeltaRecord]) -> std::io::Result<()> {
        let mut w = BufWriter::with_capacity(1 << 18, File::create(path)?);
        w.write_all(DELTA_MAGIC)?;
        for &v in &[DELTA_VERSION, d_model as u32, n_peers as u32, records.len() as u32, 0u32, 0u32] {
            w.write_all(&v.to_le_bytes())?;
        }
        for rec in records {
            w.write_all(&rec.peer_idx.to_le_bytes())?;
            w.write_all(&rec.expert_id.to_le_bytes())?;
            w.write_all(&rec.vu_local.to_le_bytes())?;
            w.write_all(&rec.vv_local.to_le_bytes())?;
            for &x in &rec.delta_u { w.write_all(&x.to_le_bytes())?; }
            for &x in &rec.delta_v { w.write_all(&x.to_le_bytes())?; }
        }
        w.flush()
    }
}
