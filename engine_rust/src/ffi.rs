//! Genesis-v6 C-ABI FFI Bridge (`ffi.rs`) for Python `ctypes` (`genesis_v6.dll`).
//! Enables zero-overhead live token-level mixing: P_BitNet x P_Hopfield x P_Infini-gram.
//! Zero external crates | Strictly <= 160 LOC.

use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::slice;

use crate::bpe::ByteBpeTokenizer;
use crate::generate::step_token_in_state;
use crate::gv8_codec::convert_gv6_to_gv8;
use crate::model::{GenesisV6Model, InferenceState};
use crate::peer_delta_sidecar::PeerDeltaSidecar;
use crate::reason_tree::expand_parallel_branches;

pub struct V6Handle {
    pub model: GenesisV6Model,
    pub bpe: ByteBpeTokenizer,
    pub state: InferenceState,
}

#[no_mangle]
pub unsafe extern "C" fn v6_load_handle(
    model_path: *const c_char,
    bpe_path: *const c_char,
) -> *mut V6Handle {
    if model_path.is_null() || bpe_path.is_null() {
        return std::ptr::null_mut();
    }
    let mp = match CStr::from_ptr(model_path).to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    let bp = match CStr::from_ptr(bpe_path).to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    let model = match GenesisV6Model::load(mp) {
        Ok(m) => m,
        Err(_) => return std::ptr::null_mut(),
    };
    let bpe = match ByteBpeTokenizer::load(bp) {
        Ok(b) => b,
        Err(_) => return std::ptr::null_mut(),
    };
    let state = InferenceState::new(&model.cfg);
    Box::into_raw(Box::new(V6Handle { model, bpe, state }))
}

#[no_mangle]
pub unsafe extern "C" fn v6_free_handle(ptr: *mut V6Handle) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

#[no_mangle]
pub unsafe extern "C" fn v6_vocab_size(ptr: *const V6Handle) -> c_int {
    if ptr.is_null() {
        return 0;
    }
    (*ptr).model.cfg.vocab as c_int
}

#[no_mangle]
pub unsafe extern "C" fn v6_reset_state(ptr: *mut V6Handle) {
    if !ptr.is_null() {
        (*ptr).state.reset();
    }
}

/// State-safe dynamic Recurrent Depth (`R = 1..16`, preserves step=0 / Prelude / Coda / KV-cache!).
#[no_mangle]
pub unsafe extern "C" fn v6_set_think_steps(ptr: *mut V6Handle, think_steps: c_int) {
    if !ptr.is_null() {
        let h = &mut *ptr;
        let r = (think_steps.max(1) as usize).clamp(1, 16);
        h.state.set_think_steps(&h.model.cfg, r);
    }
}

#[no_mangle]
pub unsafe extern "C" fn v6_get_think_steps(ptr: *const V6Handle) -> c_int {
    if ptr.is_null() { return 0; }
    (*ptr).state.think_steps as c_int
}

#[no_mangle]
pub unsafe extern "C" fn v6_feed_token(
    ptr: *mut V6Handle,
    tok_id: u16,
    logits_out: *mut f32,
    logits_len: c_int,
) -> c_int {
    if ptr.is_null() || logits_out.is_null() || logits_len <= 0 {
        return -1;
    }
    let h = &mut *ptr;
    let out = slice::from_raw_parts_mut(logits_out, logits_len as usize);
    step_token_in_state(&h.model, &mut h.state, tok_id, out);
    h.model.cfg.vocab as c_int
}

#[no_mangle]
pub unsafe extern "C" fn v6_encode(
    ptr: *const V6Handle,
    utf8_text: *const c_char,
    out_ids: *mut u16,
    max_ids: c_int,
) -> c_int {
    if ptr.is_null() || utf8_text.is_null() || out_ids.is_null() || max_ids <= 0 {
        return -1;
    }
    let text = match CStr::from_ptr(utf8_text).to_str() {
        Ok(s) => s,
        Err(_) => return -2,
    };
    let ids = (*ptr).bpe.encode(text);
    let n = ids.len().min(max_ids as usize);
    let dst = slice::from_raw_parts_mut(out_ids, n);
    dst.copy_from_slice(&ids[..n]);
    n as c_int
}

#[no_mangle]
pub unsafe extern "C" fn v6_decode(
    ptr: *const V6Handle,
    ids: *const u16,
    n_ids: c_int,
    out_utf8: *mut u8,
    max_bytes: c_int,
) -> c_int {
    if ptr.is_null() || ids.is_null() || out_utf8.is_null() || n_ids < 0 || max_bytes <= 0 {
        return -1;
    }
    let id_slice = slice::from_raw_parts(ids, n_ids as usize);
    let decoded = (*ptr).bpe.decode(id_slice);
    let bytes = decoded.as_bytes();
    let n = bytes.len().min((max_bytes as usize).saturating_sub(1));
    let dst = slice::from_raw_parts_mut(out_utf8, n + 1);
    dst[..n].copy_from_slice(&bytes[..n]);
    dst[n] = 0;
    n as c_int
}

/// Persistent in-RAM multi-core parallel Tree-of-Thought expansion (0 disk reload, O(1) fork!).
/// Writes `n_branches` UTF-8 strings separated by ASCII RS (`0x1E`) into `out_utf8`.
/// Returns total UTF-8 byte length written (or negative error code).
#[no_mangle]
pub unsafe extern "C" fn v6_generate_parallel_branches(
    ptr: *const V6Handle,
    utf8_prompt: *const c_char,
    n_branches: c_int,
    max_new: c_int,
    base_temp: f32,
    top_p: f32,
    think_steps: c_int,
    seed: u64,
    out_utf8: *mut u8,
    max_bytes: c_int,
    out_total_tokens: *mut c_int,
) -> c_int {
    if ptr.is_null() || utf8_prompt.is_null() || out_utf8.is_null() || max_bytes <= 1 {
        return -1;
    }
    let prompt = match CStr::from_ptr(utf8_prompt).to_str() {
        Ok(s) => s,
        Err(_) => return -2,
    };
    let h = &*ptr;
    let branches = expand_parallel_branches(
        &h.model,
        &h.bpe,
        prompt,
        n_branches.max(1) as usize,
        max_new.clamp(1, 4096) as usize,
        base_temp,
        top_p,
        think_steps.clamp(1, 16) as usize,
        seed,
    );
    let mut joined = String::new();
    let mut tot_toks = 0usize;
    for (i, b) in branches.iter().enumerate() {
        if i > 0 {
            joined.push('\x1e');
        }
        joined.push_str(&b.text);
        tot_toks += b.n_tokens;
    }
    if !out_total_tokens.is_null() {
        *out_total_tokens = tot_toks as c_int;
    }
    let bytes = joined.as_bytes();
    let n = bytes.len().min((max_bytes as usize).saturating_sub(1));
    let dst = slice::from_raw_parts_mut(out_utf8, n + 1);
    dst[..n].copy_from_slice(&bytes[..n]);
    dst[n] = 0;
    n as c_int
}

/// Live-loads sparse f32 PEER delta sidecar (`sleep_peer_delta.bin`, `< 20 MiB`) without reloading base model!
/// Returns number of active delta rows loaded (>= 0), or negative error code on validation/IO failure.
#[no_mangle]
pub unsafe extern "C" fn v6_load_peer_delta_sidecar(
    ptr: *mut V6Handle,
    delta_path: *const c_char,
) -> c_int {
    if ptr.is_null() || delta_path.is_null() {
        return -1;
    }
    let dp = match CStr::from_ptr(delta_path).to_str() {
        Ok(s) => s,
        Err(_) => return -2,
    };
    let h = &mut *ptr;
    if h.model.peers.is_empty() {
        return -4;
    }
    let max_exp = h.model.cfg.peer_sub_k * h.model.cfg.peer_sub_k;
    match PeerDeltaSidecar::load(dp, h.model.cfg.d_model, h.model.peers.len(), max_exp) {
        Ok(sidecar) => {
            let n = sidecar.total_rows as c_int;
            h.model.peer_delta = sidecar;
            n
        }
        Err(code) => code as c_int,
    }
}

#[no_mangle]
pub unsafe extern "C" fn v6_clear_peer_delta_sidecar(ptr: *mut V6Handle) -> c_int {
    if ptr.is_null() { return -1; }
    (*ptr).model.peer_delta = PeerDeltaSidecar::empty();
    0
}

#[no_mangle]
pub unsafe extern "C" fn v6_peer_delta_row_count(ptr: *const V6Handle) -> c_int {
    if ptr.is_null() { return -1; }
    (*ptr).model.peer_delta.total_rows as c_int
}

/// Returns routed PEER expert IDs recorded during the most recent `v6_feed_token` call.
#[no_mangle]
pub unsafe extern "C" fn v6_last_peer_ids(
    ptr: *const V6Handle,
    peer_idx: c_int,
    out_ids: *mut u32,
    max_ids: c_int,
) -> c_int {
    if ptr.is_null() || out_ids.is_null() || max_ids <= 0 || peer_idx < 0 || peer_idx >= 2 {
        return -1;
    }
    let h = &*ptr;
    let pi = peer_idx as usize;
    let n = h.state.scratch.last_peer_counts[pi].min(max_ids as usize);
    let dst = slice::from_raw_parts_mut(out_ids, n);
    dst.copy_from_slice(&h.state.scratch.last_peer_ids[pi][..n]);
    n as c_int
}

/// Converts a `GV6MOD01..03` checkpoint to packed Full-RAM `GV8MOD01` (`peer_fmt`: 1=Q8_F16, 2=Q8_F32, 3=Q14).
#[no_mangle]
pub unsafe extern "C" fn v6_convert_gv6_to_gv8(
    in_path: *const c_char,
    out_path: *const c_char,
    peer_fmt: c_int,
) -> c_int {
    if in_path.is_null() || out_path.is_null() { return -1; }
    let ip = match CStr::from_ptr(in_path).to_str() { Ok(s) => s, Err(_) => return -2 };
    let op = match CStr::from_ptr(out_path).to_str() { Ok(s) => s, Err(_) => return -2 };
    match convert_gv6_to_gv8(ip, op, peer_fmt.max(1) as u32) {
        Ok(()) => 0,
        Err(_) => -3,
    }
}
