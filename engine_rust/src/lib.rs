//! Genesis-v6 C-ABI Dynamic Library Root (`lib.rs` -> `genesis_v6.dll`).

pub mod attn_rope;
pub mod bpe;
pub mod ffi;
pub mod gemm_bitnet;
pub mod gemm_f32;
pub mod generate;
pub mod gradcheck;
pub mod gv8_codec;
pub mod gv8_kernels;
pub mod model;
pub mod optim;
pub mod peer_delta_sidecar;
pub mod peer_pk;
pub mod reason_tree;
pub mod simd_avx2;
pub mod train;
pub mod train_bptt;
