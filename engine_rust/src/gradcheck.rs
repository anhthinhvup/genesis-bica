//! Genesis-v7 Finite-Difference Numerical Gradient Check (`gradcheck.rs`).
//! Verifies exact chain-rule gradients across BitNet-MinGRU-MoE + SlidingWindowAttn (RoPE) + PEER + RecurrentDepth!
//! Zero external crates | Strictly <= 75 LOC.

use crate::model::{GenesisV6Model, ModelConfig};
use crate::optim::{AdamWState, GradBuf};
use crate::train_bptt::forward_backward_seq;

pub fn run_grad_check() -> bool {
    let cfg = ModelConfig {
        vocab: 32, d_model: 16, n_layers: 4, n_experts: 4, top_k: 2, d_ff: 16,
        peer_sub_k: 8, peer_heads: 2, peer_k: 4, recur_steps: 2, attn_window: 8,
    };
    let mut model = GenesisV6Model::new(cfg, 12345);
    let seq: Vec<u16> = vec![3, 7, 11, 15, 4, 9, 21, 5, 19, 2];
    let mut grads = GradBuf::new(&model);
    let mut rng = 777u64;
    let (base_loss, _) = forward_backward_seq(&model, &seq, &mut grads, 0, 0.0, false, &mut rng);

    let mut max_rel = 0.0f64;
    let n_slices = grads.flat_slices().len();
    // Tail slices: peers[0](4), peers[1](4), attn(5: norm, wq, wk, wv, wo)
    let a_norm = n_slices - 5;
    let a_wq = n_slices - 4;
    let a_wk = n_slices - 3;
    let a_wv = n_slices - 2;
    let a_wo = n_slices - 1;
    let p0_norm = n_slices - 13;
    let p0_wq = n_slices - 12;
    let checks: [(&str, usize, usize); 11] = [
        ("emb[7,3]", 0, 7 * 16 + 3),
        ("norm_f[5]", 1, 5),
        ("recur_inj[4]", 2, 4),
        ("L0.norm1[4]", 3, 4),
        ("PEER0.norm[3]", p0_norm, 3),
        ("PEER0.wq[19]", p0_wq, 19),
        ("ATTN.norm[2]", a_norm, 2),
        ("ATTN.wq[21]", a_wq, 21),
        ("ATTN.wk[11]", a_wk, 11),
        ("ATTN.wv[7]", a_wv, 7),
        ("ATTN.wo[9]", a_wo, 9),
    ];
    let mut g_dummy = GradBuf::new(&model);
    for &(name, slice_idx, elem_idx) in &checks {
        let ana = grads.flat_slices()[slice_idx][elem_idx] as f64;
        // Adaptive finite-diff step: avoids Top-K boundary flips on large grads & f32 roundoff on tiny grads
        let eps = if ana.abs() > 0.05 { 1e-4f32 } else { 5e-4f32 };
        let orig = model.flat_param_slices_mut()[slice_idx][elem_idx];
        model.flat_param_slices_mut()[slice_idx][elem_idx] = orig + eps;
        let (lp, _) = forward_backward_seq(&model, &seq, &mut g_dummy, 0, 0.0, false, &mut rng);
        model.flat_param_slices_mut()[slice_idx][elem_idx] = orig - eps;
        let (lm, _) = forward_backward_seq(&model, &seq, &mut g_dummy, 0, 0.0, false, &mut rng);
        model.flat_param_slices_mut()[slice_idx][elem_idx] = orig;
        let num = (lp - lm) / (2.0 * (eps as f64));
        let rel = (ana - num).abs() / (ana.abs().max(num.abs()).max(2e-3));
        if rel > max_rel { max_rel = rel; }
        println!("  [GRADCHECK] {:<14} ana={:+.6} num={:+.6} rel_err={:.2e}", name, ana, num, rel);
    }

    let mut opt = AdamWState::new(&mut model);
    let mut last_loss = (base_loss as f32) / ((seq.len() - 1) as f32);
    let start_loss = last_loss;
    for _ in 0..40 {
        grads.zero();
        let (l, cnt) = forward_backward_seq(&model, &seq, &mut grads, 0, 0.01, false, &mut rng);
        last_loss = (l as f32) / (cnt as f32);
        opt.step_update(&mut model, &grads, 1.0 / (cnt as f32), 0.02, 0.01, 1.0);
    }
    println!(
        "  [V7 HYBRID ATTN+PEER+RECUR CONVERGE] loss: {:.4} -> {:.4} (max_rel_err={:.2e})",
        start_loss, last_loss, max_rel
    );
    max_rel < 5e-2 && last_loss < 0.25
}
