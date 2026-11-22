//! Genesis-v6.5 Batched-GEMM Exact BPTT + PEER Product-Key + Recurrent Depth (`train_bptt.rs`).
//! Strictly <= 375 LOC | Zero external crates | 100% Exact Chain Rule.

use crate::attn_rope::AttnBatchCache;
use crate::bpe::{TOK_BOT, TOK_EOS, TOK_USER};
use crate::gemm_bitnet::{gemm_bitnet_bwd, gemm_bitnet_fwd};
use crate::gemm_f32::lm_head_fwd_bwd;
use crate::generate::{relu2, select_topk_experts, sigmoid};
use crate::model::{GenesisV6Model, SparseBlock};
use crate::optim::{GradBuf, LayerGrads};
use crate::peer_pk::PeerBatchCache;
use crate::simd_avx2::{axpy_f32, dot_f32, rmsnorm_bwd, rmsnorm_fwd};

struct BlockCache {
    x0: Vec<f32>, xn: Vec<f32>, inv1: Vec<f32>, xc: Vec<f32>, z: Vec<f32>, hc: Vec<f32>,
    h: Vec<f32>, x1: Vec<f32>, u: Vec<f32>, inv2: Vec<f32>, top_idx: Vec<usize>, top_w: Vec<f32>,
    up_pre: Vec<f32>, up_act: Vec<f32>, dn: Vec<f32>, buckets: Vec<Vec<(usize, usize, f32)>>,
}

struct ExecCache {
    layer_idx: usize,
    peer_opt: Option<usize>,
    has_attn: bool,
    is_inj: bool,
    blk: BlockCache,
    peer_cache: Option<PeerBatchCache>,
    attn_cache: Option<AttnBatchCache>,
}

fn block_fwd(layer: &SparseBlock, x_cur: &mut [f32], t_len: usize, d: usize, df: usize, ne: usize, tk: usize) -> BlockCache {
    let mut c = BlockCache {
        x0: x_cur.to_vec(), xn: vec![0.0; t_len * d], inv1: vec![0.0; t_len], xc: vec![0.0; t_len * d],
        z: vec![0.0; t_len * d], hc: vec![0.0; t_len * d], h: vec![0.0; t_len * d], x1: vec![0.0; t_len * d],
        u: vec![0.0; t_len * d], inv2: vec![0.0; t_len], top_idx: vec![0; t_len * tk], top_w: vec![0.0; t_len * tk],
        up_pre: vec![0.0; t_len * tk * df], up_act: vec![0.0; t_len * tk * df], dn: vec![0.0; t_len * tk * d],
        buckets: (0..ne).map(|_| Vec::new()).collect(),
    };
    let mut z_pre = vec![0.0f32; t_len * d];
    let mut h_pre = vec![0.0f32; t_len * d];
    let mut yo = vec![0.0f32; t_len * d];
    let mut r_logits = vec![0.0f32; ne];
    for t in 0..t_len {
        let td = t * d;
        c.inv1[t] = rmsnorm_fwd(&c.x0[td..td + d], &layer.norm1, &mut c.xn[td..td + d]);
        for i in 0..d {
            let p2 = if t >= 2 { c.xn[(t - 2) * d + i] } else { 0.0 };
            let p1 = if t >= 1 { c.xn[(t - 1) * d + i] } else { 0.0 };
            c.xc[td + i] = layer.conv_w[i] * p2 + layer.conv_w[d + i] * p1 + layer.conv_w[2 * d + i] * c.xn[td + i];
        }
    }
    gemm_bitnet_fwd(&layer.wz.q, layer.wz.gamma, d, d, &c.xc, &mut z_pre, t_len);
    gemm_bitnet_fwd(&layer.wh.q, layer.wh.gamma, d, d, &c.xc, &mut h_pre, t_len);
    let mut h_prev = vec![0.0f32; d];
    for t in 0..t_len {
        let td = t * d;
        for i in 0..d {
            let zi = sigmoid(z_pre[td + i]);
            let hi = h_pre[td + i].tanh();
            let hn = (1.0 - zi) * h_prev[i] + zi * hi;
            c.z[td + i] = zi; c.hc[td + i] = hi; c.h[td + i] = hn; h_prev[i] = hn;
        }
    }
    gemm_bitnet_fwd(&layer.wo.q, layer.wo.gamma, d, d, &c.h, &mut yo, t_len);
    for t in 0..t_len {
        let td = t * d;
        for i in 0..d { c.x1[td + i] = c.x0[td + i] + yo[td + i]; }
        c.inv2[t] = rmsnorm_fwd(&c.x1[td..td + d], &layer.norm2, &mut c.u[td..td + d]);
        for e in 0..ne { r_logits[e] = dot_f32(&layer.router[e * d..(e + 1) * d], &c.u[td..td + d]); }
        let tk_off = t * tk;
        select_topk_experts(&r_logits, tk, &mut c.top_idx[tk_off..tk_off + tk], &mut c.top_w[tk_off..tk_off + tk]);
        for j in 0..tk { c.buckets[c.top_idx[tk_off + j]].push((t, j, c.top_w[tk_off + j])); }
    }
    x_cur.copy_from_slice(&c.x1);
    let mut u_pack = Vec::new();
    let mut up_pre_p = Vec::new();
    let mut up_act_p = Vec::new();
    let mut dn_p = Vec::new();
    for e in 0..ne {
        let bkt = &c.buckets[e];
        let m = bkt.len();
        if m == 0 { continue; }
        u_pack.clear();
        for &(t, _, _) in bkt { u_pack.extend_from_slice(&c.u[t * d..(t + 1) * d]); }
        up_pre_p.resize(m * df, 0.0); up_act_p.resize(m * df, 0.0); dn_p.resize(m * d, 0.0);
        let eu = &layer.exp_up[e]; let ed = &layer.exp_dn[e];
        gemm_bitnet_fwd(&eu.q, eu.gamma, df, d, &u_pack, &mut up_pre_p, m);
        for idx in 0..m * df { up_act_p[idx] = relu2(up_pre_p[idx]); }
        gemm_bitnet_fwd(&ed.q, ed.gamma, d, df, &up_act_p, &mut dn_p, m);
        for (bi, &(t, j, p)) in bkt.iter().enumerate() {
            let up_off = (t * tk + j) * df; let dn_off = (t * tk + j) * d;
            c.up_pre[up_off..up_off + df].copy_from_slice(&up_pre_p[bi * df..(bi + 1) * df]);
            c.up_act[up_off..up_off + df].copy_from_slice(&up_act_p[bi * df..(bi + 1) * df]);
            c.dn[dn_off..dn_off + d].copy_from_slice(&dn_p[bi * d..(bi + 1) * d]);
            let td = t * d;
            for i in 0..d { x_cur[td + i] += p * dn_p[bi * d + i]; }
        }
    }
    c
}

fn block_bwd(
    layer: &SparseBlock, c: &BlockCache, dx_cur: &[f32], g: &mut LayerGrads,
    t_len: usize, d: usize, df: usize, ne: usize, tk: usize, aux_lb: f32,
) -> Vec<f32> {
    let mut dx1 = dx_cur.to_vec();
    let mut du_all = vec![0.0f32; t_len * d];
    let total_routes = (t_len * tk) as f32;
    let mut u_pack = Vec::new(); let mut up_act_p = Vec::new(); let mut d_dn_p = Vec::new();
    let mut d_up_act_p = Vec::new(); let mut d_up_pre_p = Vec::new(); let mut du_p = Vec::new();
    for e in 0..ne {
        let bkt = &c.buckets[e];
        let m = bkt.len();
        if m == 0 { continue; }
        u_pack.clear(); up_act_p.clear(); d_dn_p.clear();
        for &(t, j, p) in bkt {
            u_pack.extend_from_slice(&c.u[t * d..(t + 1) * d]);
            let up_off = (t * tk + j) * df;
            up_act_p.extend_from_slice(&c.up_act[up_off..up_off + df]);
            let td = t * d;
            for i in 0..d { d_dn_p.push(p * dx_cur[td + i]); }
        }
        d_up_act_p.clear(); d_up_act_p.resize(m * df, 0.0);
        let ed = &layer.exp_dn[e];
        gemm_bitnet_bwd(&ed.q, ed.gamma, d, df, &up_act_p, &d_dn_p, &mut d_up_act_p, &mut g.exp_dn[e], m);
        d_up_pre_p.clear(); d_up_pre_p.resize(m * df, 0.0);
        for (bi, &(t, j, _)) in bkt.iter().enumerate() {
            let up_off = (t * tk + j) * df;
            for k in 0..df {
                let pre = c.up_pre[up_off + k];
                if pre > 0.0 && pre < 6.0 { d_up_pre_p[bi * df + k] = d_up_act_p[bi * df + k] * 2.0 * pre; }
            }
        }
        du_p.clear(); du_p.resize(m * d, 0.0);
        let eu = &layer.exp_up[e];
        gemm_bitnet_bwd(&eu.q, eu.gamma, df, d, &u_pack, &d_up_pre_p, &mut du_p, &mut g.exp_up[e], m);
        for (bi, &(t, _, _)) in bkt.iter().enumerate() {
            let td = t * d;
            for i in 0..d { du_all[td + i] += du_p[bi * d + i]; }
        }
    }
    for t in 0..t_len {
        let td = t * d; let tk_off = t * tk;
        let mut dp = [0.0f32; 8]; let mut sum_p_dp = 0.0f32;
        for j in 0..tk {
            dp[j] = dot_f32(&dx_cur[td..td + d], &c.dn[(tk_off + j) * d..(tk_off + j + 1) * d]);
            sum_p_dp += c.top_w[tk_off + j] * dp[j];
        }
        for j in 0..tk {
            let e = c.top_idx[tk_off + j]; let p = c.top_w[tk_off + j];
            let mut dr = p * (dp[j] - sum_p_dp);
            if aux_lb > 0.0 { dr += aux_lb * ((c.buckets[e].len() as f32) / total_routes - 1.0 / (ne as f32)); }
            axpy_f32(dr, &c.u[td..td + d], &mut g.router[e * d..(e + 1) * d]);
            axpy_f32(dr, &layer.router[e * d..(e + 1) * d], &mut du_all[td..td + d]);
        }
        rmsnorm_bwd(&c.x1[td..td + d], &layer.norm2, c.inv2[t], &du_all[td..td + d], &mut dx1[td..td + d], &mut g.norm2);
    }
    let mut dx0 = dx1.clone();
    let mut dh_wo = vec![0.0f32; t_len * d];
    gemm_bitnet_bwd(&layer.wo.q, layer.wo.gamma, d, d, &c.h, &dx1, &mut dh_wo, &mut g.wo, t_len);
    let mut dh_next = vec![0.0f32; d];
    let mut dz_pre = vec![0.0f32; t_len * d];
    let mut dh_pre = vec![0.0f32; t_len * d];
    let zeros_d = vec![0.0f32; d];
    for t in (0..t_len).rev() {
        let td = t * d;
        let h_prev = if t > 0 { &c.h[(t - 1) * d..t * d] } else { &zeros_d[..] };
        for i in 0..d {
            let dhi = dh_wo[td + i] + dh_next[i];
            let zi = c.z[td + i]; let hi = c.hc[td + i];
            dh_next[i] = (1.0 - zi) * dhi;
            dz_pre[td + i] = dhi * (hi - h_prev[i]) * zi * (1.0 - zi);
            dh_pre[td + i] = dhi * zi * (1.0 - hi * hi);
        }
    }
    let mut dxc = vec![0.0f32; t_len * d];
    gemm_bitnet_bwd(&layer.wz.q, layer.wz.gamma, d, d, &c.xc, &dz_pre, &mut dxc, &mut g.wz, t_len);
    gemm_bitnet_bwd(&layer.wh.q, layer.wh.gamma, d, d, &c.xc, &dh_pre, &mut dxc, &mut g.wh, t_len);
    let mut dxn = vec![0.0f32; t_len * d];
    for t in 0..t_len {
        let td = t * d;
        for i in 0..d {
            let di = dxc[td + i];
            dxn[td + i] += layer.conv_w[2 * d + i] * di; g.conv_w[2 * d + i] += c.xn[td + i] * di;
            if t >= 1 { dxn[(t - 1) * d + i] += layer.conv_w[d + i] * di; g.conv_w[d + i] += c.xn[(t - 1) * d + i] * di; }
            if t >= 2 { dxn[(t - 2) * d + i] += layer.conv_w[i] * di; g.conv_w[i] += c.xn[(t - 2) * d + i] * di; }
        }
    }
    for t in 0..t_len {
        let td = t * d;
        rmsnorm_bwd(&c.x0[td..td + d], &layer.norm1, c.inv1[t], &dxn[td..td + d], &mut dx0[td..td + d], &mut g.norm1);
    }
    dx0
}

/// Exact forward + backward pass supporting Sliding-Window Causal Attention (RoPE), PEER,
/// Recurrent Depth (`R >= 1`), and optional SFT `<BOT>` Loss-Masking (`sft_mask = true`).
/// Returns `(total_loss, active_token_count)`.
pub fn forward_backward_seq(
    model: &GenesisV6Model,
    tokens: &[u16],
    grads: &mut GradBuf,
    sampled_k: usize,
    aux_lb: f32,
    sft_mask: bool,
    rng: &mut u64,
) -> (f64, usize) {
    if tokens.len() < 2 { return (0.0, 0); }
    let t_len = tokens.len() - 1;
    let d = model.cfg.d_model; let df = model.cfg.d_ff;
    let ne = model.cfg.n_experts; let tk = model.cfg.top_k; let v = model.cfg.vocab;

    // Build SFT loss mask (only supervise `<BOT> ... <EOS>` response tokens when `sft_mask == true`)
    let mut loss_mask = vec![true; t_len];
    if sft_mask {
        let mut in_bot = false;
        let mut any_bot = false;
        for t in 0..t_len {
            let tok = tokens[t];
            if tok == TOK_BOT { in_bot = true; }
            else if tok == TOK_EOS || tok == TOK_USER { in_bot = false; }
            loss_mask[t] = in_bot;
            if in_bot { any_bot = true; }
        }
        if !any_bot { loss_mask.fill(true); }
    }

    let mut x_cur = vec![0.0f32; t_len * d];
    for t in 0..t_len {
        let tid = (tokens[t] as usize).min(v - 1);
        x_cur[t * d..(t + 1) * d].copy_from_slice(&model.emb[tid * d..(tid + 1) * d]);
    }
    let plan = model.exec_plan(model.cfg.recur_steps);
    let mut exec_caches: Vec<ExecCache> = Vec::with_capacity(plan.len());
    let mut s0 = vec![0.0f32; t_len * d];

    for (slot_idx, &(li, peer_opt, attn_opt, is_inj)) in plan.iter().enumerate() {
        if is_inj {
            for t in 0..t_len {
                let td = t * d;
                for i in 0..d { x_cur[td + i] += model.recur_inj_alpha[i] * s0[td + i]; }
            }
        }
        let blk = block_fwd(&model.layers[li], &mut x_cur, t_len, d, df, ne, tk);
        let peer_cache = peer_opt.map(|pi| model.peers[pi].forward_batch_residual(&mut x_cur, t_len));
        let has_attn = attn_opt.is_some() && model.attn.is_some();
        let attn_cache = if has_attn {
            Some(model.attn.as_ref().unwrap().forward_batch_residual(&mut x_cur, t_len))
        } else { None };
        if slot_idx == 0 { s0.copy_from_slice(&x_cur); }
        exec_caches.push(ExecCache { layer_idx: li, peer_opt, has_attn, is_inj, blk, peer_cache, attn_cache });
    }

    // Final RMSNorm + Register-Tiled AVX2 Batched LM Head (Gemma-2 SoftCap + Exact/Sampled Softmax + SFT Mask)
    let mut dx_cur = vec![0.0f32; t_len * d];
    let (total_loss, active_cnt) = lm_head_fwd_bwd(
        &model.emb, &model.norm_f, &x_cur, tokens, &loss_mask, v, d, t_len,
        sampled_k, rng, &mut dx_cur, &mut grads.norm_f, &mut grads.emb,
    );

    // Reverse pass across exec_caches (exact Attention + PEER + MinGRU + Recurrent Depth backward!)
    let mut ds0 = vec![0.0f32; t_len * d];
    let mut dx_tmp = vec![0.0f32; t_len * d];
    for (slot_idx, ec) in exec_caches.iter().enumerate().rev() {
        if slot_idx == 0 {
            for idx in 0..t_len * d { dx_cur[idx] += ds0[idx]; }
        }
        if ec.has_attn {
            if let (Some(ref a), Some(ref ac), Some(ref mut ag)) = (&model.attn, &ec.attn_cache, &mut grads.attn) {
                a.backward_batch_residual(ac, &dx_cur, &mut dx_tmp, ag, t_len);
                dx_cur.copy_from_slice(&dx_tmp);
            }
        }
        if let (Some(pi), Some(ref pc)) = (ec.peer_opt, &ec.peer_cache) {
            model.peers[pi].backward_batch_residual(pc, &dx_cur, &mut dx_tmp, &mut grads.peers[pi], t_len);
            dx_cur.copy_from_slice(&dx_tmp);
        }
        dx_cur = block_bwd(&model.layers[ec.layer_idx], &ec.blk, &dx_cur, &mut grads.layers[ec.layer_idx], t_len, d, df, ne, tk, aux_lb);
        if ec.is_inj {
            for t in 0..t_len {
                let td = t * d;
                for i in 0..d {
                    let g_i = dx_cur[td + i];
                    grads.recur_inj_alpha[i] += g_i * s0[td + i];
                    ds0[td + i] += model.recur_inj_alpha[i] * g_i;
                }
            }
        }
    }
    for t in 0..t_len {
        let tid = (tokens[t] as usize).min(v - 1);
        for i in 0..d { grads.emb[tid * d + i] += dx_cur[t * d + i]; }
    }
    (total_loss, active_cnt.max(1))
}
