//! Genesis-v6 Multi-Threaded CPU Training Loop & Benchmark (`train.rs`).
//! Uses `std::thread::scope` data-parallelism + Cosine LR + Low-RAM streaming (< 300 MB).
//! Zero external crates | Strictly <= 220 LOC.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::time::Instant;

use crate::bpe::ByteBpeTokenizer;
use crate::generate::{generate_text, step_token, StepScratch};
use crate::model::{GenesisV6Model, InferenceState, ModelConfig};
use crate::optim::{AdamWState, GradBuf};
use crate::train_bptt::forward_backward_seq;

pub struct TrainArgs {
    pub tokens_path: String,
    pub bpe_path: String,
    pub out_model: String,
    pub resume_model: Option<String>,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_experts: usize,
    pub top_k: usize,
    pub d_ff: usize,
    pub peer_sub_k: usize,
    pub peer_heads: usize,
    pub peer_k: usize,
    pub recur_steps: usize,
    pub attn_window: usize,
    pub sft_mask: bool,
    pub seq_len: usize,
    pub threads: usize,
    pub steps: usize,
    pub max_seconds: f64,
    pub lr: f32,
    pub sampled_k: usize,
    pub log_every: usize,
}

/// Low-RAM sliding-window reader over arbitrarily large binary `u16` token streams (`8 MB` RAM!).
struct TokenWindow {
    file: File,
    total_tokens: usize,
    win_start: usize,
    buf: Vec<u16>,
    win_cap: usize,
}

impl TokenWindow {
    fn open(path: &str, win_cap: usize) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let bytes = file.metadata()?.len() as usize;
        let total_tokens = bytes / 2;
        let cap = win_cap.min(total_tokens).max(1024);
        let mut tw = Self { file, total_tokens, win_start: 0, buf: Vec::with_capacity(cap), win_cap: cap };
        tw.load_window(0)?;
        Ok(tw)
    }

    fn load_window(&mut self, start_tok: usize) -> std::io::Result<()> {
        let st = start_tok % self.total_tokens;
        let rem = self.total_tokens - st;
        let n = self.win_cap.min(rem);
        self.file.seek(SeekFrom::Start((st as u64) * 2))?;
        let mut raw = vec![0u8; n * 2];
        self.file.read_exact(&mut raw)?;
        self.buf.clear();
        for ch in raw.chunks_exact(2) {
            self.buf.push(u16::from_le_bytes([ch[0], ch[1]]));
        }
        self.win_start = st;
        Ok(())
    }

    fn ensure_range(&mut self, global_pos: usize, len: usize) -> std::io::Result<usize> {
        if global_pos >= self.win_start && global_pos + len <= self.win_start + self.buf.len() {
            return Ok(global_pos - self.win_start);
        }
        let target = if global_pos + len <= self.total_tokens { global_pos } else { 0 };
        self.load_window(target)?;
        Ok(0)
    }
}

/// Benchmarks real multi-threaded CPU training throughput (tokens/sec) & exact step latency.
pub fn bench_throughput(cfg: ModelConfig, seq_len: usize, threads: usize, sampled_k: usize, steps: usize) {
    let mut model = GenesisV6Model::new(cfg, 42);
    let mut opt = AdamWState::new(&mut model);
    let mut thread_grads: Vec<GradBuf> = (0..threads).map(|_| GradBuf::new(&model)).collect();
    let fake_toks: Vec<u16> = (0..(seq_len + 1) * threads).map(|i| ((i * 37 + 19) % cfg.vocab) as u16).collect();

    println!(
        "[V7-BENCH] params={:.2}M | d={} L={} E={} k={} V={} | PEER={}x{} (k={}) ATTN_W={} R={} | seq={} th={} sk={}",
        (cfg.num_params() as f64) / 1e6, cfg.d_model, cfg.n_layers, cfg.n_experts, cfg.top_k, cfg.vocab,
        model.peers.len(), cfg.peer_sub_k * cfg.peer_sub_k, cfg.peer_heads * cfg.peer_k, cfg.attn_window,
        cfg.recur_steps, seq_len, threads, sampled_k
    );
    let t0 = Instant::now();
    let mut total_toks = 0usize;
    let mut last_loss = 0.0f32;
    for st in 0..steps {
        let model_ref = &model;
        let losses: Vec<(f32, usize)> = std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(threads);
            for (ti, gb) in thread_grads.iter_mut().enumerate() {
                let slice = &fake_toks[ti * (seq_len + 1)..(ti + 1) * (seq_len + 1)];
                let seed = (st as u64 + 1) * 10007 + (ti as u64 + 1) * 97;
                handles.push(s.spawn(move || {
                    gb.zero();
                    let mut rng = seed;
                    let (l, c) = forward_backward_seq(model_ref, slice, gb, sampled_k, 0.01, false, &mut rng);
                    (l as f32, c)
                }));
            }
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let (first, rest) = thread_grads.split_first_mut().unwrap();
        for other in rest { first.add_from(other); }
        let batch_toks: usize = losses.iter().map(|x| x.1).sum::<usize>().max(1);
        last_loss = losses.iter().map(|x| x.0).sum::<f32>() / (batch_toks as f32);
        opt.step_update(&mut model, first, 1.0 / (batch_toks as f32), 3e-3, 0.01, 1.0);
        total_toks += threads * seq_len;
    }
    let dt = t0.elapsed().as_secs_f64().max(1e-6);
    println!(
        "[V6-BENCH] {} steps ({} tokens) in {:.2}s => {:.1} tok/s | last_loss={:.4}",
        steps, total_toks, dt, (total_toks as f64) / dt, last_loss
    );
}

fn save_atomic(model: &GenesisV6Model, out_path: &str) -> std::io::Result<()> {
    let tmp = format!("{}.tmp", out_path);
    model.save(&tmp)?;
    let _ = std::fs::remove_file(out_path);
    std::fs::rename(&tmp, out_path)
}

fn eval_heldout_val_loss(model: &GenesisV6Model, val_toks: &[u16], seq_len: usize) -> f32 {
    let stride = seq_len + 1;
    let n_seq = val_toks.len() / stride;
    if n_seq == 0 { return 99.0; }
    let mut st = InferenceState::new(&model.cfg);
    let mut sc = StepScratch::new(model);
    let mut logits = vec![0.0f32; model.cfg.vocab];
    let mut loss_sum = 0.0f64;
    let mut count = 0usize;
    for s in 0..n_seq {
        st.reset();
        let slice = &val_toks[s * stride..(s + 1) * stride];
        for t in 0..seq_len {
            step_token(model, &mut st, &mut sc, slice[t], &mut logits);
            let y = (slice[t + 1] as usize).min(model.cfg.vocab - 1);
            let mut mx = f32::NEG_INFINITY;
            for &lv in &logits { if lv > mx { mx = lv; } }
            let mut se = 0.0f64;
            for &lv in &logits { se += ((lv - mx) as f64).exp(); }
            let lp = ((logits[y] - mx) as f64) - se.max(1e-18).ln();
            loss_sum -= lp;
            count += 1;
        }
    }
    (loss_sum / (count.max(1) as f64)) as f32
}

/// Runs multi-threaded CPU training with held-out validation, PaLM Spike-Skip Guard & random window jumping.
pub fn run_train(args: &TrainArgs) -> std::io::Result<()> {
    let bpe = ByteBpeTokenizer::load(&args.bpe_path)?;
    let mut tw = TokenWindow::open(&args.tokens_path, 4_194_304)?;
    let val_seqs = 12usize;
    let val_len = val_seqs * (args.seq_len + 1);
    if tw.total_tokens < val_len + args.seq_len * 8 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "token stream too short"));
    }
    let val_toks: Vec<u16> = tw.buf[..val_len].to_vec();

    let mut model = if let Some(ref rp) = args.resume_model {
        println!("[V7-TRAIN] Resuming model from {} (recur_steps={})", rp, args.recur_steps.max(1));
        let mut m = GenesisV6Model::load(rp)?;
        // Support Curriculum Depth transition on --resume (e.g. Phase 1 R=1 -> Phase 2 R=2!)
        m.cfg.recur_steps = args.recur_steps.max(1);
        m
    } else {
        let cfg = ModelConfig {
            vocab: bpe.vocab_size, d_model: args.d_model, n_layers: args.n_layers,
            n_experts: args.n_experts, top_k: args.top_k, d_ff: args.d_ff,
            peer_sub_k: args.peer_sub_k, peer_heads: args.peer_heads,
            peer_k: args.peer_k, recur_steps: args.recur_steps.max(1),
            attn_window: args.attn_window,
        };
        GenesisV6Model::new(cfg, 202607)
    };
    let cfg = model.cfg;
    let threads = args.threads.max(1);
    let mut opt = AdamWState::new(&mut model);
    let mut thread_grads: Vec<GradBuf> = (0..threads).map(|_| GradBuf::new(&model)).collect();
    let train_min = val_len;
    let train_span = (tw.total_tokens - train_min - args.seq_len - 2).max(1);

    println!(
        "[V7-TRAIN] params={:.2}M | corpus={:.2}M tok (val={}) | V={} D={} L={} E={}(k={}) | PEER={}x{} ATTN_W={} R={} sft={} | th={} seq={} sk={}",
        (cfg.num_params() as f64) / 1e6, (tw.total_tokens as f64) / 1e6, val_len, cfg.vocab, cfg.d_model,
        cfg.n_layers, cfg.n_experts, cfg.top_k, model.peers.len(), cfg.peer_sub_k * cfg.peer_sub_k,
        cfg.attn_window, cfg.recur_steps, args.sft_mask, threads, args.seq_len, args.sampled_k
    );
    let _ = std::io::stdout().flush();

    let t_start = Instant::now();
    let mut total_toks = 0u64;
    let mut ema_loss = 0.0f32;
    let mut best_val = f32::INFINITY;
    let mut skips = 0u64;
    let mut cursor = train_min;
    let mut jump_rng = 0xD1B5_4A32_D192_ED03u64;
    let warmup = (args.steps / 30).clamp(50, 400);
    let best_path = if args.out_model.ends_with(".bin") {
        args.out_model.replace(".bin", "_best.bin")
    } else {
        format!("{}.best", args.out_model)
    };

    for step in 1..=args.steps {
        let elapsed = t_start.elapsed().as_secs_f64();
        if args.max_seconds > 0.0 && elapsed >= args.max_seconds {
            println!("[V6-TRAIN] Reached max_seconds={:.1}s at step {}", args.max_seconds, step - 1);
            let _ = std::io::stdout().flush();
            break;
        }
        let lr = if step <= warmup {
            args.lr * (step as f32) / (warmup as f32)
        } else {
            let prog = ((step - warmup) as f32) / ((args.steps.saturating_sub(warmup)).max(1) as f32);
            let cos = 0.5 * (1.0 + (std::f32::consts::PI * prog).cos());
            args.lr * (0.08 + 0.92 * cos)
        };

        // Periodic random jump across the entire shuffled token file for true global i.i.d. mixing!
        if step % 200 == 1 {
            jump_rng ^= jump_rng << 13; jump_rng ^= jump_rng >> 7; jump_rng ^= jump_rng << 17;
            cursor = train_min + ((jump_rng as usize) % train_span);
        }
        let span = threads * (args.seq_len + 137) + args.seq_len + 2;
        if cursor + span >= tw.total_tokens {
            cursor = train_min + ((cursor + 997) % train_span);
        }
        let local_base = tw.ensure_range(cursor, span.min(tw.total_tokens - train_min - 1))?;
        let local_cap = tw.buf.len().saturating_sub(args.seq_len + 1).max(1);
        let mut starts = Vec::with_capacity(threads);
        for ti in 0..threads {
            let off = (local_base + ti * (args.seq_len + 137)) % local_cap;
            starts.push(off);
        }
        cursor = train_min + ((cursor - train_min + threads * (args.seq_len + 137)) % train_span);

        let step_sampled_k = if args.steps >= 500 && step * 50 >= args.steps * 49 { 0 } else { args.sampled_k };
        let model_ref = &model;
        let toks_ref = &tw.buf;
        let seq_len = args.seq_len;

        let sft_flag = args.sft_mask;
        let losses: Vec<(f32, usize)> = std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(threads);
            for (ti, gb) in thread_grads.iter_mut().enumerate() {
                let st_pos = starts[ti];
                let slice = &toks_ref[st_pos..st_pos + seq_len + 1];
                let seed = (step as u64) * 6364136223846793005u64 ^ ((ti as u64) + 1);
                handles.push(s.spawn(move || {
                    gb.zero();
                    let mut rng = seed.max(1);
                    let (l, c) = forward_backward_seq(model_ref, slice, gb, step_sampled_k, 0.01, sft_flag, &mut rng);
                    (l as f32, c)
                }));
            }
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        let (first, rest) = thread_grads.split_first_mut().unwrap();
        for other in rest { first.add_from(other); }
        let active_toks: usize = losses.iter().map(|x| x.1).sum::<usize>().max(1);
        let batch_toks = threads * seq_len;
        let mean_loss = losses.iter().map(|x| x.0).sum::<f32>() / (active_toks as f32);
        let scale = 1.0 / (active_toks as f32);
        let pre_gnorm = first.grad_norm(scale);

        // PaLM / Llama-3 Spike-Skip Guard: never let an outlier batch corrupt BitNet/PEER weights!
        let is_spike = !mean_loss.is_finite()
            || !pre_gnorm.is_finite()
            || (step > 100 && (pre_gnorm > 15.0 || mean_loss > ema_loss + 3.0));
        let gnorm = if is_spike {
            skips += 1;
            pre_gnorm
        } else {
            ema_loss = if step == 1 { mean_loss } else { 0.95 * ema_loss + 0.05 * mean_loss };
            opt.step_update(&mut model, first, scale, lr, 0.01, 1.0)
        };
        total_toks += batch_toks as u64;

        if step == 1 || step % args.log_every == 0 || step == args.steps {
            let dt = t_start.elapsed().as_secs_f64().max(1e-6);
            println!(
                "step {:>6}/{} | toks={:.2}M | loss={:.4} (ema={:.4}, ppl={:.1}) | gnorm={:.2} | skp={} | lr={:.5} | {:.0} tok/s | {:.1}m",
                step, args.steps, (total_toks as f64) / 1e6, mean_loss, ema_loss, ema_loss.min(12.0).exp(),
                gnorm, skips, lr, (total_toks as f64) / dt, dt / 60.0
            );
            let _ = std::io::stdout().flush();
        }
        let ckpt_every = (args.log_every * 5).max(125);
        if step % ckpt_every == 0 {
            let _ = save_atomic(&model, &args.out_model);
            let v_loss = eval_heldout_val_loss(&model, &val_toks, args.seq_len);
            if v_loss.is_finite() && v_loss + 0.005 < best_val {
                best_val = v_loss;
                let _ = save_atomic(&model, &best_path);
            }
            let s1 = generate_text(&model, &bpe, "<USER> chào <BOT>", 40, 0.65, 0.9, 42 + step as u64);
            let s2 = generate_text(&model, &bpe, "<USER> bạn khỏe không <BOT>", 40, 0.65, 0.9, 99 + step as u64);
            let s3 = generate_text(&model, &bpe, "<USER> hôm nay mình mệt quá <BOT>", 40, 0.65, 0.9, 177 + step as u64);
            println!(
                "  [CKPT step={} val_loss={:.4} (val_ppl={:.1}) best_val={:.4}]",
                step, v_loss, v_loss.min(12.0).exp(), best_val.min(99.9)
            );
            println!("    S1 chào -> {}", s1.replace('\n', " "));
            println!("    S2 bạn khỏe không -> {}", s2.replace('\n', " "));
            println!("    S3 hôm nay mình mệt quá -> {}", s3.replace('\n', " "));
            let _ = std::io::stdout().flush();
        }
    }

    save_atomic(&model, &args.out_model)?;
    println!(
        "[V6.5-TRAIN] Saved final model to {} ({:.1}m, {:.2}M tokens, best_val={:.4}, skips={})",
        args.out_model, t_start.elapsed().as_secs_f64() / 60.0, (total_toks as f64) / 1e6, best_val.min(99.9), skips
    );
    let _ = std::io::stdout().flush();
    Ok(())
}
