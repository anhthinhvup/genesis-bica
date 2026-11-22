//! Genesis-v7 Pure-Rust AVX2 CLI (`main.rs`): `gradcheck`, `bench`, `train-bpe`, `encode`, `train`, `chat`.
//! Zero external crates | Zero C:\ writes | Strictly <= 175 LOC.

mod attn_rope;
mod bpe;
mod ffi;
mod gemm_bitnet;
mod gemm_f32;
mod generate;
mod gradcheck;
mod gv8_codec;
mod gv8_kernels;
mod model;
mod optim;
mod peer_delta_sidecar;
mod peer_pk;
mod reason_tree;
mod simd_avx2;
mod train;
mod train_bptt;

use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};

use bpe::ByteBpeTokenizer;
use generate::generate_text_with_think;
use gradcheck::run_grad_check;
use model::{GenesisV6Model, ModelConfig};
use train::{bench_throughput, run_train, TrainArgs};

fn get_arg(args: &[String], flag: &str, default: &str) -> String {
    for i in 0..args.len() {
        if args[i] == flag && i + 1 < args.len() { return args[i + 1].clone(); }
    }
    default.to_string()
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: genesis_v6 <gradcheck|bench|train-bpe|encode|train|chat> [flags]");
        std::process::exit(1);
    }
    match args[1].as_str() {
        "gradcheck" => {
            println!("[V7] Running exact finite-difference gradient check (BitNet+AttnRoPE+PEER+RecurrentDepth)...");
            let ok = run_grad_check();
            println!("[V7] GRADCHECK RESULT: {}", if ok { "PASS" } else { "FAIL" });
            if !ok { std::process::exit(2); }
        }
        "bench" => {
            let cfg = ModelConfig {
                vocab: get_arg(&args, "--vocab", "4096").parse().unwrap_or(4096),
                d_model: get_arg(&args, "--d-model", "192").parse().unwrap_or(192),
                n_layers: get_arg(&args, "--layers", "4").parse().unwrap_or(4),
                n_experts: get_arg(&args, "--experts", "8").parse().unwrap_or(8),
                top_k: get_arg(&args, "--top-k", "2").parse().unwrap_or(2),
                d_ff: get_arg(&args, "--d-ff", "192").parse().unwrap_or(192),
                peer_sub_k: get_arg(&args, "--peer-sub-k", "256").parse().unwrap_or(256),
                peer_heads: get_arg(&args, "--peer-heads", "4").parse().unwrap_or(4),
                peer_k: get_arg(&args, "--peer-k", "4").parse().unwrap_or(4),
                recur_steps: get_arg(&args, "--recur-steps", "1").parse().unwrap_or(1),
                attn_window: get_arg(&args, "--attn-window", "128").parse().unwrap_or(128),
            };
            let seq = get_arg(&args, "--seq", "128").parse().unwrap_or(128);
            let th = get_arg(&args, "--threads", "4").parse().unwrap_or(4);
            let sk = get_arg(&args, "--sampled-k", "128").parse().unwrap_or(128);
            let steps = get_arg(&args, "--steps", "20").parse().unwrap_or(20);
            bench_throughput(cfg, seq, th, sk, steps);
        }
        "train-bpe" => {
            let inp = get_arg(&args, "--input", "");
            let out = get_arg(&args, "--output", "");
            let vocab = get_arg(&args, "--vocab", "4096").parse().unwrap_or(4096);
            let min_freq = get_arg(&args, "--min-freq", "4").parse().unwrap_or(4);
            let text = fs::read_to_string(&inp).expect("read input corpus");
            println!("[V7-BPE] Training Byte-BPE V={} on {} bytes...", vocab, text.len());
            let bpe = ByteBpeTokenizer::train(&text, vocab, min_freq);
            bpe.save(&out).expect("save bpe");
            println!("[V7-BPE] Saved vocab_size={} to {}", bpe.vocab_size, out);
        }
        "encode" => {
            let inp = get_arg(&args, "--input", "");
            let bpe_p = get_arg(&args, "--bpe", "");
            let out = get_arg(&args, "--output", "");
            let bpe = ByteBpeTokenizer::load(&bpe_p).expect("load bpe");
            let rdr = BufReader::with_capacity(1 << 20, File::open(&inp).expect("open corpus"));
            let mut w = BufWriter::with_capacity(1 << 20, File::create(&out).expect("create tokens file"));
            let mut total_bytes: u64 = 0;
            let mut total_toks: u64 = 0;
            for line_res in rdr.lines() {
                let mut line = line_res.expect("read line");
                line.push('\n');
                total_bytes += line.len() as u64;
                let ids = bpe.encode(&line);
                total_toks += ids.len() as u64;
                for &id in &ids { w.write_all(&id.to_le_bytes()).expect("write token"); }
            }
            w.flush().expect("flush tokens");
            println!(
                "[V7-ENC] Encoded {} bytes -> {} tokens ({:.2} bytes/tok) to {}",
                total_bytes, total_toks, (total_bytes as f64) / (total_toks.max(1) as f64), out
            );
        }
        "train" => {
            let resume = if has_flag(&args, "--resume") { Some(get_arg(&args, "--resume", "")) } else { None };
            let ta = TrainArgs {
                tokens_path: get_arg(&args, "--tokens", ""),
                bpe_path: get_arg(&args, "--bpe", ""),
                out_model: get_arg(&args, "--output", ""),
                resume_model: resume,
                d_model: get_arg(&args, "--d-model", "192").parse().unwrap_or(192),
                n_layers: get_arg(&args, "--layers", "4").parse().unwrap_or(4),
                n_experts: get_arg(&args, "--experts", "8").parse().unwrap_or(8),
                top_k: get_arg(&args, "--top-k", "2").parse().unwrap_or(2),
                d_ff: get_arg(&args, "--d-ff", "192").parse().unwrap_or(192),
                peer_sub_k: get_arg(&args, "--peer-sub-k", "256").parse().unwrap_or(256),
                peer_heads: get_arg(&args, "--peer-heads", "4").parse().unwrap_or(4),
                peer_k: get_arg(&args, "--peer-k", "4").parse().unwrap_or(4),
                recur_steps: get_arg(&args, "--recur-steps", "1").parse().unwrap_or(1),
                attn_window: get_arg(&args, "--attn-window", "128").parse().unwrap_or(128),
                sft_mask: has_flag(&args, "--sft-mask"),
                seq_len: get_arg(&args, "--seq", "128").parse().unwrap_or(128),
                threads: get_arg(&args, "--threads", "4").parse().unwrap_or(4),
                steps: get_arg(&args, "--steps", "200000").parse().unwrap_or(200000),
                max_seconds: get_arg(&args, "--max-seconds", "0").parse().unwrap_or(0.0),
                lr: get_arg(&args, "--lr", "0.0012").parse().unwrap_or(0.0012),
                sampled_k: get_arg(&args, "--sampled-k", "128").parse().unwrap_or(128),
                log_every: get_arg(&args, "--log-every", "50").parse().unwrap_or(50),
            };
            run_train(&ta).expect("training failed");
        }
        "chat" => {
            let mp = get_arg(&args, "--model", "");
            let bp = get_arg(&args, "--bpe", "");
            let prompt = get_arg(&args, "--prompt", "<USER> chào <BOT>");
            let max_new = get_arg(&args, "--max-new", "64").parse().unwrap_or(64);
            let temp = get_arg(&args, "--temp", "0.60").parse().unwrap_or(0.60);
            let top_p = get_arg(&args, "--top-p", "0.9").parse().unwrap_or(0.9);
            let seed = get_arg(&args, "--seed", "42").parse().unwrap_or(42);
            let think = get_arg(&args, "--think-steps", "2").parse().unwrap_or(2);
            let model = GenesisV6Model::load(&mp).expect("load model");
            let bpe = ByteBpeTokenizer::load(&bp).expect("load bpe");
            let reply = generate_text_with_think(&model, &bpe, &prompt, max_new, temp, top_p, seed, think);
            println!("{}", reply);
        }
        "convert-gv8" => {
            let inp = get_arg(&args, "--input", "");
            let out = get_arg(&args, "--output", "");
            let pfmt: u32 = get_arg(&args, "--peer-fmt", "1").parse().unwrap_or(1);
            let t0 = std::time::Instant::now();
            gv8_codec::convert_gv6_to_gv8(&inp, &out, pfmt).expect("convert gv6 -> gv8");
            println!("[GV8] Converted {} -> {} (peer_fmt={}) in {:.2}s", inp, out, pfmt, t0.elapsed().as_secs_f64());
        }
        other => {
            eprintln!("Unknown subcommand: {}", other);
            std::process::exit(1);
        }
    }
}
