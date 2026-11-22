# Benchmarks & Reproducibility Artifacts

This directory contains our **100% symmetric, SHA-256 locked** international benchmark evaluation harness and verified JSON snapshots measured on identical local CPU hardware (`Intel Core i5-10210U`, 4C/8T, 15W, 16 GB RAM, `0%` GPU).

## 1. SHA-256 Locked Official HuggingFace Datasets

Every dataset is fetched directly from its official HuggingFace test/validation split with zero manual editing and zero train/test overlap:

| Benchmark | Official HuggingFace Source | Split | Total Items | SHA-256 Prefix |
|---|---|---|---:|---|
| **FOLIO** | `tasksource/folio` (Yale NLP FOL) | `validation` | `203` | `6b1df66ccc811fe7` |
| **HumanEval** | `openai/openai_humaneval` | `test` | `164` | `9329d3a61b18b326` |
| **MATH-500** | `HuggingFaceH4/MATH-500` | `test` | `500` | `55e02317c57e6c22` |
| **MBPP-sanitized** | `google-research-datasets/mbpp` | `test` | `257` | `7f47e0d8a7e5011d` |
| **GSM8K** | `openai/gsm8k` (`main`) | `test` | `1,319` | `ca6e7c5108b7ab7b` |
| **ARC-Challenge** | `allenai/ai2_arc` | `test` | `1,172` | `b90e27562e002b98` |
| **MMLU-Pro** | `TIGER-Lab/MMLU-Pro` | `test` | `12,032` | `51d7b60cb50f027a` |
| **TruthfulQA-MC** | `truthfulqa/truthful_qa` | `validation` | `817` | `23f08e230ca4ed66` |
| **UIT-ViQuAD 2.0** | `taidng/UIT-ViQuAD2.0` | `validation` | `3,000+` | `8c81bdd75bd39eef` |

Full SHA-256 manifests are stored in:
- [`results/world_benchmarks_manifest.json`](./results/world_benchmarks_manifest.json)
- [`results/public_bench_manifest.json`](./results/public_bench_manifest.json)

## 2. Verified Result Artifacts

- [`results/world_bench_loop29_final.json`](./results/world_bench_loop29_final.json): Complete 7-suite official international benchmark audit (`FOLIO`, `MATH-500`, `HumanEval`, `MBPP-sanitized`, `GSM8K`, `ARC-Challenge`, `MMLU-Pro`).
- [`results/genesis_v8_bica_vs_multib_results.json`](./results/genesis_v8_bica_vs_multib_results.json): Live same-machine `i5-10210U` CPU comparison (`llama-bench.exe` vs Pure-Rust AVX2 `.gv8`) across decode speed (`565–2,156 tok/s` vs `17.7–46.4 tok/s`), exact symbolic arithmetic, fabricated OOD abstention, and cross-session 1-shot belief revision.
- [`results/head_to_head_results.json`](./results/head_to_head_results.json): Symmetric 120-item public benchmark arena (`seed=42`: `40 MBPP` + `40 GSM8K` + `20 TruthfulQA` + `20 UIT-ViQuAD 2.0`) against `Qwen2.5-0.5B`, `Llama-3.2-1B`, `Qwen2.5-1.5B`, and `Qwen2.5-Coder-1.5B`.

## 3. Reproducing the GPU Throughput Microbenchmark

```bash
# Compare NVIDIA H100 vs NVIDIA B200 throughput on Genesis-v8 Native BICA (457.07M, W=1024)
modal run training/modal_v8_bench.py
```
