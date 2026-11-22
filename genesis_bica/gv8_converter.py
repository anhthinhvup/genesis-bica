"""
Genesis-v8 `GV8MOD01` Packed Full-RAM Converter (`mel.v6_lm.gv8_converter`, Spec #87).
Converts `GV6MOD01..03` checkpoints to `GV8MOD01`:
  - Drops inference-unused `BitLinear.w: f32` and `PEER vu/vv` optimizer states;
  - Packs BitNet b1.58 ternary weights into bit-exact 2-bit block-32 (`{00:0, 01:+1, 10:-1}`);
  - Packs PEER experts into Full-RAM `Q8 per-row` (`i8` + `f16`/`f32` row scale) or `Q14` lossless.
Strictly <= 280 LOC | Gate-1 AST clean (0 torch/sklearn/transformers/re) | 0 B on C:\\.
"""
from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
import struct
import time
from typing import Any
import numpy as np

from ._env_guard import assert_not_c_drive, enforce_e_drive_environment

enforce_e_drive_environment()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DLL = PROJECT_ROOT / "models" / "genesis_chat_3m" / "bin" / "genesis_v6.dll"

GV8_MAGIC = b"GV8MOD01"
GV6_VERSIONS = {b"GV6MOD01": 1, b"GV6MOD02": 2, b"GV6MOD03": 3}
BITNET_FMT_I2_B32 = 1
PEER_FMT_Q8_F16 = 1
PEER_FMT_Q8_F32 = 2
PEER_FMT_Q14_I16 = 3
DENSE_FMT_F32 = 0
Q14_INV = 1.0 / 16384.0

PEER_FMT_NAMES = {
    "q8_f16": PEER_FMT_Q8_F16,
    "q8_f32": PEER_FMT_Q8_F32,
    "q14": PEER_FMT_Q14_I16,
}


def inspect_gv8_header(path: Path | str) -> dict[str, Any]:
    """Reads and returns the 64-byte `GV8MOD01` binary header metadata."""
    p = Path(path)
    with p.open("rb") as f:
        magic = f.read(8)
        if magic != GV8_MAGIC:
            raise ValueError(f"Expected magic {GV8_MAGIC!r}, got {magic!r}")
        u = struct.unpack("<14I", f.read(56))
    return {
        "magic": magic.decode("ascii"),
        "vocab": u[0],
        "d_model": u[1],
        "n_layers": u[2],
        "n_experts": u[3],
        "top_k": u[4],
        "d_ff": u[5],
        "peer_sub_k": u[6],
        "peer_heads": u[7],
        "peer_k": u[8],
        "recur_steps": u[9],
        "attn_window": u[10],
        "bitnet_fmt": u[11],
        "peer_fmt": u[12],
        "dense_fmt": u[13],
        "file_bytes": p.stat().st_size,
    }


def pack_2bit_block32_numpy(q_i8: np.ndarray, rows: int, cols: int) -> bytes:
    """Packs `q_i8` (`{-1,0,+1}`, shape `[rows, cols]`) into 2-bit block-32 layout matching Rust."""
    q2d = np.ascontiguousarray(q_i8, dtype=np.int8).reshape(rows, cols)
    codes = np.zeros_like(q2d, dtype=np.uint8)
    codes[q2d > 0] = 1
    codes[q2d < 0] = 2
    row_bytes = (cols + 3) // 4
    out = np.zeros((rows, row_bytes), dtype=np.uint8)
    full32 = (cols // 32) * 32
    if full32 > 0:
        nb = full32 // 32
        blk = codes[:, :full32].reshape(rows, nb, 4, 8)
        packed = blk[:, :, 0, :] | (blk[:, :, 1, :] << 2) | (blk[:, :, 2, :] << 4) | (blk[:, :, 3, :] << 6)
        out[:, : nb * 8] = packed.reshape(rows, nb * 8)
    if full32 < cols:
        rem = cols - full32
        tail = np.zeros((rows, ((rem + 3) // 4) * 4), dtype=np.uint8)
        tail[:, :rem] = codes[:, full32:]
        t4 = tail.reshape(rows, -1, 4)
        out[:, full32 // 4 :] = t4[:, :, 0] | (t4[:, :, 1] << 2) | (t4[:, :, 2] << 4) | (t4[:, :, 3] << 6)
    return out.tobytes()


def _quant_bitlinear_pack(wf: np.ndarray, rows: int, cols: int) -> tuple[float, bytes]:
    gamma = max(float(np.mean(np.abs(wf, dtype=np.float32), dtype=np.float64)), 1e-6)
    inv = np.float32(1.0 / gamma)
    scaled = wf * inv
    q = np.zeros(wf.shape, dtype=np.int8)
    q[scaled >= 0.5] = 1
    q[scaled <= -0.5] = -1
    return float(np.float32(gamma)), pack_2bit_block32_numpy(q, rows, cols)


def _quant_peer_q14_to_q8_numpy(src_i16: np.ndarray, use_f16: bool) -> tuple[bytes, bytes]:
    n_exp, d = src_i16.shape
    scales_raw = np.zeros(n_exp, dtype=np.float16 if use_f16 else np.float32)
    q8_out = np.zeros((n_exp, d), dtype=np.int8)
    chunk = 16384
    for s0 in range(0, n_exp, chunk):
        s1 = min(s0 + chunk, n_exp)
        blk = src_i16[s0:s1]
        mx = np.max(np.abs(blk.astype(np.int32)), axis=1)
        sc = (mx.astype(np.float32) * np.float32(Q14_INV)) / np.float32(127.0)
        if use_f16:
            sc16 = sc.astype(np.float16)
            scales_raw[s0:s1] = sc16
            act = sc16.astype(np.float32)
        else:
            scales_raw[s0:s1] = sc
            act = sc
        nz = act > 0.0
        inv = np.zeros_like(act)
        inv[nz] = np.float32(Q14_INV) / act[nz]
        qf = np.rint(blk.astype(np.float32) * inv[:, None])
        q8_out[s0:s1] = np.clip(qf, -127.0, 127.0).astype(np.int8)
    return scales_raw.tobytes(), q8_out.tobytes()


def convert_gv6_to_gv8_numpy(
    src_path: Path | str,
    dst_path: Path | str,
    peer_fmt: int = PEER_FMT_Q8_F32,
) -> dict[str, Any]:
    """Standalone NumPy streaming converter from `GV6MOD01..03` to `GV8MOD01`."""
    src = Path(src_path)
    dst = Path(dst_path)
    assert_not_c_drive(dst)
    if peer_fmt not in (PEER_FMT_Q8_F16, PEER_FMT_Q8_F32, PEER_FMT_Q14_I16):
        raise ValueError(f"Unsupported peer_fmt: {peer_fmt}")
    t0 = time.perf_counter()
    tmp = dst.with_suffix(dst.suffix + ".tmp")

    with src.open("rb") as fr, tmp.open("wb") as fw:
        magic = fr.read(8)
        if magic not in GV6_VERSIONS:
            raise ValueError(f"Unsupported GV6 magic: {magic!r}")
        ver = GV6_VERSIONS[magic]
        n_hdr = 6 if ver == 1 else (10 if ver == 2 else 11)
        u = list(struct.unpack(f"<{n_hdr}I", fr.read(4 * n_hdr)))
        while len(u) < 11:
            u.append(0)
        vocab, d, n_layers, n_exp_moe, top_k, d_ff = u[0], u[1], u[2], u[3], u[4], u[5]
        peer_sub_k = u[6] if ver >= 2 else 0
        peer_heads = u[7] if ver >= 2 else 4
        peer_k = u[8] if ver >= 2 else 8
        recur_steps = max(u[9], 1) if ver >= 2 else 1
        attn_window = u[10] if ver >= 3 else 0

        fw.write(GV8_MAGIC)
        fw.write(
            struct.pack(
                "<14I",
                vocab, d, n_layers, n_exp_moe, top_k, d_ff,
                peer_sub_k, peer_heads, peer_k, recur_steps, attn_window,
                BITNET_FMT_I2_B32, peer_fmt, DENSE_FMT_F32,
            )
        )

        def copy_f32(n: int) -> None:
            raw = fr.read(n * 4)
            if len(raw) != n * 4:
                raise EOFError("Truncated GV6 dense tensor")
            fw.write(raw)

        def conv_bitlinear(rows: int, cols: int) -> None:
            raw = fr.read(rows * cols * 4)
            wf = np.frombuffer(raw, dtype="<f4")
            gamma, q2 = _quant_bitlinear_pack(wf, rows, cols)
            fw.write(struct.pack("<f", gamma))
            fw.write(q2)

        copy_f32(vocab * d)
        copy_f32(d)
        if ver >= 2:
            copy_f32(d)
        else:
            fw.write(np.full(d, 0.25, dtype="<f4").tobytes())

        for _ in range(n_layers):
            copy_f32(d)
            copy_f32(3 * d)
            conv_bitlinear(d, d)
            conv_bitlinear(d, d)
            conv_bitlinear(d, d)
            copy_f32(d)
            copy_f32(n_exp_moe * d)
            for _ in range(n_exp_moe):
                conv_bitlinear(d_ff, d)
            for _ in range(n_exp_moe):
                conv_bitlinear(d, d_ff)

        n_peers = (2 if n_layers >= 4 else 1) if peer_sub_k > 0 else 0
        for _ in range(n_peers):
            nh = max(peer_heads, 1)
            sd = (d // nh) // 2
            n_exp = peer_sub_k * peer_sub_k
            copy_f32(d)
            copy_f32(d * d)
            copy_f32(nh * peer_sub_k * sd)
            copy_f32(nh * peer_sub_k * sd)
            fr.seek(2 * n_exp * 4, os.SEEK_CUR)  # drop vu, vv
            u16 = np.frombuffer(fr.read(n_exp * d * 2), dtype="<i2").reshape(n_exp, d)
            v16 = np.frombuffer(fr.read(n_exp * d * 2), dtype="<i2").reshape(n_exp, d)
            if peer_fmt in (PEER_FMT_Q8_F16, PEER_FMT_Q8_F32):
                use_f16 = peer_fmt == PEER_FMT_Q8_F16
                us_b, uq_b = _quant_peer_q14_to_q8_numpy(u16, use_f16)
                vs_b, vq_b = _quant_peer_q14_to_q8_numpy(v16, use_f16)
                fw.write(us_b)
                fw.write(vs_b)
                fw.write(uq_b)
                fw.write(vq_b)
            else:
                fw.write(u16.tobytes())
                fw.write(v16.tobytes())

        if attn_window > 0:
            copy_f32(d + 4 * d * d)

    os.replace(tmp, dst)
    elapsed = time.perf_counter() - t0
    src_b = src.stat().st_size
    dst_b = dst.stat().st_size
    return {
        "src_bytes": src_b,
        "dst_bytes": dst_b,
        "ratio": src_b / max(dst_b, 1),
        "elapsed_sec": elapsed,
        "backend": "numpy",
        "header": inspect_gv8_header(dst),
    }


def convert_gv6_to_gv8(
    src_path: Path | str,
    dst_path: Path | str,
    peer_fmt: int = PEER_FMT_Q8_F32,
    dll_path: Path | None = None,
    prefer_rust: bool = True,
) -> dict[str, Any]:
    """Converts `GV6MOD01..03` -> `GV8MOD01` via Rust `genesis_v6.dll` (bit-exact gamma) or NumPy."""
    src = Path(src_path)
    dst = Path(dst_path)
    assert_not_c_drive(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dll_p = Path(dll_path) if dll_path is not None else DEFAULT_DLL
    if prefer_rust and dll_p.is_file():
        t0 = time.perf_counter()
        dll = ctypes.CDLL(str(dll_p))
        fn = dll.v6_convert_gv6_to_gv8
        fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        fn.restype = ctypes.c_int
        rc = fn(str(src).encode("utf-8"), str(dst).encode("utf-8"), int(peer_fmt))
        if rc != 0:
            raise RuntimeError(f"v6_convert_gv6_to_gv8 failed with rc={rc}")
        elapsed = time.perf_counter() - t0
        src_b = src.stat().st_size
        dst_b = dst.stat().st_size
        return {
            "src_bytes": src_b,
            "dst_bytes": dst_b,
            "ratio": src_b / max(dst_b, 1),
            "elapsed_sec": elapsed,
            "backend": "rust_dll",
            "header": inspect_gv8_header(dst),
        }
    return convert_gv6_to_gv8_numpy(src, dst, peer_fmt=peer_fmt)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Convert Genesis GV6MOD03 checkpoint to GV8MOD01 Full-RAM Q8")
    ap.add_argument("src", help="Input GV6 .bin path on Drive E:")
    ap.add_argument("dst", help="Output GV8 .gv8 path on Drive E:")
    ap.add_argument("--peer-fmt", choices=tuple(PEER_FMT_NAMES.keys()), default="q8_f32")
    ap.add_argument("--numpy-only", action="store_true", help="Force pure NumPy backend")
    args = ap.parse_args(argv)
    res = convert_gv6_to_gv8(
        args.src,
        args.dst,
        peer_fmt=PEER_FMT_NAMES[args.peer_fmt],
        prefer_rust=not args.numpy_only,
    )
    print(
        f"[GV8-CONV] {args.src} -> {args.dst} | backend={res['backend']} | "
        f"{res['src_bytes'] / (1024 ** 2):.2f} MiB -> {res['dst_bytes'] / (1024 ** 2):.2f} MiB "
        f"({res['ratio']:.2f}x) in {res['elapsed_sec']:.2f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
