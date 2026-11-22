//! Integration proofs for Genesis-v8 `.gv8` kernels (BitNet 2-bit exactness & IEEE-754 f16).

use genesis_bica::gv8_kernels::{
    f16_to_f32, f32_to_f16, matvec_bitnet_2bit, pack_2bit_block32,
};

#[test]
fn test_f16_roundtrip_normals_and_zero() {
    for &x in &[0.0f32, -0.0, 1.0, -1.0, 0.5, 3.140625, -42.5, 1024.0] {
        let h = f32_to_f16(x);
        let y = f16_to_f32(h);
        assert!((x - y).abs() <= 1e-2, "f16 mismatch for {x}: got {y}");
    }
}

#[test]
fn test_bitnet_2bit_matvec_exactness() {
    let rows = 8;
    let cols = 64;
    let mut q = vec![0i8; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            q[r * cols + c] = match (r * 7 + c * 3) % 3 {
                0 => 0,
                1 => 1,
                _ => -1,
            };
        }
    }
    let gamma = 0.375f32;
    let x: Vec<f32> = (0..cols).map(|i| ((i as f32) - 31.5) * 0.05).collect();
    let packed = pack_2bit_block32(&q, rows, cols);

    let mut y_packed = vec![0.0f32; rows];
    matvec_bitnet_2bit(&packed, gamma, rows, cols, &x, &mut y_packed);

    for r in 0..rows {
        let mut ref_sum = 0.0f32;
        for c in 0..cols {
            ref_sum += (q[r * cols + c] as f32) * x[c];
        }
        let expected = ref_sum * gamma;
        assert!(
            (y_packed[r] - expected).abs() < 1e-5,
            "row {r}: packed={} expected={}",
            y_packed[r],
            expected
        );
    }
}
