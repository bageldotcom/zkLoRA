//! Benchmark harness for zkLoRA native proving and verification.
//!
//! Usage: cargo run --release --example bench_prove -- <in_dim> <rank> <out_dim> [reps]

use std::time::Instant;

use _native_prover::bench_support::{bench_statement_and_witness, prove_verify_once};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let in_dim: usize = args.get(1).map(|v| v.parse().unwrap()).unwrap_or(8);
    let rank: usize = args.get(2).map(|v| v.parse().unwrap()).unwrap_or(2);
    let out_dim: usize = args.get(3).map(|v| v.parse().unwrap()).unwrap_or(8);
    let reps: usize = args.get(4).map(|v| v.parse().unwrap()).unwrap_or(1);

    let (statement_json, witness_json, k) = bench_statement_and_witness(in_dim, rank, out_dim);
    println!("shape in_dim={in_dim} rank={rank} out_dim={out_dim} k={k} reps={reps}");

    for rep in 0..reps {
        let start = Instant::now();
        let (prove_ms, verify_ms, proof_len) = prove_verify_once(&statement_json, &witness_json);
        println!(
            "rep={rep} prove_ms={prove_ms:.1} verify_ms={verify_ms:.1} total_ms={:.1} proof_bytes={proof_len}",
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
}
