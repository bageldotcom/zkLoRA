//! Benchmark harness for zkLoRA native proving and verification.
//!
//! Usage:
//!   cargo run --release --example bench_prove -- <in_dim> <rank> <out_dim> [reps] [backend]
//!
//! `backend` is `sigma` (default, the v4 commit-and-prove backend) or
//! `halo2` (the legacy v3 circuit).

use std::time::Instant;

use _native_prover::bench_support::{
    bench_statement_and_witness, bench_statement_and_witness_v4, prove_verify_once,
    prove_verify_once_v4,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let in_dim: usize = args.get(1).map(|v| v.parse().unwrap()).unwrap_or(8);
    let rank: usize = args.get(2).map(|v| v.parse().unwrap()).unwrap_or(2);
    let out_dim: usize = args.get(3).map(|v| v.parse().unwrap()).unwrap_or(8);
    let reps: usize = args.get(4).map(|v| v.parse().unwrap()).unwrap_or(1);
    let backend: String = args.get(5).cloned().unwrap_or_else(|| "sigma".to_string());

    if backend == "halo2" {
        let (statement_json, witness_json, k) =
            bench_statement_and_witness(in_dim, rank, out_dim);
        println!("backend=halo2 shape in_dim={in_dim} rank={rank} out_dim={out_dim} k={k} reps={reps}");
        for rep in 0..reps {
            let start = Instant::now();
            let (prove_ms, verify_ms, proof_len) =
                prove_verify_once(&statement_json, &witness_json);
            println!(
                "rep={rep} prove_ms={prove_ms:.1} verify_ms={verify_ms:.1} total_ms={:.1} proof_bytes={proof_len}",
                start.elapsed().as_secs_f64() * 1000.0
            );
        }
        return;
    }

    let setup_start = Instant::now();
    let (statement_json, witness_json, setup_json) =
        bench_statement_and_witness_v4(in_dim, rank, out_dim);
    println!(
        "backend=sigma shape in_dim={in_dim} rank={rank} out_dim={out_dim} reps={reps} adapter_setup_ms={:.1} setup_bytes={}",
        setup_start.elapsed().as_secs_f64() * 1000.0,
        setup_json.len()
    );
    for rep in 0..reps {
        let start = Instant::now();
        let (prove_ms, verify_ms, proof_len) =
            prove_verify_once_v4(&statement_json, &witness_json, &setup_json);
        println!(
            "rep={rep} prove_ms={prove_ms:.1} verify_ms={verify_ms:.1} total_ms={:.1} proof_bytes={proof_len}",
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
}
