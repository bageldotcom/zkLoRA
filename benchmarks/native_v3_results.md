# Native backend v3 performance results

Measured on a 4-core x86-64 container, 15 GB RAM, release builds, default
fixed-point config (`scale_bits=20, value_bits=63, intermediate_bits=127`).
"v2 baseline" is the previous native circuit (per-bit range checks, no key
caching) at the same commit environment; "warm" means params/proving key are
cached from a previous proof of the same shape (the steady-state for real
workloads, where every invocation of a module shares one circuit shape).

Reproduce with `cargo run --release --example bench_prove -- <in> <rank> <out> <reps>`
and `python benchmarks/run_benchmarks.py`.

## Single proof + verify (Rust, `bench_prove`)

| shape (in×rank×out) | v2 k | v2 prove | v2 verify | v3 k | v3 prove cold | v3 prove warm | v3 verify warm | warm speedup |
|---|---|---|---|---|---|---|---|---|
| 2×1×2   | 15 | 22.9 s | 18.4 s | 12 | 2.9 s | 0.73 s | 0.020 s | 31× / ~860× |
| 8×2×8   | ~18 | >290 s (timed out) | — | 13 | 6.0 s | 1.4 s | 0.034 s | >200× |
| 16×2×16 | ~19 | est. >370 s | — | 14 | 27.2 s | 7.0 s | 0.16 s | >50× |
| 64×4×64 | ~22 | infeasible | — | 17 | 47.2 s | 10.8 s | 0.22 s | n/a (was infeasible) |
| 768×2×256 | ~23 | infeasible (est. ~190 GB) | — | 19 | 429 s | 81 s | 1.3 s (17.1 s cold) | n/a (was infeasible) |
| 768×4×768 | ~25 | infeasible | — | 20 | needs >15 GB RAM host | — | — | n/a |
| 768×4×2304 | ~26 | infeasible (est. >200 GB params/pk) | — | 21 | needs >15 GB RAM host | — | — | n/a |

Verification cold (first proof of a shape) pays one `keygen_vk`; in v2 this
cost was paid for *every* proof.

## End-to-end Python pipeline (`run_benchmarks.py`, 4 workers)

| shape | invocations | prove wall | prove/proof | verify wall | verify/proof |
|---|---|---|---|---|---|
| 16×2×16 | 8 | 71.4 s | 8.9 s | 3.4 s | 0.42 s |
| 32×4×32 | 6 | 107.4 s | 17.9 s | 3.8 s | 0.63 s |

Per-proof wall time includes the one-time cold keygen amortised across the
batch.

## Supporting paths

| path | before | after | speedup |
|---|---|---|---|
| server per-invocation compute, 16 rows @ 768×4×2304 (quantize + delta) | ~233 ms | 13.5 ms | ~17× (≥30× at 1 row) |
| exact delta, 16 rows @ 768×4×2304 | 171 ms | 7.4 ms | 23× |
| hiding Merkle root, 200k leaves | 0.353 s | 0.043 s | 8.3× |

All fast paths are value-identical to the Python reference implementations
(randomised parity tests, including an 80k-value quantisation fuzz across
four fixed-point configs).

## Memory

Proving memory is dominated by halo2 extended-domain evaluations (Poseidon
gate degree ⇒ 8× extended domain), and halo2 0.3 keeps all advice cosets
resident during create_proof. Measured: 12.1 GB peak at k=19 (768×2×256);
k=20 and k=21 shapes exceed a 15 GB host (OOM-killed); plan on 32–64 GB for
768-dim × 4-rank modules with wide outputs.
The proving-key cache holds up to `ZKLORA_PK_CACHE_CAP` (default 2) shapes.
