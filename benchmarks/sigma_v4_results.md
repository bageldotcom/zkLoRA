# Sigma-v4 backend performance results

Measured on the same class of host as the v3 results (4-core x86-64
container, release builds, default fixed-point config `scale_bits=20,
value_bits=63, intermediate_bits=127`, trivial scaling). "v3" numbers are
from `native_v3_results.md` on identical hardware. The v4 backend has no
proving-key or SRS generation, so there is no warm/cold split: the first
proof of a shape costs the same as every other (v3 "cold" paid keygen on
every fresh process, which is the realistic first-use cost).

Reproduce with `cargo run --release --example bench_prove -- <in> <rank> <out> <reps> sigma`
(`halo2` as the last argument benchmarks the legacy backend) and
`python benchmarks/run_benchmarks.py`.

## Single proof + verify (Rust, `bench_prove`, steady state)

| shape (in×rank×out) | v3 prove warm | v3 prove cold | v4 prove | speedup warm / cold | v3 verify | v4 verify | proof bytes |
|---|---|---|---|---|---|---|---|
| 2×1×2     | 0.73 s | 2.9 s   | 15 ms  | 49× / 193×    | 0.020 s | 16 ms  | 31 KB |
| 8×2×8     | 1.4 s  | 6.0 s   | 17 ms  | 82× / 353×    | 0.034 s | 17 ms  | 36 KB |
| 16×2×16   | 7.0 s  | 27.2 s  | 21 ms  | 333× / 1295×  | 0.16 s  | 20 ms  | 45 KB |
| 64×4×64   | 10.8 s | 47.2 s  | 35 ms  | 309× / 1349×  | 0.22 s  | 36 ms  | 99 KB |
| 768×2×256 | 81 s   | 429 s   | 97 ms  | 835× / 4423×  | 1.3 s   | 101 ms | 327 KB |
| 768×4×768 | infeasible (>15 GB) | infeasible | 183 ms | ∞ | — | 205 ms | 611 KB |
| 768×4×2304 | infeasible (est. >200 GB pk) | infeasible | 470 ms | ∞ | — | 587 ms | 2.2 MB |

Proving memory drops from gigabytes (12.1 GB at k=19; OOM beyond) to tens of
megabytes for every shape: the halo2 extended-domain evaluations are gone
entirely.

## End-to-end Python pipeline (`run_benchmarks.py`, 4 workers)

| shape | invocations | v3 prove wall | v4 prove wall | speedup | v3 verify wall | v4 verify wall |
|---|---|---|---|---|---|---|
| 16×2×16 | 8 | 71.4 s | 0.098 s | **729×** | 3.4 s | 0.23 s |
| 32×4×32 | 6 | 107.4 s | 0.097 s | **1107×** | 3.8 s | 0.61 s |
| 768×4×768 | 4 | infeasible | 0.48 s | ∞ | — | 12.7 s* |

\* dominated by the one-time adapter-setup verification (Bulletproofs over
all 24,576 committed weights), paid once per adapter per verifier process
and cached afterwards; the per-invocation verification is ~0.2 s.

## One-time adapter setup (manifest creation)

The per-weight work that v3 paid inside **every** invocation proof is paid
once per adapter when the manifest is written: Pedersen row commitments, an
aggregated exact range proof for every weight, and a linking proof.

| adapter shape | weights | setup prove | setup blob |
|---|---|---|---|
| 16×2×16 | 64 | 1.8 s | 17 KB |
| 768×2×256 | 2,048 | 14 s | 485 KB |
| 768×4×2304 | 12,288 | 82 s | 2.5 MB |

This artifact ships inside the pinned adapter manifest; it contains no
weight or salt material.

## Range engines

Per-invocation proofs default to the sumcheck-based LogUp range engine
(microseconds of field work per range entry). `ZKLORA_RANGE_ENGINE=bulletproofs`
opts into Bulletproofs instead: proofs shrink ~5-8× at ~5-10× slower proving
(still 50-100× faster than v3). Both engines prove the identical exact
intervals under the same discrete-log + Fiat-Shamir assumptions, and the
verifier accepts either.
