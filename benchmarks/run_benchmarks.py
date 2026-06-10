#!/usr/bin/env python3
"""End-to-end zkLoRA pipeline benchmark.

Generates invocation witnesses for a synthetic LoRA module, produces native
proof artifacts, and verifies them against the transcript and adapter
manifest, reporting wall-clock timings for each stage.

Usage:
  python benchmarks/run_benchmarks.py [--in_dim 16] [--rank 2] [--out_dim 16]
                                      [--invocations 8]
"""

import argparse
import json
import random
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zklora.proof_contract import (  # noqa: E402
    FixedPointConfig,
    InvocationWitness,
    adapter_manifest_entry,
    compute_delta_quantized,
    statement_from_witness,
    transcript_entry_from_statement,
)
from zklora.zk_proof_generator import batch_verify_proofs, generate_proofs  # noqa: E402


def build_witnesses(in_dim, rank, out_dim, invocations, seed=7):
    rng = random.Random(seed)
    fp = FixedPointConfig()
    magnitude = 1 << fp.scale_bits
    a = [[rng.randint(-magnitude, magnitude) for _ in range(in_dim)] for _ in range(rank)]
    b = [[rng.randint(-magnitude, magnitude) for _ in range(rank)] for _ in range(out_dim)]
    witnesses = []
    for index in range(invocations):
        x = [rng.randint(-magnitude, magnitude) for _ in range(in_dim)]
        delta = compute_delta_quantized(a, b, x, 1, 1, fp)
        witnesses.append(
            InvocationWitness(
                session_id="bench-session",
                module_name="bench.module.c_attn",
                invocation_index=index,
                input_shape=[in_dim],
                output_shape=[out_dim],
                x=x,
                delta=delta,
                a=a,
                b=b,
                scaling_num=1,
                scaling_den=1,
                adapter_metadata={"rank": rank, "in_dim": in_dim, "out_dim": out_dim},
                fixed_point=fp,
            )
        )
    manifest = [adapter_manifest_entry("bench.module.c_attn", a, b, 1, 1, fp)]
    return witnesses, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in_dim", type=int, default=16)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--out_dim", type=int, default=16)
    parser.add_argument("--invocations", type=int, default=8)
    args = parser.parse_args()

    print(
        f"shape in_dim={args.in_dim} rank={args.rank} out_dim={args.out_dim} "
        f"invocations={args.invocations}"
    )

    start = time.time()
    witnesses, manifest = build_witnesses(
        args.in_dim, args.rank, args.out_dim, args.invocations
    )
    print(f"witness generation: {time.time() - start:.2f}s")

    with tempfile.TemporaryDirectory() as tmp:
        proof_dir = Path(tmp) / "artifacts"
        start = time.time()
        _, _, elapsed, total_params, proofs = generate_proofs(
            records=witnesses, output_dir=str(proof_dir)
        )
        prove_wall = time.time() - start
        print(
            f"proof generation: {prove_wall:.2f}s total "
            f"({prove_wall / max(proofs, 1):.2f}s/proof, {proofs} proofs)"
        )

        transcript = [
            transcript_entry_from_statement(statement_from_witness(w)) for w in witnesses
        ]
        start = time.time()
        verify_time, verified = batch_verify_proofs(
            proof_dir=str(proof_dir),
            transcript=transcript,
            expected_adapters={"adapters": manifest},
        )
        verify_wall = time.time() - start
        print(
            f"verification: {verify_wall:.2f}s total "
            f"({verify_wall / max(verified, 1):.2f}s/proof, {verified} proofs)"
        )
        result = {
            "shape": [args.in_dim, args.rank, args.out_dim],
            "invocations": args.invocations,
            "prove_wall_s": round(prove_wall, 3),
            "prove_per_proof_s": round(prove_wall / max(proofs, 1), 3),
            "verify_wall_s": round(verify_wall, 3),
            "verify_per_proof_s": round(verify_wall / max(verified, 1), 3),
            "total_params": total_params,
        }
        print(json.dumps(result))


if __name__ == "__main__":
    main()
