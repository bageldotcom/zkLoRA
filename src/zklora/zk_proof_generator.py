from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .proof_contract import (
    InvocationWitness,
    ProofContractError,
    TranscriptEntry,
    verify_artifacts,
    write_invocation_artifacts,
)


def generate_proofs(
    records: Iterable[InvocationWitness] | None = None,
    output_dir: str = "proof_artifacts",
    verbose: bool = False,
    **_legacy_kwargs,
) -> tuple[float, float, float, int, int]:
    """Generate native zkLoRA proof artifacts for captured LoRA invocations.

    The old external-backend implementation scanned model-export directories. The native backend is
    transcript-first: callers pass invocation witnesses captured during multi-party
    inference. Legacy keyword arguments are accepted so old callers fail with a clear
    no-records result instead of importing removed proof backends.
    """

    import time

    start = time.time()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    proofs = 0
    total_params = 0

    record_list = list(records or [])
    if not record_list:
        raise ProofContractError(
            "native zkLoRA proof generation requires captured invocation records"
        )

    for record in record_list:
        write_invocation_artifacts(output_dir, record)
        total_params += record.rank * record.in_dim + record.out_dim * record.rank
        proofs += 1
        if verbose:
            print(
                f"Generated native zkLoRA proof artifact for "
                f"{record.module_name}#{record.invocation_index}"
            )

    elapsed = time.time() - start
    return (0.0, 0.0, elapsed, total_params, proofs)


def batch_verify_proofs(
    proof_dir: str = "proof_artifacts",
    transcript: str | Iterable[TranscriptEntry] | None = None,
    expected_adapters=None,
    verbose: bool = False,
) -> tuple[float, int]:
    """Verify native zkLoRA proof artifacts against the base user's transcript."""

    if transcript is None:
        raise ProofContractError(
            "native zkLoRA verification requires the base user's transcript"
        )
    if expected_adapters is None:
        raise ProofContractError(
            "native zkLoRA verification requires a pre-inference adapter manifest"
        )
    total_time, count = verify_artifacts(proof_dir, transcript, expected_adapters)
    if verbose:
        print(f"Verified {count} native zkLoRA proof artifacts in {total_time:.2f}s")
    return total_time, count
