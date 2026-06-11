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

    import os
    import time
    from concurrent.futures import ThreadPoolExecutor

    start = time.time()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    proofs = 0
    total_params = 0

    record_list = list(records or [])
    if not record_list:
        raise ProofContractError(
            "native zkLoRA proof generation requires captured invocation records"
        )
    # Artifact paths are derived from (session_id, module_name,
    # invocation_index); duplicate keys would race and silently overwrite
    # each other's files during concurrent generation, so reject them here
    # rather than relying on the verifier's coverage check.
    seen_keys: set[tuple[str, str, int]] = set()
    for record in record_list:
        key = (record.session_id, record.module_name, int(record.invocation_index))
        if key in seen_keys:
            raise ProofContractError(f"duplicate invocation record for {key}")
        seen_keys.add(key)

    def _generate(record: InvocationWitness) -> None:
        write_invocation_artifacts(output_dir, record)
        if verbose:
            print(
                f"Generated native zkLoRA proof artifact for "
                f"{record.module_name}#{record.invocation_index}"
            )

    # The native prover releases the GIL, so independent invocation proofs can
    # be generated concurrently; artifact paths are unique per record.
    try:
        configured = int(os.environ.get("ZKLORA_PROVE_WORKERS", ""))
    except ValueError:
        configured = 0
    max_workers = configured if configured > 0 else max(os.cpu_count() or 1, 1)
    max_workers = min(len(record_list), max_workers)
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_generate, record) for record in record_list]
            for future in futures:
                future.result()
    else:
        for record in record_list:
            _generate(record)

    for record in record_list:
        total_params += record.rank * record.in_dim + record.out_dim * record.rank
        proofs += 1

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
