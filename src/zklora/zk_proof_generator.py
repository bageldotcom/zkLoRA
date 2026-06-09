from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

from .proof_contract import (
    InvocationWitness,
    ProofContractError,
    TranscriptEntry,
    write_invocation_artifacts,
)

_BACKEND_ENV = "ZKLORA_PROVER_BACKEND"
_BACKEND_PROJECTION = "projection-v1"
_BACKEND_LEGACY = "legacy-halo2"


def _prover_backend() -> str:
    backend = os.environ.get(_BACKEND_ENV, _BACKEND_PROJECTION)
    if backend not in (_BACKEND_PROJECTION, _BACKEND_LEGACY):
        raise ProofContractError(
            f"unknown {_BACKEND_ENV} value {backend!r}; expected "
            f"{_BACKEND_PROJECTION!r} or {_BACKEND_LEGACY!r}"
        )
    return backend


def _generate_proofs_legacy(
    record_list: list[InvocationWitness], output_dir: str, verbose: bool
) -> tuple[int, int]:
    proofs = 0
    total_params = 0
    for record in record_list:
        write_invocation_artifacts(output_dir, record)
        total_params += record.rank * record.in_dim + record.out_dim * record.rank
        proofs += 1
        if verbose:
            print(
                f"Generated native zkLoRA proof artifact for "
                f"{record.module_name}#{record.invocation_index}"
            )
    return proofs, total_params


def generate_proofs(
    records: Iterable[InvocationWitness] | None = None,
    output_dir: str = "proof_artifacts",
    verbose: bool = False,
    *,
    adapter_manifest: str | os.PathLike[str] | dict[str, Any] | None = None,
    manifest_secret_path: str | os.PathLike[str] | None = None,
    **_legacy_kwargs,
) -> tuple[float, float, float, int, int]:
    """Generate native zkLoRA proof artifacts for captured LoRA invocations.

    The default ``projection-v1`` backend writes one artifact set per
    contiguous batch of invocations (``proofs`` in the returned tuple counts
    artifact sets, not rows) and requires the pinned schema-3 adapter manifest
    plus the contributor secret used to commit it. Setting
    ``ZKLORA_PROVER_BACKEND=legacy-halo2`` selects the unsupported legacy
    rollback hatch, which writes one schema-2 artifact set per invocation row.
    Legacy keyword arguments are accepted so old callers fail with a clear
    no-records result instead of importing removed proof backends.
    """

    import time

    start = time.time()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    record_list = list(records or [])
    if not record_list:
        raise ProofContractError(
            "native zkLoRA proof generation requires captured invocation records"
        )

    backend = _prover_backend()
    if backend == _BACKEND_LEGACY:
        proofs, total_params = _generate_proofs_legacy(record_list, output_dir, verbose)
    else:
        from .proof_v3 import generate_batch_proofs

        proofs, total_params = generate_batch_proofs(
            record_list,
            output_dir,
            adapter_manifest=adapter_manifest,
            manifest_secret_path=manifest_secret_path,
            verbose=verbose,
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
    from .proof_v3 import verify_artifacts_mixed

    total_time, count = verify_artifacts_mixed(proof_dir, transcript, expected_adapters)
    if verbose:
        print(f"Verified {count} native zkLoRA proof artifacts in {total_time:.2f}s")
    return total_time, count
