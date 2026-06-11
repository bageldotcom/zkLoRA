"""Legacy v3 (halo2, schema 2) artifacts must keep verifying.

This test reconstructs proof artifacts byte-for-byte the way `main`
(commit 8b53724, backend `zklora-halo2-v3`, schema_version 2) wrote them --
poseidon adapter commitment, halo2 proof bytes, legacy circuit/vk
fingerprints -- and checks that the current verifier accepts them through
the legacy path while still rejecting tampering. Only the legacy-artifact
test needs the built native extension; the salt and duplicate-record
hardening tests are pure Python and run everywhere.
"""

import hashlib
import importlib
import os
from dataclasses import asdict
from pathlib import Path

import pytest

if os.environ.get("ZKLORA_REQUIRE_NATIVE_EXTENSION") == "1":
    native = importlib.import_module("zklora._native_prover")
else:
    try:
        native = importlib.import_module("zklora._native_prover")
    except ImportError:
        native = None

requires_native = pytest.mark.skipif(
    native is None,
    reason="native PyO3 extension is not built in this environment",
)

from zklora.proof_contract import (
    COMMITMENT_SCHEME,
    LEGACY_ADAPTER_COMMITMENT_SCHEME,
    LEGACY_BACKEND_ID,
    LEGACY_PROOF_KIND,
    LEGACY_SCHEMA_VERSION,
    FixedPointConfig,
    ProofContractError,
    artifact_prefix,
    canonical_json,
    circuit_id,
    compute_delta_quantized,
    digest_hex,
    lora_commitment,
    transcript_entry_from_statement,
    verify_artifacts,
    vk_fingerprint,
)


def _legacy_adapter_commitment(a, b, scaling_num, scaling_den, fp):
    # main's adapter_commitment_payload used SCHEMA_VERSION = 2 and fed the
    # canonical JSON to the native poseidon hasher.
    payload = {
        "schema_version": LEGACY_SCHEMA_VERSION,
        "in_dim": len(a[0]),
        "rank": len(a),
        "out_dim": len(b),
        "fixed_point": asdict(fp),
        "scaling_num": int(scaling_num),
        "scaling_den": int(scaling_den),
        "a": a,
        "b": b,
    }
    return str(native.adapter_commitment(canonical_json(payload)))


def _write_legacy_artifacts(output_dir, fp):
    """Replicates main's statement_from_witness + write_invocation_artifacts."""
    a = [[2, -1]]
    b = [[3], [-2]]
    x = [4, 5]
    delta = compute_delta_quantized(a, b, x, 1, 2, fp)
    adapter_metadata = {"rank": 1, "in_dim": 2, "out_dim": 2}
    circuit = circuit_id(2, 1, 2, fp, backend=LEGACY_BACKEND_ID)
    statement = {
        "schema_version": LEGACY_SCHEMA_VERSION,
        "backend": LEGACY_BACKEND_ID,
        "session_id": "legacy-session",
        "module_name": "transformer.h.0.attn.c_attn",
        "invocation_index": 0,
        "input_shape": [2],
        "output_shape": [2],
        "x": x,
        "delta": delta,
        "scaling": {"num": 1, "den": 2},
        "fixed_point": asdict(fp),
        "adapter_metadata": adapter_metadata,
        "lora_commitment": lora_commitment(a, b, adapter_metadata),
        "adapter_commitment": {
            "scheme": LEGACY_ADAPTER_COMMITMENT_SCHEME,
            "value": _legacy_adapter_commitment(a, b, 1, 2, fp),
        },
        "circuit_id": circuit,
        "vk_fingerprint": vk_fingerprint(circuit, backend=LEGACY_BACKEND_ID),
        "commitment_scheme": COMMITMENT_SCHEME,
        "invocation_strategy": "one-proof-per-module-invocation-v1",
    }
    statement["statement_digest"] = digest_hex(
        {k: v for k, v in statement.items() if k != "statement_digest"}
    )

    native_statement = canonical_json(
        {
            "x": statement["x"],
            "delta": statement["delta"],
            "fixed_point": statement["fixed_point"],
            "rank": 1,
            "scaling_num": 1,
            "scaling_den": 2,
            "adapter_commitment": statement["adapter_commitment"]["value"],
            "statement_digest": statement["statement_digest"],
        }
    )
    proof_bytes = native.prove(native_statement, canonical_json({"a": a, "b": b}))

    prefix = artifact_prefix(output_dir, statement)
    vk = {
        "schema_version": LEGACY_SCHEMA_VERSION,
        "backend": LEGACY_BACKEND_ID,
        "circuit_id": circuit,
        "vk_fingerprint": statement["vk_fingerprint"],
    }
    pk = {
        "schema_version": LEGACY_SCHEMA_VERSION,
        "backend": LEGACY_BACKEND_ID,
        "circuit_id": circuit,
        "pk_fingerprint": digest_hex({"pk_for_circuit": circuit}),
    }
    meta = {
        "schema_version": LEGACY_SCHEMA_VERSION,
        "backend": LEGACY_BACKEND_ID,
        "proof_file": f"{prefix.name}.zklora.proof",
        "statement_file": f"{prefix.name}.zklora.statement.json",
        "vk_file": f"{prefix.name}.zklora.vk",
        "pk_file": f"{prefix.name}.zklora.pk",
        "statement_digest": statement["statement_digest"],
        "statement_file_digest": digest_hex(statement),
        "proof_digest": hashlib.sha256(proof_bytes).hexdigest(),
        "proof_kind": LEGACY_PROOF_KIND,
        "vk_digest": digest_hex(vk),
        "circuit_id": circuit,
        "vk_fingerprint": statement["vk_fingerprint"],
    }
    Path(f"{prefix}.zklora.proof").write_bytes(proof_bytes)
    Path(f"{prefix}.zklora.vk").write_text(canonical_json(vk) + "\n", encoding="utf-8")
    Path(f"{prefix}.zklora.pk").write_text(canonical_json(pk) + "\n", encoding="utf-8")
    Path(f"{prefix}.zklora.statement.json").write_text(
        canonical_json(statement) + "\n", encoding="utf-8"
    )
    Path(f"{prefix}.zklora.meta.json").write_text(
        canonical_json(meta) + "\n", encoding="utf-8"
    )

    manifest_entry = {
        "module_name": statement["module_name"],
        "rank": 1,
        "in_dim": 2,
        "out_dim": 2,
        "fixed_point": asdict(fp),
        "scaling": {"num": 1, "den": 2},
        "adapter_commitment": statement["adapter_commitment"],
    }
    return statement, manifest_entry, prefix


@requires_native
def test_legacy_v3_artifacts_still_verify_and_reject_tampering(tmp_path):
    fp = FixedPointConfig(scale_bits=0, value_bits=8, intermediate_bits=16)
    statement, manifest_entry, prefix = _write_legacy_artifacts(tmp_path, fp)
    transcript = [transcript_entry_from_statement(statement)]

    total_time, count = verify_artifacts(tmp_path, transcript, [manifest_entry])
    assert count == 1
    assert total_time >= 0

    Path(f"{prefix}.zklora.proof").write_bytes(b"tampered-legacy-proof")
    with pytest.raises(ProofContractError, match="proof bytes"):
        verify_artifacts(tmp_path, transcript, [manifest_entry])


def test_adapter_salt_is_path_keyed_and_atomic(tmp_path, monkeypatch):
    import concurrent.futures

    import zklora.proof_contract as pc

    monkeypatch.setattr(pc, "_ADAPTER_SALTS", {})
    salt_file = tmp_path / "salt-a"
    monkeypatch.setenv("ZKLORA_ADAPTER_SALT_FILE", str(salt_file))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        salts = {pool.submit(pc.adapter_salt).result() for _ in range(16)}
    assert len(salts) == 1
    persisted = salt_file.read_text().strip()
    assert persisted == next(iter(salts))

    # A different configured path yields its own salt at the next call
    # instead of being shadowed by the first one touched.
    other_file = tmp_path / "salt-b"
    monkeypatch.setenv("ZKLORA_ADAPTER_SALT_FILE", str(other_file))
    other = pc.adapter_salt()
    assert other != persisted
    monkeypatch.setenv("ZKLORA_ADAPTER_SALT_FILE", str(salt_file))
    assert pc.adapter_salt() == persisted


def test_adapter_salt_empty_file_is_reported_not_looped(tmp_path, monkeypatch):
    # A stale empty salt file (e.g. left by a crash) must surface as a clear
    # bounded error, not poison the path forever; and salt creation itself
    # must never place an empty file at the path (content is hard-linked in
    # atomically), so this state can only come from pre-fix code or manual
    # tampering.
    import time

    import zklora.proof_contract as pc

    monkeypatch.setattr(pc, "_ADAPTER_SALTS", {})
    salt_file = tmp_path / "salt-stale"
    salt_file.write_text("")
    monkeypatch.setenv("ZKLORA_ADAPTER_SALT_FILE", str(salt_file))
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    with pytest.raises(pc.ProofContractError, match="empty"):
        pc.adapter_salt()


def test_generate_proofs_rejects_duplicate_invocation_keys(tmp_path, monkeypatch):
    import zklora.proof_contract as pc
    from zklora.proof_contract import InvocationWitness, quantize_nested
    from zklora.zk_proof_generator import generate_proofs

    class FakeNative:
        def adapter_commitment_v4(self, adapter_json, salt):
            return hashlib.sha256(f"{salt}|{adapter_json}".encode()).hexdigest()

        def prove_v4(self, statement_json, witness_json, salt):
            return b"proof"

    monkeypatch.setattr(pc, "_native_module", lambda: FakeNative())
    fp = FixedPointConfig(scale_bits=2, value_bits=16, intermediate_bits=32)
    a = quantize_nested([[1.0, 1.0]], fp)
    b = quantize_nested([[1.0], [1.0]], fp)
    x = [4, 4]
    delta = compute_delta_quantized(a, b, x, 1, 1, fp)
    witness = InvocationWitness(
        session_id="s",
        module_name="m.c_attn",
        invocation_index=0,
        input_shape=[2],
        output_shape=[2],
        x=x,
        delta=delta,
        a=a,
        b=b,
        scaling_num=1,
        scaling_den=1,
        adapter_metadata={"rank": 1},
        fixed_point=fp,
    )
    with pytest.raises(ProofContractError, match="duplicate invocation record"):
        generate_proofs(records=[witness, witness], output_dir=str(tmp_path))
