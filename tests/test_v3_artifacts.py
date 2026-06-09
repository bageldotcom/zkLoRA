import base64
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import zklora.proof_contract as proof_contract  # noqa: E402
from zklora.proof_contract import (  # noqa: E402
    FixedPointConfig,
    InvocationWitness,
    ProofContractError,
    TranscriptEntry,
    adapter_manifest_entry,
    canonical_json,
    compute_delta_quantized,
    flatten,
    load_json,
    quantize_nested,
    write_invocation_artifacts,
)
from zklora.proof_v3 import (  # noqa: E402
    BACKEND_ID_V3,
    PROOF_KIND_V3,
    adapter_manifest_entry_v3,
    adapter_manifest_payload_v3,
    build_batches,
    check_bounds_composition,
    derive_bounds,
    expand_statement_rows,
)
from zklora.zk_proof_generator import batch_verify_proofs, generate_proofs  # noqa: E402

MODULE = "transformer.h.0.attn.c_attn"
SECOND_MODULE = "transformer.h.1.attn.c_attn"
FP = FixedPointConfig(scale_bits=2, value_bits=16, intermediate_bits=32)
SEED = "ab" * 32


class FakeNativeBoth:
    """Deterministic stand-in for the native module covering v2 and v3."""

    def adapter_commitment(self, adapter_json):
        return str(abs(hash(adapter_json)))

    def prove(self, statement_json, witness_json):
        return f"{statement_json}|{witness_json}".encode()

    def verify(self, statement_json, proof):
        return proof.startswith(statement_json.encode() + b"|")

    def adapter_commit_v3(self, adapter_json, seed_hex):
        base = hashlib.sha256((adapter_json + seed_hex).encode()).hexdigest()
        return canonical_json(
            {
                "a_commitment": [base],
                "b_commitment": [base[::-1]],
                "commitment_nonce": hashlib.sha256(base.encode()).hexdigest(),
                "range_proof": base64.b64encode(b"fake-range-proof").decode(),
            }
        )

    def verify_adapter_manifest_v3(self, entry_json):
        return True

    def prove_v3(self, statement_json, rows_json, witness_json):
        return (
            b"ZKL3"
            + hashlib.sha256(statement_json.encode()).digest()
            + hashlib.sha256(rows_json.encode()).digest()
            + hashlib.sha256(witness_json.encode()).digest()
        )

    def verify_v3(self, statement_json, rows_json, manifest_entry_json, proof):
        return (
            proof[:4] == b"ZKL3"
            and proof[4:36] == hashlib.sha256(statement_json.encode()).digest()
            and proof[36:68] == hashlib.sha256(rows_json.encode()).digest()
        )


@pytest.fixture(autouse=True)
def fake_native(monkeypatch):
    monkeypatch.setattr(proof_contract, "_native_module", lambda: FakeNativeBoth())
    monkeypatch.delenv("ZKLORA_PROVER_BACKEND", raising=False)
    monkeypatch.delenv("ZKLORA_CHUNK_ROWS", raising=False)
    monkeypatch.delenv("ZKLORA_CONTRIBUTOR_SECRET", raising=False)


def _adapter():
    a = quantize_nested([[2.0, -1.0]], FP)
    b = quantize_nested([[3.0], [-2.0]], FP)
    return a, b


def _records(n_rows=5, start=0, session_id="s1", module=MODULE):
    a, b = _adapter()
    records = []
    for offset in range(n_rows):
        x = flatten(quantize_nested([1.5 + offset, -0.5], FP))
        delta = compute_delta_quantized(a, b, x, 1, 2, FP)
        records.append(
            InvocationWitness(
                session_id=session_id,
                module_name=module,
                invocation_index=start + offset,
                input_shape=[2],
                output_shape=[2],
                x=x,
                delta=delta,
                a=a,
                b=b,
                scaling_num=1,
                scaling_den=2,
                adapter_metadata={"rank": 1},
                fixed_point=FP,
            )
        )
    return records


def _transcript(records):
    return [
        TranscriptEntry(
            session_id=r.session_id,
            module_name=r.module_name,
            invocation_index=r.invocation_index,
            input_shape=r.input_shape,
            output_shape=r.output_shape,
            x=r.x,
            delta=r.delta,
            fixed_point=r.fixed_point,
            scaling_num=r.scaling_num,
            scaling_den=r.scaling_den,
        )
        for r in records
    ]


def _v3_manifest(extra_entries=()):
    a, b = _adapter()
    entry = adapter_manifest_entry_v3(MODULE, a, b, 1, 2, FP, SEED)
    return adapter_manifest_payload_v3([entry, *extra_entries])


def _generate(tmp_path, records, payload, chunk, monkeypatch):
    monkeypatch.setenv("ZKLORA_CHUNK_ROWS", str(chunk))
    secret = tmp_path / "secrets" / "contributor.secret.json"
    out_dir = tmp_path / "artifacts"
    generate_proofs(
        records,
        output_dir=str(out_dir),
        adapter_manifest=payload,
        manifest_secret_path=secret,
    )
    return out_dir


def test_v3_batch_artifacts_roundtrip(tmp_path, monkeypatch):
    records = _records(5)
    payload = _v3_manifest()
    out_dir = _generate(tmp_path, records, payload, chunk=2, monkeypatch=monkeypatch)

    names = sorted(p.name for p in out_dir.glob("*.zklora.statement.json"))
    assert names == [
        f"s1.{MODULE}.0000.zklora.statement.json",
        f"s1.{MODULE}.0002.zklora.statement.json",
        f"s1.{MODULE}.0004.zklora.statement.json",
    ]
    statement = load_json(out_dir / names[0])
    assert statement["schema_version"] == 3
    assert statement["backend"] == BACKEND_ID_V3
    assert statement["count"] == 2
    for forbidden in ("x", "delta", "lora_commitment", "invocation_index"):
        assert forbidden not in statement
    meta = load_json(out_dir / f"s1.{MODULE}.0000.zklora.meta.json")
    assert meta["proof_kind"] == PROOF_KIND_V3
    assert meta["audit_status"] == "unaudited"

    elapsed, count = batch_verify_proofs(
        proof_dir=str(out_dir),
        transcript=_transcript(records),
        expected_adapters=payload,
    )
    assert elapsed >= 0
    assert count == 3


def test_nonzero_start_and_short_final_chunk(tmp_path, monkeypatch):
    records = _records(5, start=3)
    payload = _v3_manifest()
    out_dir = _generate(tmp_path, records, payload, chunk=4, monkeypatch=monkeypatch)

    names = sorted(p.name for p in out_dir.glob("*.zklora.statement.json"))
    assert names == [
        f"s1.{MODULE}.0003.zklora.statement.json",
        f"s1.{MODULE}.0007.zklora.statement.json",
    ]
    assert load_json(out_dir / names[0])["count"] == 4
    assert load_json(out_dir / names[1])["count"] == 1
    _, count = batch_verify_proofs(
        proof_dir=str(out_dir),
        transcript=_transcript(records),
        expected_adapters=payload,
    )
    assert count == 2


def test_missing_batch_rejected(tmp_path, monkeypatch):
    records = _records(5)
    payload = _v3_manifest()
    out_dir = _generate(tmp_path, records, payload, chunk=2, monkeypatch=monkeypatch)
    (out_dir / f"s1.{MODULE}.0002.zklora.statement.json").unlink()

    with pytest.raises(ProofContractError, match="coverage mismatch"):
        batch_verify_proofs(
            proof_dir=str(out_dir),
            transcript=_transcript(records),
            expected_adapters=payload,
        )


def test_overlapping_coverage_rejected(tmp_path, monkeypatch):
    records = _records(5)
    payload = _v3_manifest()
    out_dir = _generate(tmp_path, records, payload, chunk=2, monkeypatch=monkeypatch)
    # Regenerate with a different chunking into the same directory: the new
    # 5-row batch at start 0 overlaps the surviving 2-row batches.
    _generate(tmp_path, records, payload, chunk=5, monkeypatch=monkeypatch)

    with pytest.raises(ProofContractError, match="duplicate proof coverage"):
        batch_verify_proofs(
            proof_dir=str(out_dir),
            transcript=_transcript(records),
            expected_adapters=payload,
        )


def test_transcript_tamper_and_reorder_rejected(tmp_path, monkeypatch):
    records = _records(4)
    payload = _v3_manifest()
    out_dir = _generate(tmp_path, records, payload, chunk=2, monkeypatch=monkeypatch)

    tampered = _transcript(records)
    tampered[1] = replace(tampered[1], x=[tampered[1].x[0] + 1, *tampered[1].x[1:]])
    with pytest.raises(ProofContractError, match="row digest mismatch"):
        batch_verify_proofs(
            proof_dir=str(out_dir),
            transcript=tampered,
            expected_adapters=payload,
        )

    reordered = _transcript(records)
    first, second = reordered[0], reordered[1]
    reordered[0] = replace(first, x=second.x, delta=second.delta)
    reordered[1] = replace(second, x=first.x, delta=first.delta)
    with pytest.raises(ProofContractError, match="row digest mismatch"):
        batch_verify_proofs(
            proof_dir=str(out_dir),
            transcript=reordered,
            expected_adapters=payload,
        )


def test_zero_artifact_module_reported_missing(tmp_path, monkeypatch):
    records = _records(2)
    payload = _v3_manifest()
    out_dir = _generate(tmp_path, records, payload, chunk=2, monkeypatch=monkeypatch)

    ghost = _records(1, module=SECOND_MODULE)
    with pytest.raises(ProofContractError, match="coverage mismatch"):
        batch_verify_proofs(
            proof_dir=str(out_dir),
            transcript=_transcript(records + ghost),
            expected_adapters=payload,
        )


def test_secret_path_guards(tmp_path, monkeypatch):
    records = _records(2)
    payload = _v3_manifest()
    out_dir = tmp_path / "artifacts"
    with pytest.raises(ProofContractError, match="must not live inside"):
        generate_proofs(
            records,
            output_dir=str(out_dir),
            adapter_manifest=payload,
            manifest_secret_path=out_dir / "contributor.secret.json",
        )

    out_dir = _generate(tmp_path, records, payload, chunk=2, monkeypatch=monkeypatch)
    (out_dir / "leaked.secret.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ProofContractError, match="refusing to verify"):
        batch_verify_proofs(
            proof_dir=str(out_dir),
            transcript=_transcript(records),
            expected_adapters=payload,
        )


def test_mixed_v2_and_v3_directory(tmp_path, monkeypatch):
    a, b = _adapter()
    v2_witness = InvocationWitness(
        session_id="s1",
        module_name=SECOND_MODULE,
        invocation_index=0,
        input_shape=[2],
        output_shape=[2],
        x=flatten(quantize_nested([1.5, -0.5], FP)),
        delta=compute_delta_quantized(
            a, b, flatten(quantize_nested([1.5, -0.5], FP)), 1, 2, FP
        ),
        a=a,
        b=b,
        scaling_num=1,
        scaling_den=2,
        adapter_metadata={"rank": 1},
        fixed_point=FP,
    )
    v2_entry = adapter_manifest_entry(SECOND_MODULE, a, b, 1, 2, FP)
    payload = _v3_manifest(extra_entries=[v2_entry])

    records = _records(3)
    out_dir = _generate(tmp_path, records, payload, chunk=2, monkeypatch=monkeypatch)
    write_invocation_artifacts(out_dir, v2_witness)

    transcript = _transcript(records + [v2_witness])
    _, count = batch_verify_proofs(
        proof_dir=str(out_dir),
        transcript=transcript,
        expected_adapters=payload,
    )
    assert count == 3  # two v3 batches + one v2 single-row artifact


def test_expand_statement_rows_v2_and_v3(tmp_path, monkeypatch):
    records = _records(4)
    payload = _v3_manifest()
    out_dir = _generate(tmp_path, records, payload, chunk=3, monkeypatch=monkeypatch)
    transcript = _transcript(records)

    statement = load_json(out_dir / f"s1.{MODULE}.0000.zklora.statement.json")
    rows = expand_statement_rows(statement, transcript)
    assert [r.invocation_index for r in rows] == [0, 1, 2]
    assert rows[0].x == records[0].x

    v2_statement = proof_contract.statement_from_witness(records[0])
    v2_rows = expand_statement_rows(v2_statement)
    assert len(v2_rows) == 1
    assert v2_rows[0].x == records[0].x


def test_legacy_hatch_generates_v2_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("ZKLORA_PROVER_BACKEND", "legacy-halo2")
    records = _records(3)
    a, b = _adapter()
    out_dir = tmp_path / "artifacts"
    generate_proofs(records, output_dir=str(out_dir))

    statements = sorted(out_dir.glob("*.zklora.statement.json"))
    assert len(statements) == 3
    assert all(load_json(p)["schema_version"] == 2 for p in statements)

    monkeypatch.delenv("ZKLORA_PROVER_BACKEND")
    _, count = batch_verify_proofs(
        proof_dir=str(out_dir),
        transcript=_transcript(records),
        expected_adapters=[adapter_manifest_entry(MODULE, a, b, 1, 2, FP)],
    )
    assert count == 3


def test_unknown_backend_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("ZKLORA_PROVER_BACKEND", "sumcheck-mle")
    with pytest.raises(ProofContractError, match="unknown ZKLORA_PROVER_BACKEND"):
        generate_proofs(_records(1), output_dir=str(tmp_path / "artifacts"))


def test_statement_and_proof_tamper_rejected(tmp_path, monkeypatch):
    records = _records(2)
    payload = _v3_manifest()
    out_dir = _generate(tmp_path, records, payload, chunk=2, monkeypatch=monkeypatch)
    statement_path = out_dir / f"s1.{MODULE}.0000.zklora.statement.json"

    statement = load_json(statement_path)
    statement["count"] = 1
    statement["row_digests"] = statement["row_digests"][:1]
    statement_path.write_text(
        json.dumps(statement, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ProofContractError, match="statement digest mismatch"):
        batch_verify_proofs(
            proof_dir=str(out_dir),
            transcript=_transcript(records),
            expected_adapters=payload,
        )

    out_dir = _generate(
        tmp_path / "second", records, payload, chunk=2, monkeypatch=monkeypatch
    )
    (out_dir / f"s1.{MODULE}.0000.zklora.proof").write_bytes(b"ZKL3tampered")
    with pytest.raises(ProofContractError, match="metadata proof digest mismatch"):
        batch_verify_proofs(
            proof_dir=str(out_dir),
            transcript=_transcript(records),
            expected_adapters=payload,
        )


def test_manifest_commitment_binds_statements(tmp_path, monkeypatch):
    records = _records(2)
    payload = _v3_manifest()
    out_dir = _generate(tmp_path, records, payload, chunk=2, monkeypatch=monkeypatch)

    a, b = _adapter()
    other_entry = adapter_manifest_entry_v3(SECOND_MODULE, a, b, 1, 2, FP, SEED)
    drifted = adapter_manifest_payload_v3([payload["adapters"][0], other_entry])
    with pytest.raises(ProofContractError, match="manifest commitment mismatch"):
        batch_verify_proofs(
            proof_dir=str(out_dir),
            transcript=_transcript(records),
            expected_adapters=drifted,
        )


def test_batch_grouping_contract():
    records = _records(3)
    bad = records + _records(1, start=5)
    with pytest.raises(ProofContractError, match="non-contiguous"):
        build_batches(bad, target_rows=4)

    duplicated = records + [records[-1]]
    with pytest.raises(ProofContractError, match="duplicate invocation index"):
        build_batches(duplicated, target_rows=4)

    a, b = _adapter()
    drifted_b = quantize_nested([[3.0], [-1.0]], FP)
    mixed = records + [
        InvocationWitness(
            **{
                **records[-1].__dict__,
                "invocation_index": 3,
                "b": drifted_b,
                "delta": compute_delta_quantized(a, drifted_b, records[-1].x, 1, 2, FP),
            }
        )
    ]
    with pytest.raises(ProofContractError, match="inconsistent adapter weights"):
        build_batches(mixed, target_rows=4)


def test_bounds_composition_guards():
    bounds = derive_bounds([[4, -2]], [[6, -4]], FP, 1, 2)
    assert bounds["n_rem"] == FP.scale_bits
    assert bounds["proved_u"] >= bounds["b_u"]
    check_bounds_composition(
        bounds, in_dim=2, rank=1, fixed_point=FP, scaling_num=1, scaling_den=2
    )

    huge = FixedPointConfig(scale_bits=20, value_bits=130, intermediate_bits=260)
    big_bounds = derive_bounds([[huge.value_bound]], [[huge.value_bound]], huge, 1, 1)
    with pytest.raises(ProofContractError, match="field-safe"):
        check_bounds_composition(
            big_bounds,
            in_dim=2,
            rank=1,
            fixed_point=huge,
            scaling_num=1,
            scaling_den=1,
        )
