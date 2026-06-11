import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zklora.proof_contract import (
    FixedPointConfig,
    InvocationWitness,
    ProofContractError,
    TranscriptEntry,
    adapter_manifest_entry,
    compute_delta_quantized,
    flatten,
    load_json,
    quantize_nested,
    statement_from_witness,
    transcript_entry_from_statement,
    verify_artifacts,
    write_invocation_artifacts,
)
import zklora.proof_contract as proof_contract


class FakeNative:
    def adapter_commitment_v4(self, adapter_json, salt):
        return hashlib.sha256(f"{salt}|{adapter_json}".encode()).hexdigest()

    def adapter_setup_v4(self, adapter_json, salt):
        return json.dumps(
            {"core": {"digest": self.adapter_commitment_v4(adapter_json, salt)}}
        )

    def prove_v4(self, statement_json, witness_json, salt):
        return f"{statement_json}|{witness_json}".encode()

    def verify_v4(self, statement_json, proof, setup_json):
        return proof.startswith(statement_json.encode() + b"|")

    def verify_adapter_setup_v4(self, setup_json):
        return True


@pytest.fixture(autouse=True)
def fake_native(monkeypatch):
    monkeypatch.setattr(proof_contract, "_native_module", lambda: FakeNative())


def _valid_witness(session_id="s1", invocation_index=0):
    fp = FixedPointConfig(scale_bits=2, value_bits=16, intermediate_bits=32)
    a = quantize_nested([[2.0, -1.0]], fp)
    b = quantize_nested([[3.0], [-2.0]], fp)
    x = flatten(quantize_nested([1.5, -0.5], fp))
    delta = compute_delta_quantized(
        a,
        b,
        x,
        scaling_num=1,
        scaling_den=2,
        config=fp,
    )
    return InvocationWitness(
        session_id=session_id,
        module_name="transformer.h.0.attn.c_attn",
        invocation_index=invocation_index,
        input_shape=[2],
        output_shape=[2],
        x=x,
        delta=delta,
        a=a,
        b=b,
        scaling_num=1,
        scaling_den=2,
        adapter_metadata={"rank": 1, "alpha": 1},
        fixed_point=fp,
    )


def _manifest_for(witness):
    return [
        adapter_manifest_entry(
            witness.module_name,
            witness.a,
            witness.b,
            witness.scaling_num,
            witness.scaling_den,
            witness.fixed_point,
        )
    ]


def test_import_zklora_without_ezkl_or_onnx_side_effects():
    sys.modules.pop("zklora", None)
    module = importlib.import_module("zklora")
    assert module.__version__
    assert "ezkl" not in sys.modules
    assert "onnx" not in sys.modules
    assert "onnxruntime" not in sys.modules


def test_statement_round_trip_is_canonical_and_transcript_bound(tmp_path):
    witness = _valid_witness()
    paths = write_invocation_artifacts(tmp_path, witness)
    statement = load_json(paths["statement"])
    meta = load_json(paths["meta"])
    transcript_entry = transcript_entry_from_statement(statement)

    assert meta["proof_kind"] == "native-sigma-v4"
    total_time, count = verify_artifacts(
        tmp_path, [transcript_entry], _manifest_for(witness)
    )
    assert total_time >= 0
    assert count == 1

    with open(paths["statement"], "r", encoding="utf-8") as f:
        first = f.read()
    with open(paths["statement"], "w", encoding="utf-8") as f:
        json.dump(statement, f, sort_keys=True, separators=(",", ":"))
        f.write("\n")
    with open(paths["statement"], "r", encoding="utf-8") as f:
        assert f.read() == first


def test_transcript_mismatch_rejects_even_with_valid_artifact(tmp_path):
    witness = _valid_witness()
    paths = write_invocation_artifacts(tmp_path, witness)
    statement = load_json(paths["statement"])
    entry = transcript_entry_from_statement(statement)
    mismatched = TranscriptEntry(
        session_id=entry.session_id,
        module_name=entry.module_name,
        invocation_index=entry.invocation_index,
        input_shape=entry.input_shape,
        output_shape=entry.output_shape,
        x=[entry.x[0] + 1, *entry.x[1:]],
        delta=entry.delta,
        fixed_point=entry.fixed_point,
        scaling_num=entry.scaling_num,
        scaling_den=entry.scaling_den,
    )

    with pytest.raises(ProofContractError, match="does not match verifier transcript"):
        verify_artifacts(tmp_path, [mismatched], _manifest_for(witness))


def test_expected_adapter_manifest_mismatch_rejects(tmp_path):
    witness = _valid_witness()
    paths = write_invocation_artifacts(tmp_path, witness)
    statement = load_json(paths["statement"])
    transcript = [transcript_entry_from_statement(statement)]
    manifest = _manifest_for(witness)
    manifest[0]["adapter_commitment"]["value"] = "0" * 64

    with pytest.raises(ProofContractError, match="expected adapter manifest"):
        verify_artifacts(tmp_path, transcript, manifest)


def test_tampered_vk_and_proof_reject(tmp_path):
    witness = _valid_witness()
    paths = write_invocation_artifacts(tmp_path, witness)
    statement = load_json(paths["statement"])
    transcript = [transcript_entry_from_statement(statement)]

    vk = load_json(paths["vk"])
    vk["vk_fingerprint"] = "bad"
    Path(paths["vk"]).write_text(
        json.dumps(vk, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ProofContractError, match="fingerprint"):
        verify_artifacts(tmp_path, transcript, _manifest_for(witness))

    write_invocation_artifacts(tmp_path, witness)
    Path(paths["proof"]).write_bytes(b"tampered")
    with pytest.raises(ProofContractError, match="proof bytes"):
        verify_artifacts(tmp_path, transcript, _manifest_for(witness))

    paths = write_invocation_artifacts(tmp_path, witness)
    statement = load_json(paths["statement"])
    statement["adapter_commitment"]["value"] = "0" * 64
    statement["statement_digest"] = "00" * 32
    Path(paths["statement"]).write_text(
        json.dumps(statement, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    transcript = [transcript_entry_from_statement(statement)]
    with pytest.raises(ProofContractError, match="statement digest"):
        verify_artifacts(tmp_path, transcript, _manifest_for(witness))


def test_statement_binding_tampering_rejects(tmp_path):
    witness = _valid_witness()
    paths = write_invocation_artifacts(tmp_path, witness)
    statement = load_json(paths["statement"])

    statement["fixed_point"]["scale_bits"] += 1
    Path(paths["statement"]).write_text(
        json.dumps(statement, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    transcript = [transcript_entry_from_statement(statement)]
    with pytest.raises(ProofContractError, match="statement digest"):
        verify_artifacts(tmp_path, transcript, _manifest_for(witness))

    paths = write_invocation_artifacts(tmp_path, witness)
    statement = load_json(paths["statement"])
    statement["scaling"]["den"] = 3
    Path(paths["statement"]).write_text(
        json.dumps(statement, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    transcript = [transcript_entry_from_statement(statement)]
    with pytest.raises(ProofContractError, match="statement digest"):
        verify_artifacts(tmp_path, transcript, _manifest_for(witness))


def test_missing_or_tampered_invocation_rejects_whole_session(tmp_path):
    first = _valid_witness(invocation_index=0)
    second = _valid_witness(invocation_index=1)
    first_paths = write_invocation_artifacts(tmp_path, first)
    second_paths = write_invocation_artifacts(tmp_path, second)
    transcript = [
        transcript_entry_from_statement(load_json(first_paths["statement"])),
        transcript_entry_from_statement(load_json(second_paths["statement"])),
    ]

    manifest = _manifest_for(first)
    verify_artifacts(tmp_path, transcript, manifest)

    Path(second_paths["statement"]).unlink()
    with pytest.raises(ProofContractError, match="coverage mismatch"):
        verify_artifacts(tmp_path, transcript, manifest)


def test_delta_relation_and_alpha_over_rank_scaling():
    witness = _valid_witness()
    statement = statement_from_witness(witness)
    assert statement["scaling"] == {"num": 1, "den": 2}

    bad = InvocationWitness(
        **{
            **witness.__dict__,
            "delta": [witness.delta[0] + 1, witness.delta[1]],
        }
    )
    with pytest.raises(ProofContractError, match="does not match"):
        statement_from_witness(bad)
