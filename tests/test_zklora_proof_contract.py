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
    compute_delta_quantized,
    flatten,
    load_json,
    quantize_nested,
    statement_from_witness,
    transcript_entry_from_statement,
    verify_artifacts,
    write_invocation_artifacts,
)


def _valid_witness(session_id="s1", invocation_index=0):
    fp = FixedPointConfig(scale_bits=20)
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

    assert meta["proof_kind"] == "native-halo2"
    total_time, count = verify_artifacts(tmp_path, [transcript_entry])
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
        verify_artifacts(tmp_path, [mismatched])


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
        verify_artifacts(tmp_path, transcript)

    write_invocation_artifacts(tmp_path, witness)
    Path(paths["proof"]).write_bytes(b"tampered")
    with pytest.raises(ProofContractError, match="proof bytes"):
        verify_artifacts(tmp_path, transcript)

    paths = write_invocation_artifacts(tmp_path, witness)
    statement = load_json(paths["statement"])
    statement["native_lora_commitment"]["value"] += 1
    Path(paths["statement"]).write_text(
        json.dumps(statement, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    transcript = [transcript_entry_from_statement(statement)]
    with pytest.raises(ProofContractError, match="proof bytes"):
        verify_artifacts(tmp_path, transcript)


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
    with pytest.raises(ProofContractError, match="circuit_id"):
        verify_artifacts(tmp_path, transcript)

    paths = write_invocation_artifacts(tmp_path, witness)
    statement = load_json(paths["statement"])
    statement["scaling"]["den"] = 3
    Path(paths["statement"]).write_text(
        json.dumps(statement, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    transcript = [transcript_entry_from_statement(statement)]
    with pytest.raises(ProofContractError, match="proof bytes"):
        verify_artifacts(tmp_path, transcript)


def test_missing_or_tampered_invocation_rejects_whole_session(tmp_path):
    first = _valid_witness(invocation_index=0)
    second = _valid_witness(invocation_index=1)
    first_paths = write_invocation_artifacts(tmp_path, first)
    second_paths = write_invocation_artifacts(tmp_path, second)
    transcript = [
        transcript_entry_from_statement(load_json(first_paths["statement"])),
        transcript_entry_from_statement(load_json(second_paths["statement"])),
    ]

    verify_artifacts(tmp_path, transcript)

    Path(second_paths["statement"]).unlink()
    with pytest.raises(ProofContractError, match="coverage mismatch"):
        verify_artifacts(tmp_path, transcript)


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
