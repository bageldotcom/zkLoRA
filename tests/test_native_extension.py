import hashlib
import importlib
import json
import os

import pytest

if os.environ.get("ZKLORA_REQUIRE_NATIVE_EXTENSION") == "1":
    native = importlib.import_module("zklora._native_prover")
else:
    native = pytest.importorskip(
        "zklora._native_prover",
        reason="native PyO3 extension is not built in this environment",
    )


def _canonical_json(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _adapter_payload(b_value=1):
    return {
        "schema_version": 2,
        "in_dim": 1,
        "rank": 1,
        "out_dim": 1,
        "fixed_point": {
            "scale_bits": 0,
            "value_bits": 3,
            "intermediate_bits": 4,
        },
        "scaling_num": 1,
        "scaling_den": 1,
        "a": [[1]],
        "b": [[b_value]],
    }


def test_native_extension_statement_digest_matches_sha256():
    statement = _canonical_json({"delta": [1], "x": [1]})

    assert (
        native.statement_digest(statement)
        == hashlib.sha256(statement.encode("utf-8")).hexdigest()
    )


def test_native_extension_adapter_commitment_is_deterministic():
    payload = _canonical_json(_adapter_payload())

    first = native.adapter_commitment(payload)
    second = native.adapter_commitment(payload)

    assert first == second
    assert int(first) >= 0
    assert first != native.adapter_commitment(_canonical_json(_adapter_payload(2)))
