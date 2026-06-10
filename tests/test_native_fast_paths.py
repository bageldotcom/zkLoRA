"""Parity tests: native fast paths must be byte/value-identical to Python."""

import importlib
import os
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zklora.proof_contract import (  # noqa: E402
    FixedPointConfig,
    ProofContractError,
    _compute_delta_quantized_python,
    compute_delta_quantized,
)

if os.environ.get("ZKLORA_REQUIRE_NATIVE_EXTENSION") == "1":
    native = importlib.import_module("zklora._native_prover")
else:
    native = pytest.importorskip(
        "zklora._native_prover",
        reason="native PyO3 extension is not built in this environment",
    )


def _random_case(rng, in_dim, rank, out_dim, magnitude):
    a = [[rng.randint(-magnitude, magnitude) for _ in range(in_dim)] for _ in range(rank)]
    b = [[rng.randint(-magnitude, magnitude) for _ in range(rank)] for _ in range(out_dim)]
    x = [rng.randint(-magnitude, magnitude) for _ in range(in_dim)]
    return a, b, x


def test_native_delta_matches_python_reference_across_shapes():
    rng = random.Random(1234)
    config = FixedPointConfig(scale_bits=20, value_bits=63, intermediate_bits=127)
    for in_dim, rank, out_dim in [(1, 1, 1), (3, 2, 5), (16, 4, 8), (64, 2, 32)]:
        for scaling_num, scaling_den in [(1, 1), (3, 2), (-7, 4), (16, 16)]:
            a, b, x = _random_case(rng, in_dim, rank, out_dim, 1 << 21)
            expected = _compute_delta_quantized_python(
                a, b, x, scaling_num, scaling_den, config
            )
            actual = compute_delta_quantized(a, b, x, scaling_num, scaling_den, config)
            assert actual == expected


def test_native_delta_matches_python_on_rounding_ties():
    # scale 2 with odd raw sums exercises the canonical half-up rounding on
    # both positive and negative values.
    config = FixedPointConfig(scale_bits=1, value_bits=16, intermediate_bits=32)
    for raw in range(-9, 10):
        a = [[1]]
        b = [[1]]
        x = [raw]
        expected = _compute_delta_quantized_python(a, b, x, 1, 1, config)
        actual = compute_delta_quantized(a, b, x, 1, 1, config)
        assert actual == expected, f"mismatch for raw={raw}"


def test_native_delta_bound_violations_raise_contract_errors():
    config = FixedPointConfig(scale_bits=2, value_bits=8, intermediate_bits=16)
    too_big = config.value_bound + 1
    with pytest.raises(ProofContractError, match="exceeds signed bound"):
        compute_delta_quantized([[too_big]], [[1]], [1], 1, 1, config)
    with pytest.raises(ProofContractError, match="scaling_den must be positive"):
        compute_delta_quantized([[1]], [[1]], [1], 1, 0, config)


def test_native_merkle_root_matches_python_reference():
    import zklora.polynomial_commit as pc

    rng = random.Random(99)
    for count in [1, 2, 3, 4, 5, 8, 13, 64, 100]:
        values = [rng.uniform(-1e6, 1e6) for _ in range(count)] + [0.0, -0.0]
        nonce = bytes(rng.randrange(256) for _ in range(32))
        leaves = [pc._hash_leaf(v, nonce) for v in values]
        if len(leaves) % 2 == 1:
            leaves.append(pc.LEAF_EMPTY)
        level = leaves
        while len(level) > 1:
            nxt = [
                pc._parent_hash(level[i], level[i + 1]) for i in range(0, len(level), 2)
            ]
            if len(nxt) % 2 == 1 and len(nxt) != 1:
                nxt.append(pc.LEAF_EMPTY)
            level = nxt
        expected = level[0]
        actual = bytes(native.merkle_root([float(v) for v in values], nonce))
        assert actual == expected

    assert pc._merkle_root([], b"\x00" * 32) == pc.LEAF_EMPTY


def test_commit_and_verify_round_trip_uses_native_path(tmp_path):
    import json

    from zklora.polynomial_commit import commit_activations, verify_commitment

    payload = {"input_data": [[1.5, -2.25], [3.125, 4.0], [5.5, -6.75]]}
    path = tmp_path / "acts.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    commitment = commit_activations(str(path))
    assert verify_commitment(str(path), commitment)

    tampered = {"input_data": [[1.5, -2.25], [3.125, 4.0], [5.5, -6.7501]]}
    path.write_text(json.dumps(tampered), encoding="utf-8")
    assert not verify_commitment(str(path), commitment)
