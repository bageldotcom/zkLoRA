import json
from pathlib import Path
import sys
from pathlib import Path as _P

import pytest

np = pytest.importorskip("numpy")

# Direct import of polynomial_commit module
sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "src" / "zklora"))
import polynomial_commit  # noqa: E402

commit_activations = polynomial_commit.commit_activations
verify_commitment = polynomial_commit.verify_commitment


def test_commit_roundtrip(tmp_path: Path):
    data = {"input_data": [1, 2, 3, 4]}
    path = tmp_path / "acts.json"
    with open(path, "w") as f:
        json.dump(data, f)

    commit = commit_activations(str(path))
    assert verify_commitment(str(path), commit)


def test_empty_data(tmp_path: Path):
    """Test commitment of empty data."""
    data = {"input_data": []}
    path = tmp_path / "empty.json"
    with open(path, "w") as f:
        json.dump(data, f)

    commit = commit_activations(str(path))
    # Parse JSON commitment
    commit_data = json.loads(commit)
    assert "root" in commit_data
    assert "nonce" in commit_data
    assert commit_data["root"].startswith("0x")
    assert commit_data["nonce"].startswith("0x")
    assert len(commit_data["root"]) == 66  # "0x" + 64 hex chars
    assert len(commit_data["nonce"]) == 66  # "0x" + 64 hex chars
    assert verify_commitment(str(path), commit)


def test_single_element(tmp_path: Path):
    """Test commitment of single element."""
    data = {"input_data": [42]}
    path = tmp_path / "single.json"
    with open(path, "w") as f:
        json.dump(data, f)

    commit = commit_activations(str(path))
    assert verify_commitment(str(path), commit)


def test_float_values(tmp_path: Path):
    """Test commitment with floating point values."""
    data = {"input_data": [1.5, 2.7, -3.14, 0.0]}
    path = tmp_path / "floats.json"
    with open(path, "w") as f:
        json.dump(data, f)

    commit = commit_activations(str(path))
    assert verify_commitment(str(path), commit)


def test_nested_lists(tmp_path: Path):
    """Test commitment with nested lists (should be flattened)."""
    data = {"input_data": [[1, 2], [3, [4, 5]], 6]}
    path = tmp_path / "nested.json"
    with open(path, "w") as f:
        json.dump(data, f)

    commit = commit_activations(str(path))
    assert verify_commitment(str(path), commit)

    # Verify that nested lists with same flattened values but different nonces produce different roots
    flat_data = {"input_data": [1, 2, 3, 4, 5, 6]}
    flat_path = tmp_path / "flat.json"
    with open(flat_path, "w") as f:
        json.dump(flat_data, f)

    flat_commit = commit_activations(str(flat_path))

    # Parse both commitments
    nested_data = json.loads(commit)
    flat_commit_data = json.loads(flat_commit)

    # Different nonces should produce different roots even with same data
    assert nested_data["nonce"] != flat_commit_data["nonce"]
    assert nested_data["root"] != flat_commit_data["root"]


def test_deterministic_with_nonce(tmp_path: Path):
    """Test that commitments with same nonce are deterministic."""
    data = {"input_data": [1, 2, 3, 4, 5]}
    path = tmp_path / "determ.json"
    with open(path, "w") as f:
        json.dump(data, f)

    # Generate commitment
    commit = commit_activations(str(path))

    # Multiple verifications with same commitment should work
    for _ in range(5):
        assert verify_commitment(str(path), commit)


def test_case_insensitive_verification(tmp_path: Path):
    """Test that verification handles hex case variations."""
    data = {"input_data": [1, 2, 3]}
    path = tmp_path / "case.json"
    with open(path, "w") as f:
        json.dump(data, f)

    commit = commit_activations(str(path))
    commit_data = json.loads(commit)

    # Test various case combinations in the JSON
    lower_commit = json.dumps(
        {"root": commit_data["root"].lower(), "nonce": commit_data["nonce"].lower()}
    )
    upper_commit = json.dumps(
        {"root": commit_data["root"].upper(), "nonce": commit_data["nonce"].upper()}
    )

    assert verify_commitment(str(path), commit)
    assert verify_commitment(str(path), lower_commit)
    assert verify_commitment(str(path), upper_commit)


def test_different_data_same_length(tmp_path: Path):
    """Test that different data produces different commitments."""
    data1 = {"input_data": [1, 2, 3, 4]}
    path1 = tmp_path / "data1.json"
    with open(path1, "w") as f:
        json.dump(data1, f)

    data2 = {"input_data": [4, 3, 2, 1]}
    path2 = tmp_path / "data2.json"
    with open(path2, "w") as f:
        json.dump(data2, f)

    commit1 = commit_activations(str(path1))
    commit2 = commit_activations(str(path2))

    # Parse commitments
    commit1_data = json.loads(commit1)
    commit2_data = json.loads(commit2)

    # Different data should produce different roots (even if nonces happen to be the same)
    # But nonces should be different too (randomly generated)
    assert commit1_data["nonce"] != commit2_data["nonce"]
    assert commit1_data["root"] != commit2_data["root"]

    assert verify_commitment(str(path1), commit1)
    assert verify_commitment(str(path2), commit2)
    assert not verify_commitment(str(path1), commit2)
    assert not verify_commitment(str(path2), commit1)


def test_large_dataset(tmp_path: Path):
    """Test commitment with larger dataset."""
    # Create dataset with 1000 random values
    np.random.seed(42)
    values = np.random.randn(1000).tolist()
    data = {"input_data": values}
    path = tmp_path / "large.json"
    with open(path, "w") as f:
        json.dump(data, f)

    commit = commit_activations(str(path))
    assert verify_commitment(str(path), commit)


def test_power_of_two_padding(tmp_path: Path):
    """Test datasets that require padding to power of 2."""
    # Test various sizes that will require padding
    for size in [3, 5, 7, 9, 15, 17, 31, 33]:
        data = {"input_data": list(range(size))}
        path = tmp_path / f"pad_{size}.json"
        with open(path, "w") as f:
            json.dump(data, f)

        commit = commit_activations(str(path))
        assert verify_commitment(str(path), commit)


def test_numpy_array_input(tmp_path: Path):
    """Test commitment with numpy arrays (common in ML contexts)."""
    # Create numpy array and convert to list for JSON
    arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
    data = {"input_data": arr.tolist()}
    path = tmp_path / "numpy.json"
    with open(path, "w") as f:
        json.dump(data, f)

    commit = commit_activations(str(path))
    assert verify_commitment(str(path), commit)


def test_mixed_types(tmp_path: Path):
    """Test commitment with mixed int and float types."""
    data = {"input_data": [1, 2.5, -3, 0.0, 42]}
    path = tmp_path / "mixed.json"
    with open(path, "w") as f:
        json.dump(data, f)

    commit = commit_activations(str(path))
    assert verify_commitment(str(path), commit)


def test_commitment_format(tmp_path: Path):
    """Test the structure of the commitment."""
    data = {"input_data": [1.0, 2.0, 3.0, 4.0]}
    path = tmp_path / "format.json"
    with open(path, "w") as f:
        json.dump(data, f)

    commit = commit_activations(str(path))

    # Parse and validate structure
    commit_data = json.loads(commit)
    assert isinstance(commit_data, dict)
    assert "root" in commit_data
    assert "nonce" in commit_data
    assert commit_data["root"].startswith("0x")
    assert commit_data["nonce"].startswith("0x")
    assert len(commit_data["root"]) == 66  # "0x" + 64 hex chars
    assert len(commit_data["nonce"]) == 66  # "0x" + 64 hex chars

    # Verify it's valid hex
    try:
        int(commit_data["root"], 16)
        int(commit_data["nonce"], 16)
    except ValueError:
        pytest.fail("Commitment contains invalid hex strings")

    assert verify_commitment(str(path), commit)


def test_hiding_property(tmp_path: Path):
    """Test that the commitment scheme has hiding property."""
    data = {"input_data": [1, 2, 3, 4, 5]}
    path = tmp_path / "hiding.json"
    with open(path, "w") as f:
        json.dump(data, f)

    # Generate multiple commitments for the same data
    commitments = []
    for _ in range(10):
        commit = commit_activations(str(path))
        commitments.append(json.loads(commit))

    # All nonces should be different (with overwhelming probability)
    nonces = [c["nonce"] for c in commitments]
    assert len(set(nonces)) == len(nonces), "Nonces should be unique"

    # All roots should be different (due to different nonces)
    roots = [c["root"] for c in commitments]
    assert len(set(roots)) == len(roots), "Roots should be unique with different nonces"

    # But all commitments should verify correctly
    for commit in commitments:
        assert verify_commitment(str(path), json.dumps(commit))


def test_invalid_commitment_format(tmp_path: Path):
    """Test that invalid commitment formats are rejected."""
    data = {"input_data": [1, 2, 3]}
    path = tmp_path / "invalid.json"
    with open(path, "w") as f:
        json.dump(data, f)

    # Test various invalid formats
    assert not verify_commitment(str(path), "not json")
    assert not verify_commitment(str(path), "{}")
    assert not verify_commitment(str(path), '{"root": "0x123"}')  # missing nonce
    assert not verify_commitment(str(path), '{"nonce": "0x123"}')  # missing root
    assert not verify_commitment(
        str(path), '{"root": "invalid", "nonce": "0x123"}'
    )  # invalid hex
