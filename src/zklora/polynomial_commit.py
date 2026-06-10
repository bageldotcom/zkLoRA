import json
import os
from typing import List, Union

from blake3 import blake3  # type: ignore

try:  # native fast path; byte-identical to the Python implementation below
    from zklora import _native_prover as _native_merkle_module
except ImportError:  # pragma: no cover - extension not built in this env
    _native_merkle_module = None

# Merkle-based vector commitment parameters
LEAF_EMPTY = b"\x00" * 32  # same as EMPTY_HASH in Rust implementation


def _hash_leaf(value: Union[int, float], nonce: bytes) -> bytes:
    """Hash a single scalar value with a nonce for hiding property.

    The value is first serialized as 8-byte big-endian (matching Rust f64::to_be_bytes),
    then concatenated with the nonce before hashing.
    """
    import struct

    if isinstance(value, float):
        byte_repr = struct.pack(">d", value)  # '>d' = big-endian double (f64)
    else:
        # Treat ints as floats to match Rust f64 representation
        byte_repr = struct.pack(">d", float(value))

    # Concatenate value bytes with nonce for hiding
    return blake3(byte_repr + nonce).digest()


def _parent_hash(left: bytes, right: bytes) -> bytes:
    """Aggregate two children into their parent node (binary tree)."""
    return blake3(left + right).digest()


def _merkle_root(values: List[Union[int, float]], nonce: bytes) -> bytes:
    """Compute Merkle root for a list of scalar values with hiding.

    The tree is padded on the right with EMPTY leaves in order to guarantee that
    every internal node always has exactly two children, matching the behaviour
    of dusk-merkle with `Tree::<Item, H, A>::new()` where missing sub-trees are
    equal to the constant `EMPTY_SUBTREE` (32 zero bytes).

    Args:
        values: List of numeric values to commit to
        nonce: Random bytes for hiding property
    """
    if not values:
        return LEAF_EMPTY

    if _native_merkle_module is not None and hasattr(
        _native_merkle_module, "merkle_root"
    ):
        try:
            return bytes(
                _native_merkle_module.merkle_root(
                    [float(v) for v in values], bytes(nonce)
                )
            )
        except (OverflowError, TypeError, ValueError):
            pass  # fall back to the pure-Python implementation

    # Convert to leaf hashes with nonce
    level: List[bytes] = [_hash_leaf(v, nonce) for v in values]

    # Pad to even length with EMPTY leaves
    if len(level) % 2 == 1:
        level.append(LEAF_EMPTY)

    # Build tree bottom-up until we get the root
    while len(level) > 1:
        next_level: List[bytes] = []
        for i in range(0, len(level), 2):
            left, right = level[i], level[i + 1]
            next_level.append(_parent_hash(left, right))
        if len(next_level) % 2 == 1 and len(next_level) != 1:
            next_level.append(LEAF_EMPTY)
        level = next_level
    return level[0]


def _flatten_input_data(data) -> List[Union[int, float]]:
    """Flatten ``data["input_data"]`` to a flat list of scalars.

    Uses numpy for rectangular data and a recursive fallback for ragged data,
    preserving depth-first leaf order in both cases.
    """
    try:
        import numpy as np  # local import to avoid hard dependency

        return np.asarray(data["input_data"], dtype=np.float64).reshape(-1).tolist()
    except Exception:

        def _flatten(x):
            for y in x:
                if isinstance(y, (list, tuple)):
                    yield from _flatten(y)
                else:
                    yield y

        return list(_flatten(data["input_data"]))


def _merkle_root_from_file(activations_path: str, nonce: bytes) -> bytes:
    """Compute the hiding Merkle root of an activations file.

    JSON parsing stays in Python on purpose: it defines the exact float
    semantics of the commitment (the nearest-f64 produced by ``json``/``float``),
    and a native JSON parser does not always round identically. The leaf and
    tree hashing then runs through ``_merkle_root``, which uses the native
    BLAKE3 fast path when available and is byte-identical to the pure-Python
    reference.
    """
    with open(activations_path, "r") as f:
        data = json.load(f)
    return _merkle_root(_flatten_input_data(data), nonce)


# --------------------------------------------------------------------------------------
# Public API (names preserved for backwards compatibility)
# --------------------------------------------------------------------------------------


def commit_activations(activations_path: str) -> str:
    """Return hiding Merkle commitment of activations stored in JSON file.

    The JSON is expected to contain a key `input_data` pointing to a list
    of numeric scalars. The commitment includes a random nonce for hiding.

    Returns:
        JSON string containing both the Merkle root and nonce:
        {"root": "0x...", "nonce": "0x..."}
    """
    # Generate random nonce for hiding property
    nonce = os.urandom(32)

    # Compute Merkle root with nonce (native single-pass when available)
    root = _merkle_root_from_file(activations_path, nonce)

    # Return JSON with both root and nonce
    commitment_data = {"root": "0x" + root.hex(), "nonce": "0x" + nonce.hex()}
    return json.dumps(commitment_data)


def verify_commitment(activations_path: str, commitment: str) -> bool:
    """Verify a hiding Merkle commitment against activations.

    Args:
        activations_path: Path to JSON file with activations
        commitment: JSON string containing root and nonce

    Returns:
        True if commitment is valid, False otherwise
    """
    try:
        # Parse commitment JSON
        commitment_data = json.loads(commitment)
        root_hex = commitment_data["root"]
        nonce_hex = commitment_data["nonce"]

        # Remove "0x" or "0X" prefix if present (case insensitive)
        if root_hex.lower().startswith("0x"):
            root_hex = root_hex[2:]
        if nonce_hex.lower().startswith("0x"):
            nonce_hex = nonce_hex[2:]

        # Convert hex to bytes
        expected_root = bytes.fromhex(root_hex)
        nonce = bytes.fromhex(nonce_hex)

    except (json.JSONDecodeError, KeyError, ValueError):
        # Invalid commitment format
        return False

    # Recompute root with provided nonce (native single-pass when available)
    computed_root = _merkle_root_from_file(activations_path, nonce)

    # Compare roots
    return computed_root == expected_root
