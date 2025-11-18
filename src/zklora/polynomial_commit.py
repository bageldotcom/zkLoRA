import json
import os
from typing import List, Union

from blake3 import blake3  # type: ignore

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
    with open(activations_path, "r") as f:
        data = json.load(f)

    # Flatten arbitrarily nested lists using numpy when available for speed
    try:
        import numpy as np  # local import to avoid hard dependency

        flat_vals = (
            np.asarray(data["input_data"], dtype=np.float64).reshape(-1).tolist()
        )
    except Exception:
        # fallback: naïve Python flatten
        def _flatten(x):
            for y in x:
                if isinstance(y, (list, tuple)):
                    yield from _flatten(y)
                else:
                    yield y

        flat_vals = list(_flatten(data["input_data"]))

    # Generate random nonce for hiding property
    nonce = os.urandom(32)

    # Compute Merkle root with nonce
    root = _merkle_root(flat_vals, nonce)

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

    # Load and flatten activations
    with open(activations_path, "r") as f:
        data = json.load(f)

    try:
        import numpy as np

        flat_vals = (
            np.asarray(data["input_data"], dtype=np.float64).reshape(-1).tolist()
        )
    except Exception:

        def _flatten(x):
            for y in x:
                if isinstance(y, (list, tuple)):
                    yield from _flatten(y)
                else:
                    yield y

        flat_vals = list(_flatten(data["input_data"]))

    # Recompute root with provided nonce
    computed_root = _merkle_root(flat_vals, nonce)

    # Compare roots
    return computed_root == expected_root
