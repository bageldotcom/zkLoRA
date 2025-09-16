import numpy as np

def fixed_point_encode(x, fractional_bits: int = 10, total_bits: int = 32):
    """Encode floating-point values to *signed* fixed-point two's complement integers.

    Parameters
    ----------
    x : array-like or scalar
        Values convertible to ``np.float16``.
    fractional_bits : int, default ``10``
        Number of bits reserved for the fractional part of the fixed-point number.
    total_bits : int, default ``32``
        Total bit-width of the resulting fixed-point integer. ``total_bits`` must be
        large enough to accommodate the integer, fractional **and** sign bit. 32 is
        a sensible default because we return ``np.uint32`` values.

    Returns
    -------
    list[int]
        Two's-complement encoded unsigned integers (``np.uint32``) representing the
        input values at the requested precision.
    """

    # Convert to higher precision float before scaling to avoid precision loss.
    arr = np.asarray(x, dtype=np.float32)
    multiplier = 2 ** fractional_bits

    # Scale and round towards nearest integer.
    fixed = np.round(arr * multiplier).astype(np.int64)  # temporary wider type

    # Check that the value fits in the desired bit-width (including sign bit).
    max_val = (1 << (total_bits - 1)) - 1
    min_val = -1 << (total_bits - 1)
    if np.any(fixed > max_val) or np.any(fixed < min_val):
        raise OverflowError(
            f"Value out of range for {total_bits}-bit fixed-point representation"
        )

    # Convert to two's complement by masking – this yields an *unsigned* view of
    # exactly the same bits.
    fixed_twos = (fixed & ((1 << total_bits) - 1)).astype(np.uint32)

    return fixed_twos.tolist()

# Fixed point decoding function for 16-bit floating point numbers from u32 fixed-point representation
def fixed_point_decode(y, fractional_bits: int = 10, total_bits: int = 32):
    """Decode signed fixed-point two's complement integers back to floats.

    Parameters
    ----------
    y : array-like or scalar
        Unsigned integers (``np.uint32`` or compatible) storing two's-complement
        fixed-point values produced by :func:`fixed_point_encode`.
    fractional_bits : int, default ``10``
        Number of fractional bits that were used during encoding.
    total_bits : int, default ``32``
        Total width of the stored integers. Must match the value passed to
        :func:`fixed_point_encode`.

    Returns
    -------
    list[np.float16]
        The decoded floating-point values.
    """

    arr_unsigned = np.asarray(y, dtype=np.uint32)

    # Re-interpret the unsigned integers as signed two's complement.
    signed = arr_unsigned.view(np.int32) if total_bits == 32 else arr_unsigned.astype(np.int64)

    multiplier = 2 ** fractional_bits
    decoded = signed.astype(np.float32) / multiplier
    return decoded.astype(np.float16).tolist()

if __name__ == "__main__":
    # Basic sanity check for the fixed-point codec.
    test_values = np.array([-3.25, -1.5, -0.125, 0.0, 0.125, 1.5, 3.25], dtype=np.float32)

    encoded = fixed_point_encode(test_values, fractional_bits=10)
    decoded = np.array(fixed_point_decode(encoded, fractional_bits=10), dtype=np.float32)

    # Expect equality within one LSB of the fractional part.
    assert np.allclose(test_values, decoded, atol=1 / (2 ** 10)), (
        f"Round-trip mismatch:\noriginal: {test_values}\ndecoded : {decoded}"
    )

    print("[fp_coding] Self-test passed for signed two's-complement encoding/decoding.")
    