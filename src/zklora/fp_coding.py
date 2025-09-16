import numpy as np

def fixed_point_encode(x, fractional_bits=10):
    """
    Encodes an array (or scalar) of 16-bit floats into fixed-point representation as u32 integers.

    Parameters:
        x: array-like of np.float16 (or convertible to np.float16)
        fractional_bits: Number of bits for the fractional part (default is 10)

    Returns:
        A numpy array of type np.uint32 containing the fixed-point encoded values.
    """
    arr = np.asarray(x, dtype=np.float16)
    multiplier = 2 ** fractional_bits
    fixed_point = np.round(arr.astype(np.float32) * multiplier)
    return fixed_point.astype(np.uint32).tolist()

# Fixed point decoding function for 16-bit floating point numbers from u32 fixed-point representation
def fixed_point_decode(y, fractional_bits=10):
    """
    Decodes an array (or scalar) of u32 fixed-point encoded numbers back into 16-bit floating point numbers.

    Parameters:
        y: array-like of np.uint32 (or convertible to np.uint32)
        fractional_bits: Number of bits for the fractional part (default is 10)

    Returns:
        A numpy array of type np.float16 containing the decoded values.
    """
    arr = np.asarray(y, dtype=np.uint32)
    multiplier = 2 ** fractional_bits
    decoded = arr.astype(np.float32) / multiplier
    return decoded.astype(np.float16).tolist()