use ark_bn254::Fr;
use ark_ff::PrimeField;

/// NF4 quantization lookup table
/// 
/// These are the 16 discrete values that NF4 quantization uses.
/// They're specifically chosen to minimize quantization error for
/// neural network weights that follow a normal distribution.
pub struct NF4Field;

impl NF4Field {
    pub const NF4_VALUES: [f32; 16] = [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ];

    /// Convert an NF4 index (0-15) to a field element
    /// 
    /// This is where we bridge the gap between AI and crypto.
    /// We take a 4-bit quantized weight and turn it into something
    /// we can do ZK proofs on.
    pub fn to_field_element(nf4_index: u8) -> Fr {
        assert!(nf4_index < 16, "NF4 index must be 0-15");
        
        // We scale by 1M to avoid floating point weirdness
        let scaled = (Self::NF4_VALUES[nf4_index as usize] * 1_000_000.0) as i64;
        
        if scaled >= 0 {
            Fr::from(scaled as u64)
        } else {
            -Fr::from((-scaled) as u64)
        }
    }

    /// Go backwards - field element to NF4 index
    /// Mainly useful for testing
    pub fn from_field_element(fe: Fr) -> Option<u8> {
        for (idx, _val) in Self::NF4_VALUES.iter().enumerate() {
            let expected = Self::to_field_element(idx as u8);
            if fe == expected {
                return Some(idx as u8);
            }
        }
        None
    }

    pub const SCALE_FACTOR: i64 = 1_000_000;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        for i in 0..16 {
            let fe = NF4Field::to_field_element(i);
            let back = NF4Field::from_field_element(fe);
            assert_eq!(back, Some(i), "Roundtrip failed for index {}", i);
        }
    }

    #[test]
    #[should_panic]
    fn test_invalid_index() {
        NF4Field::to_field_element(16); // Should panic
    }
}
