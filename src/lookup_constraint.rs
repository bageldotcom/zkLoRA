use ark_bn254::Fr;
use ark_ff::Field;
use ark_r1cs_std::fields::fp::FpVar;
use ark_r1cs_std::prelude::*;
use ark_relations::r1cs::{ConstraintSystemRef, SynthesisError};

/// The main circuit for proving NF4 lookups
/// 
/// This circuit proves two things:
/// 1. The index is a valid 4-bit value (0-15)
/// 2. The value correctly corresponds to table[index]
pub struct NF4LookupCircuit {
    /// The 4-bit index we're looking up (private)
    pub index: Option<u8>,
    
    /// The value at that index (private)
    pub value: Option<Fr>,
    
    /// The table commitment (public)
    pub table_commitment: Fr,
}

impl NF4LookupCircuit {
    pub fn new(index: Option<u8>, value: Option<Fr>, table_commitment: Fr) -> Self {
        Self {
            index,
            value,
            table_commitment,
        }
    }

    /// Generate all the constraints for this circuit
    pub fn generate_constraints(
        &self,
        cs: ConstraintSystemRef<Fr>,
    ) -> Result<(), SynthesisError> {
        // Allocate the index as a private witness
        let index_var = FpVar::new_witness(cs.clone(), || {
            self.index
                .map(|i| Fr::from(i as u64))
                .ok_or(SynthesisError::AssignmentMissing)
        })?;

        // Allocate the value as a private witness
        let value_var = FpVar::new_witness(cs.clone(), || {
            self.value.ok_or(SynthesisError::AssignmentMissing)
        })?;

        // Allocate the table commitment as a public input
        let _commitment_var = FpVar::new_input(cs.clone(), || Ok(self.table_commitment))?;

        // Constraint 1: Index must be in range [0, 15]
        self.enforce_4bit_range(cs.clone(), &index_var)?;

        // Constraint 2: Value must equal table[index]
        self.enforce_lookup(cs.clone(), &index_var, &value_var)?;

        Ok(())
    }

    /// Make sure the index is actually 4 bits (0-15)
    fn enforce_4bit_range(
        &self,
        _cs: ConstraintSystemRef<Fr>,
        index: &FpVar<Fr>,
    ) -> Result<(), SynthesisError> {
        // Convert to bits and enforce only 4 bits are used
        let bits = index.to_bits_le()?;
        
        // All bits after the 4th must be zero
        for bit in bits.iter().skip(4) {
            bit.enforce_equal(&Boolean::FALSE)?;
        }

        // Also enforce that index <= 15 directly
        let fifteen = FpVar::constant(Fr::from(15u64));
        let diff = fifteen - index;
        
        // In production, we'd use a proper range proof gadget here
        // For now, we rely on the bit decomposition
        let _ = diff; // Silence unused warning

        Ok(())
    }

    /// Enforce that value = table[index]
    /// 
    /// This uses Lagrange interpolation. For each possible index i,
    /// we compute whether index == i, and if so, add table[i] to the result.
    fn enforce_lookup(
        &self,
        _cs: ConstraintSystemRef<Fr>,
        index: &FpVar<Fr>,
        value: &FpVar<Fr>,
    ) -> Result<(), SynthesisError> {
        use crate::nf4_field::NF4Field;

        let mut result = FpVar::zero();

        // For each possible index value
        for i in 0u8..16 {
            // Get the expected value from the NF4 table
            let table_value = FpVar::constant(NF4Field::to_field_element(i));
            
            // Check if index equals this i
            let i_const = FpVar::constant(Fr::from(i as u64));
            let is_equal = index.is_eq(&i_const)?;
            
            // If index == i, add table[i] to result
            // This is the Lagrange interpolation magic
            let contribution = is_equal.select(&table_value, &FpVar::zero())?;
            result += contribution;
        }

        // Enforce that the claimed value equals what we computed
        value.enforce_equal(&result)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_relations::r1cs::ConstraintSystem;
    use crate::nf4_field::NF4Field;
    use crate::commitment::NF4TableCommitment;

    #[test]
    fn test_valid_lookup() {
        let table: Vec<Fr> = (0..16)
            .map(|i| NF4Field::to_field_element(i))
            .collect();
        
        let commitment = NF4TableCommitment::new(table);
        
        let test_index = 7u8;
        let test_value = NF4Field::to_field_element(test_index);
        
        let circuit = NF4LookupCircuit::new(
            Some(test_index),
            Some(test_value),
            commitment.root(),
        );
        
        let cs = ConstraintSystem::<Fr>::new_ref();
        circuit.generate_constraints(cs.clone()).unwrap();
        
        assert!(cs.is_satisfied().unwrap(), "Valid lookup should satisfy constraints");
    }

    #[test]
    fn test_invalid_value_fails() {
        let table: Vec<Fr> = (0..16)
            .map(|i| NF4Field::to_field_element(i))
            .collect();
        
        let commitment = NF4TableCommitment::new(table);
        
        // Try to claim wrong value for index
        let circuit = NF4LookupCircuit::new(
            Some(7u8),
            Some(NF4Field::to_field_element(8)), // Wrong!
            commitment.root(),
        );
        
        let cs = ConstraintSystem::<Fr>::new_ref();
        circuit.generate_constraints(cs.clone()).unwrap();
        
        assert!(!cs.is_satisfied().unwrap(), "Invalid lookup should fail");
    }

    #[test]
    fn test_out_of_range_index() {
        let table: Vec<Fr> = (0..16)
            .map(|i| NF4Field::to_field_element(i))
            .collect();
        
        let commitment = NF4TableCommitment::new(table);
        
        // Try index 16 (out of range)
        let circuit = NF4LookupCircuit::new(
            Some(16u8),
            Some(Fr::from(0u64)),
            commitment.root(),
        );
        
        let cs = ConstraintSystem::<Fr>::new_ref();
        
        // This should either fail during constraint generation or not satisfy
        let result = circuit.generate_constraints(cs.clone());
        if result.is_ok() {
            assert!(!cs.is_satisfied().unwrap(), "Out of range index should fail");
        }
    }
}
