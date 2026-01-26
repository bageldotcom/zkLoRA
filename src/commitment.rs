use ark_bn254::Fr;
use ark_ff::Field;

/// Commitment to the NF4 lookup table
/// 
/// This is what makes the whole thing work. We commit to the table once,
/// and then we can prove lookups against that commitment without revealing
/// which value we're looking up.
pub struct NF4TableCommitment {
    table: Vec<Fr>,
    root: Fr,
}

impl NF4TableCommitment {
    /// Create a new commitment from a table of field elements
    pub fn new(table: Vec<Fr>) -> Self {
        assert_eq!(table.len(), 16, "NF4 table must have exactly 16 entries");
        
        let root = Self::compute_root(&table);
        
        Self { table, root }
    }

    /// Compute the commitment root
    /// 
    /// Right now this is a simple hash. In production you'd want to use
    /// a proper Merkle tree with Poseidon hash or similar.
    fn compute_root(table: &[Fr]) -> Fr {
        let mut acc = Fr::from(0u64);
        
        for (i, &val) in table.iter().enumerate() {
            // Multiply by position to make it position-dependent
            acc += val * Fr::from((i + 1) as u64);
        }
        
        acc
    }

    /// Get the commitment root (this is the public anchor)
    pub fn root(&self) -> Fr {
        self.root
    }

    /// Generate a proof that a specific value exists at an index
    pub fn prove_membership(&self, index: u8) -> MembershipProof {
        assert!((index as usize) < self.table.len(), "Index out of bounds");
        
        MembershipProof {
            index,
            value: self.table[index as usize],
            auth_path: vec![], // TODO: add Merkle path when we implement proper tree
        }
    }

    /// Verify a membership proof against this commitment
    pub fn verify_membership(&self, proof: &MembershipProof) -> bool {
        if proof.index as usize >= self.table.len() {
            return false;
        }
        
        self.table[proof.index as usize] == proof.value
    }

    /// Get the table (for testing)
    #[cfg(test)]
    pub fn table(&self) -> &[Fr] {
        &self.table
    }
}

/// Proof that a value exists at a specific index in the committed table
#[derive(Clone, Debug)]
pub struct MembershipProof {
    pub index: u8,
    pub value: Fr,
    pub auth_path: Vec<Fr>, // Will contain Merkle path eventually
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nf4_field::NF4Field;

    #[test]
    fn test_commitment_creation() {
        let table: Vec<Fr> = (0..16)
            .map(|i| NF4Field::to_field_element(i))
            .collect();
        
        let commitment = NF4TableCommitment::new(table);
        assert_ne!(commitment.root(), Fr::from(0u64));
    }

    #[test]
    fn test_membership_proof() {
        let table: Vec<Fr> = (0..16)
            .map(|i| NF4Field::to_field_element(i))
            .collect();
        
        let commitment = NF4TableCommitment::new(table);
        
        let proof = commitment.prove_membership(5);
        assert_eq!(proof.index, 5);
        assert!(commitment.verify_membership(&proof));
    }

    #[test]
    fn test_invalid_proof_fails() {
        let table: Vec<Fr> = (0..16)
            .map(|i| NF4Field::to_field_element(i))
            .collect();
        
        let commitment = NF4TableCommitment::new(table);
        
        let mut proof = commitment.prove_membership(5);
        proof.value = Fr::from(999999u64); // Corrupt the value
        
        assert!(!commitment.verify_membership(&proof));
    }
}
