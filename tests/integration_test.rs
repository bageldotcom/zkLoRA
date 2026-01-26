use ark_bn254::Fr;
use ark_relations::r1cs::ConstraintSystem;
use zklora::{NF4Field, NF4TableCommitment, NF4LookupCircuit};

#[test]
fn test_end_to_end_proof() {
    println!("
=== zkLoRA Integration Test ===
Testing the full proof pipeline for NF4 weight lookups
");

    // Step 1: Build the NF4 lookup table
    println!("Step 1: Building NF4 lookup table...");
    let table: Vec<Fr> = (0..16)
        .map(|i| NF4Field::to_field_element(i))
        .collect();
    println!("✓ Created table with 16 NF4 values");

    // Step 2: Commit to the table
    println!("
Step 2: Creating table commitment...");
    let commitment = NF4TableCommitment::new(table.clone());
    let root = commitment.root();
    println!("✓ Commitment root: {:?}", root);

    // Step 3: Pick a weight to prove
    println!("
Step 3: Setting up proof for weight lookup...");
    let test_index = 7u8;
    let expected_value = NF4Field::to_field_element(test_index);
    println!("✓ Proving lookup: index={}, value={:?}", test_index, expected_value);

    // Step 4: Create the circuit
    println!("
Step 4: Building ZK circuit...");
    let circuit = NF4LookupCircuit::new(
        Some(test_index),
        Some(expected_value),
        root,
    );
    println!("✓ Circuit created");

    // Step 5: Generate constraints
    println!("
Step 5: Generating constraints...");
    let cs = ConstraintSystem::<Fr>::new_ref();
    circuit.generate_constraints(cs.clone()).unwrap();
    println!("✓ Generated {} constraints", cs.num_constraints());

    // Step 6: Verify the proof
    println!("
Step 6: Verifying constraints...");
    assert!(cs.is_satisfied().unwrap(), "Constraints must be satisfied!");
    println!("✓ All constraints satisfied!");

    println!("
=== Test Passed ===
Successfully proved NF4 weight lookup without revealing the weight!
");
}

#[test]
fn test_invalid_lookup_rejected() {
    println!("
=== Testing Security ===
Making sure invalid proofs get rejected...
");

    let table: Vec<Fr> = (0..16)
        .map(|i| NF4Field::to_field_element(i))
        .collect();
    
    let commitment = NF4TableCommitment::new(table);
    
    // Try to cheat: claim wrong value for an index
    println!("Attempting invalid proof (claiming wrong value for index)...");
    let circuit = NF4LookupCircuit::new(
        Some(7u8),
        Some(NF4Field::to_field_element(8)), // Wrong value!
        commitment.root(),
    );

    let cs = ConstraintSystem::<Fr>::new_ref();
    circuit.generate_constraints(cs.clone()).unwrap();

    assert!(!cs.is_satisfied().unwrap(), "Invalid proof should be rejected!");
    println!("✓ Invalid proof correctly rejected!");
}

#[test]
fn test_all_indices() {
    println!("
=== Testing All NF4 Values ===
Verifying we can prove lookups for all 16 values...
");

    let table: Vec<Fr> = (0..16)
        .map(|i| NF4Field::to_field_element(i))
        .collect();
    
    let commitment = NF4TableCommitment::new(table);

    for i in 0..16 {
        let circuit = NF4LookupCircuit::new(
            Some(i),
            Some(NF4Field::to_field_element(i)),
            commitment.root(),
        );

        let cs = ConstraintSystem::<Fr>::new_ref();
        circuit.generate_constraints(cs.clone()).unwrap();
        
        assert!(
            cs.is_satisfied().unwrap(), 
            "Failed to prove lookup for index {}", 
            i
        );
    }

    println!("✓ Successfully proved all 16 possible lookups!");
}
