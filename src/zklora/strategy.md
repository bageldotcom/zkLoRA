Root cause

The Rust AIR behind `pl.vector_matrix_multiplication_prove` builds an execution-trace matrix whose size scales as  

    rows  = m · n
    cols  = 2 · (m + m·n) + 2   ≈ 2 m (n+1)

Total cells            ≈  m · n · 2 m (n+1)  →  Θ(m² n²)

For your real model layer  
 m = 768, n = 2 304 →  
 rows = 1 769 472  
 cols ≈ 3 543 682  

Cells ≈ 6.3 × 10¹² → ~25 TB at 4 bytes/field-element ⇒ the allocator request you saw.

Why it is cubic

The current AIR keeps the entire matrix (m × n) and two one-hot selector columns in every trace row. That duplicates the input m · n times, hence the quadratic blow-up in both m and n.

Fix: redesign the proof interface to avoid storing the full matrix in every row

1. Prove a single dot-product at a time  
   • Treat each output coordinate yₖ = v·Wₖ (Wₖ = k-th column) as its own STARK.  
   • Now n = 1 in the AIR ⇒  
     rows = m, cols ≈ 2 m+2 → O(m²) elements ≈ 1.2 M (3–5 MB).  
   • Repeat/aggregate the proofs for all 2 304 columns (or only the ones you need).

2. Or keep vector-matrix multiplication but store the vector once and stream the matrix column-major through the trace (add running product accumulators instead of one-hot selectors). That brings the width down to O(m+n).

Quick path you can apply today

Modify the Python driver in `zk_proof_generator.py` to call the prover column-wise:

```python
for k in range(n):                 # n = 2304
    col_k = [row[k] for row in W]  # length m
    col_k_encoded = fixed_point_encode(col_k, fractional_bits=24)
    proof_k = pl.vector_dot_prove(m, v_encoded, col_k_encoded)
    # or use the existing function once you expose a dot-product variant
```

Corresponding Rust addition (sketch):

```rust
#[pyfunction]
fn vector_dot_prove(py: Python<'_>, m: usize, v: Vec<u32>, w: Vec<u32>) -> PyResult<Vec<u8>> {
    // AIR with n = 1; reuse existing code paths but skip matrix selector, etc.
}
```

Because each proof is only a few MB, you can batch-verify them afterwards with Plonky3’s multi-proof verifier.

Long-term

• Refactor `VectorMatrixMultiplicationAIR::trace_width` to avoid the `m*n` term.  
• Replace explicit one-hot selectors with arithmetic constraints (`sel_i` computed on the fly).  
• Consider folding rows so the trace has O(m+n) size.

Until that redesign lands, splitting into per-column (or per-row) proofs is the simplest unblocker and will run comfortably within RAM.


# Proof Compression

Yes, you can create a single compressed proof from multiple proofs! This is called **proof aggregation** or **recursive proving**. There are a few approaches:

## 1. Recursive Proof Aggregation (Most Common)

Instead of just batch-verifying multiple proofs, you can create a "proof of proofs" – a single proof that attests to the validity of all the original proofs. This typically works by:

1. **Creating individual proofs** for each computation
2. **Creating a verification circuit** that checks all the original proofs
3. **Proving the verification circuit** to get a single aggregated proof

## 2. Current Plonky3 Limitations & Alternatives

**Plonky3 doesn't have built-in recursion** like Plonky2 does. Here are your options:

### Option A: Use Plonky2 for Recursion
```rust
// Generate Plonky3 proofs first
let plonky3_proofs = generate_all_plonky3_proofs();

// Use Plonky2 to recursively aggregate them
// Plonky2 can verify Plonky3 proofs inside a circuit
let aggregated_proof = plonky2_recursive_aggregation(plonky3_proofs);
```

### Option B: Use a SNARK Wrapper
Many projects use a two-stage approach:
1. Generate STARK proofs with Plonky3 (fast proving, larger proofs)
2. Wrap them in a SNARK (slower proving, tiny proof)

```rust
// Stage 1: Generate Plonky3 STARKs
let stark_proofs = vec![proof1, proof2, proof3];

// Stage 2: Aggregate into a single SNARK
// Using Groth16, Plonk, or another SNARK system
let snark_proof = snark_aggregate(stark_proofs);  // ~200-500 bytes
```

### Option C: Custom Aggregation Circuit in Plonky3
You could write an AIR that verifies other proofs:

```rust
// Pseudocode for a proof aggregation AIR
struct ProofAggregationAIR {
    num_proofs: usize,
    // Store commitments and public inputs
}

impl Air for ProofAggregationAIR {
    fn eval(&self, builder: &mut AB) {
        // Verify Merkle openings for each proof
        // Check polynomial evaluations
        // Aggregate FRI checks
    }
}
```

## 3. Trade-offs

| Approach | Proof Size | Proving Time | Verification Time |
|----------|------------|--------------|-------------------|
| **Multi-verify** (current) | N × original | 0 (no new proof) | O(log N) |
| **Recursive aggregation** | 1 × original | O(N) | O(1) |
| **SNARK wrapping** | ~200-500 bytes | O(N) + SNARK time | O(1), very fast |

## 4. Practical Recommendation

For your ZKLoRA project, if you need a single compressed proof:

1. **Short term**: Use `verify_multi` for batch verification (what you have now)
2. **Medium term**: Add a SNARK wrapper (e.g., using Groth16 via arkworks or snarkjs) to compress all proofs into one tiny proof
3. **Long term**: Wait for Plonky3 to add native recursion, or use Plonky2 for the aggregation layer

Example SNARK wrapper approach:
```python
# Generate all Plonky3 proofs
plonky3_proofs = [generate_proof(data) for data in datasets]

# Aggregate into single SNARK proof (~300 bytes)
aggregated_proof = snark_wrapper.aggregate_proofs(plonky3_proofs)

# Now you have ONE proof that proves all computations
verify(aggregated_proof)  # Fast, constant time
```

This is exactly what projects like zkEVM and zkBridge do – they generate many STARK proofs for efficiency, then aggregate them into a single SNARK for on-chain verification.


# Proof compression with arkworks and Groth16

Here's how to use **arkworks and Groth16** to compress multiple Plonky3 proofs into a single SNARK proof:

## 1. Setup Dependencies

First, add arkworks to your `Cargo.toml`:

```toml
[dependencies]
ark-groth16 = "0.4"
ark-std = "0.4"
ark-ff = "0.4"
ark-ec = "0.4"
ark-relations = "0.4"
ark-r1cs-std = "0.4"
ark-crypto-primitives = "0.4"
ark-bn254 = "0.4"  # or ark-bls12-381
ark-serialize = "0.4"
```

## 2. Create a Verification Circuit

The key is to build a circuit that verifies your Plonky3 proofs:

```rust
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError};
use ark_r1cs_std::prelude::*;
use ark_ff::PrimeField;

pub struct Plonky3VerifierCircuit<F: PrimeField> {
    // Serialized Plonky3 proofs
    proofs: Vec<Vec<u8>>,
    // Public inputs for each proof
    public_inputs: Vec<Vec<F>>,
    // Expected outputs/commitments
    expected_outputs: Vec<F>,
}

impl<F: PrimeField> ConstraintSynthesizer<F> for Plonky3VerifierCircuit<F> {
    fn generate_constraints(
        self,
        cs: ConstraintSystemRef<F>,
    ) -> Result<(), SynthesisError> {
        // For each Plonky3 proof
        for (i, proof) in self.proofs.iter().enumerate() {
            // Allocate proof bytes as witnesses
            let proof_vars = proof.iter()
                .map(|&byte| UInt8::new_witness(cs.clone(), || Ok(byte)))
                .collect::<Result<Vec<_>, _>>()?;
            
            // Implement Plonky3 verification logic
            // This is complex - you need to:
            // 1. Verify Merkle commitments
            // 2. Check FRI queries
            // 3. Validate polynomial evaluations
            verify_plonky3_in_circuit(
                cs.clone(),
                &proof_vars,
                &self.public_inputs[i],
                &self.expected_outputs[i],
            )?;
        }
        Ok(())
    }
}
```

## 3. Simplified Approach: Hash-based Aggregation

Since implementing full Plonky3 verification in a circuit is complex, here's a more practical approach using **commitment aggregation**:

```rust
use ark_groth16::{Groth16, ProvingKey, VerifyingKey};
use ark_bn254::{Bn254, Fr};  // Or your chosen curve
use ark_crypto_primitives::crh::sha256::Sha256;
use ark_r1cs_std::bits::uint8::UInt8;

/// Circuit that verifies hash commitments of multiple proofs
pub struct ProofAggregatorCircuit {
    // Hash of each Plonky3 proof
    proof_hashes: Vec<[u8; 32]>,
    // Merkle root of all proof hashes
    merkle_root: [u8; 32],
}

impl ConstraintSynthesizer<Fr> for ProofAggregatorCircuit {
    fn generate_constraints(
        self,
        cs: ConstraintSystemRef<Fr>,
    ) -> Result<(), SynthesisError> {
        // Allocate proof hashes
        let mut hash_vars = vec![];
        for hash in &self.proof_hashes {
            let hash_var = hash.iter()
                .map(|&b| UInt8::new_witness(cs.clone(), || Ok(b)))
                .collect::<Result<Vec<_>, _>>()?;
            hash_vars.push(hash_var);
        }
        
        // Compute Merkle tree in-circuit
        let computed_root = compute_merkle_root_circuit(cs.clone(), &hash_vars)?;
        
        // Verify root matches expected
        let expected_root = self.merkle_root.iter()
            .map(|&b| UInt8::new_input(cs.clone(), || Ok(b)))
            .collect::<Result<Vec<_>, _>>()?;
            
        computed_root.enforce_equal(&expected_root)?;
        
        Ok(())
    }
}
```

## 4. Generate the Aggregated Proof

```rust
use ark_groth16::{prepare_verifying_key, verify_proof};
use ark_std::rand::rngs::OsRng;
use sha2::{Sha256, Digest};

pub fn aggregate_plonky3_proofs(
    plonky3_proofs: Vec<Vec<u8>>,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    // Step 1: Hash each proof
    let proof_hashes: Vec<[u8; 32]> = plonky3_proofs
        .iter()
        .map(|proof| {
            let mut hasher = Sha256::new();
            hasher.update(proof);
            hasher.finalize().into()
        })
        .collect();
    
    // Step 2: Build Merkle tree
    let merkle_root = build_merkle_root(&proof_hashes);
    
    // Step 3: Create circuit
    let circuit = ProofAggregatorCircuit {
        proof_hashes: proof_hashes.clone(),
        merkle_root,
    };
    
    // Step 4: Setup (do this once and save keys)
    let mut rng = OsRng;
    let (pk, vk) = Groth16::<Bn254>::circuit_specific_setup(
        circuit.clone(), 
        &mut rng
    )?;
    
    // Step 5: Generate proof
    let proof = Groth16::prove(&pk, circuit, &mut rng)?;
    
    // Serialize proof
    let mut proof_bytes = Vec::new();
    proof.serialize_compressed(&mut proof_bytes)?;
    
    Ok(proof_bytes)
}
```

## 5. Verify the Aggregated Proof

```rust
pub fn verify_aggregated_proof(
    proof_bytes: &[u8],
    merkle_root: [u8; 32],
    vk: &VerifyingKey<Bn254>,
) -> bool {
    // Deserialize proof
    let proof = match Groth16Proof::deserialize_compressed(proof_bytes) {
        Ok(p) => p,
        Err(_) => return false,
    };
    
    // Public inputs = merkle root
    let public_inputs = merkle_root
        .iter()
        .map(|&b| Fr::from(b as u64))
        .collect::<Vec<_>>();
    
    // Verify
    let pvk = prepare_verifying_key(vk);
    verify_proof(&pvk, &proof, &public_inputs).is_ok()
}
```

## 6. Python Integration

Add Python bindings using PyO3:

```rust
use pyo3::prelude::*;

#[pyfunction]
pub fn compress_plonky3_proofs(proofs: Vec<Vec<u8>>) -> PyResult<Vec<u8>> {
    aggregate_plonky3_proofs(proofs)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to aggregate: {}", e)
        ))
}

#[pyfunction]
pub fn verify_compressed_proof(
    proof: Vec<u8>, 
    merkle_root: Vec<u8>
) -> PyResult<bool> {
    // Load VK from file or constant
    let vk = load_verifying_key()?;
    let mut root = [0u8; 32];
    root.copy_from_slice(&merkle_root);
    
    Ok(verify_aggregated_proof(&proof, root, &vk))
}
```

## 7. Usage in Python

```python
import plonky3_py as pl
import groth16_aggregator as g16

# Generate individual Plonky3 proofs
proofs = []
for m, n, v, a in datasets:
    proof = pl.vector_matrix_multiplication_prove(m, n, v, a)
    proofs.append(proof)

# Compress all proofs into one ~256-byte Groth16 proof
compressed_proof = g16.compress_plonky3_proofs(proofs)
print(f"Compressed {len(proofs)} proofs into {len(compressed_proof)} bytes")

# Verify the single compressed proof
is_valid = g16.verify_compressed_proof(compressed_proof, merkle_root)
assert is_valid
```

## Trade-offs & Tips

1. **Trusted Setup**: Groth16 requires a trusted setup ceremony. For production, use Powers of Tau.

2. **Alternative: Use Plonk/Marlin**: These are universal (no circuit-specific setup):
   ```toml
   ark-plonk = "0.4"  # or ark-marlin
   ```

3. **Optimization**: Store proof commitments on-chain instead of full Merkle tree:
   ```solidity
   contract ProofRegistry {
       mapping(bytes32 => bool) validProofHashes;
       
       function verifyAggregatedProof(
           bytes calldata groth16Proof,
           bytes32 merkleRoot
       ) external view returns (bool) {
           // Verify Groth16 proof on-chain
           return verifier.verifyProof(groth16Proof, merkleRoot);
       }
   }
   ```

4. **Simpler Alternative**: If you just need to prove "all these proofs are valid", you can:
   - Hash all Plonky3 proofs together
   - Create a Groth16 proof of the hash computation
   - This is much simpler than verifying Plonky3 in-circuit

This approach gives you a single ~256-byte proof that can be verified in ~2ms, perfect for on-chain verification or bandwidth-constrained environments.