# zkLoRA Source Code Structure

This directory contains the core implementation of zkLoRA. Here's a detailed overview of the key components and their interactions.

## Directory Structure

```
src/
├── zklora/                     # Main package directory
│   ├── __init__.py            # Package exports
│   ├── activations_commit.py   # Merkle tree interface
│   ├── base_model_user_mpi/    # Client implementation (User B)
│   ├── lora_contributor_mpi/   # Server implementation (User A)
│   ├── libs/
│   │   └── merkle/            # Rust Merkle tree implementation
│   ├── proof_contract.py       # Transcript and artifact contract
│   └── zk_proof_generator.py   # Native proof generation core
├── src/lib.rs                  # PyO3 native prover (legacy halo2 circuit + fast paths)
├── src/sigma.rs                # Sigma-v4 commit-and-prove backend (default)
├── src/logup.rs                # Zero-knowledge sumcheck LogUp range engine
├── scripts/                    # Sample usage scripts
├── pyproject.toml             # Build configuration
└── requirements.txt           # Dependencies
```

## Implementation Details

### Zero-Knowledge Architecture

The zero-knowledge proof system in zkLoRA is built on transcript-bound LoRA delta statements and a native commit-and-prove backend (sigma-v4). The `zk_proof_generator.py` module orchestrates the proof generation process by:

1. Capturing the base user's local transcript of activations and returned LoRA deltas
2. Binding each proof to a verifier-pinned pre-inference adapter manifest whose entries carry salted Pedersen row commitments plus a one-time exact range proof for every committed weight
3. Generating native `.zklora.*` proof artifacts for contributor-side LoRA invocations
4. Verifying proof artifacts against both the base user's transcript and expected adapter manifest before accepting a module

The verifier must obtain and pin `expected_adapters` out-of-band before inference starts. Contributor-generated adapter manifests are convenience handoff artifacts only; if a manifest is generated after inference or first delivered alongside proofs, it is not trusted to define the expected adapter.

### Multi-Party Inference Protocol

The MPI system enables interaction between the base model user (B) and LoRA provider (A) through:

- Length-prefixed JSON messages for activation exchange
- Asynchronous proof generation that doesn't block inference
- Efficient state management for handling multiple concurrent sessions

The `base_model_user_mpi` and `lora_contributor_mpi` directories contain the client and server implementations respectively, with careful attention to thread safety and resource management.

### Merkle Tree Implementation

The Merkle tree system, implemented in Rust for performance, provides:

- Fast commitment generation for model activations
- Efficient proof verification
- Compact representation of large activation tensors

The Rust implementation is wrapped with Python bindings in the `libs/merkle` directory.

### Performance Considerations

The sigma-v4 backend restructures the statement so that per-proof work no longer scales with the number of adapter weights (the v3 halo2 circuit re-hashed and re-range-checked every weight inside every invocation proof):

- **One-time adapter setup**: at manifest time the contributor produces salted Pedersen row commitments to A and B, per-weight commitments, an aggregated Bulletproofs range proof pinning every weight to the exact `[-value_bound, value_bound]` interval, and a Schnorr proof linking the two commitment forms. The pinned adapter commitment string is the SHA-256 of the deterministic commitment core, so pinning the manifest pins the whole setup.
- **Random-projection sigma protocols**: each invocation proof commits to the rounding quotients and remainders of the exact three-stage quantized pipeline, then Fiat-Shamir challenges project the matrix equations onto scalar equations over committed values (Schwartz-Zippel), proven with generalized Schnorr proofs plus one rank-sized quadratic inner-product argument. Per-proof group work is O(in + rank + out).
- **Zero-knowledge LogUp range engine** (`logup.rs`): rounding remainders and quotients are pinned to their exact intervals via 8-bit digit lookups proven with two sumchecks whose round polynomials are sent as Pedersen commitments; all sumcheck verifier relations are linear over committed scalars and fold into one Schnorr proof. `ZKLORA_RANGE_ENGINE=bulletproofs` opts into compact Bulletproofs instead (~5-8× smaller invocation proofs, ~5-10× slower proving).
- **Parallel batch operations**: the PyO3 bindings release the GIL, and `generate_proofs` / `batch_verify_proofs` fan out across a thread pool (`ZKLORA_PROVE_WORKERS`, `ZKLORA_VERIFY_WORKERS`); MSMs, range chunks, and sumcheck rounds parallelize internally with rayon.
- **Native fast paths with exact fallbacks**: quantized delta computation and the hiding Merkle commitment have Rust implementations that are value-identical to the Python reference paths (covered by parity tests); the Python implementations remain as exact fallbacks.

Statement semantics are identical to v3: the same canonical half-up rounding, the same exact remainder intervals, the same value/intermediate bounds, the same transcript binding and verifier trust boundary. Binding rests on discrete log over ristretto255 plus SHA-256/BLAKE3 collision resistance (the same assumption class as halo2-IPA over Pasta), and adapter hiding improves: commitments are perfectly hiding, and the deterministic blinding factors are derived from the contributor's secret salt keyed together with the full adapter content, so two adapters never share blindings even when they share a shape and a salt (commitment differences across manifests therefore reveal nothing; the v3 Poseidon chain was unsalted). The salt persists via `ZKLORA_ADAPTER_SALT_FILE` (`LoRAServer` defaults it to a file next to its artifacts; keep it stable across restarts so pinned manifests keep matching proofs). v3 artifacts (schema 2, backend `zklora-halo2-v3`) remain verifiable through the legacy path, covered by a byte-faithful regression test.

Measured on a 4-core host (see `benchmarks/sigma_v4_results.md` for the full tables): 768×2×256 proves in 97 ms and verifies in ~0.1 s (v3: 81 s warm / 429 s cold, 1.3 s verify); 16×2×16 proves in 21 ms (v3: 7 s warm); the real 768×4×768 and 768×4×2304 c_attn shapes, which exceeded a 15 GB host under v3, prove in 183 ms and 470 ms. Proving memory drops from gigabytes to tens of megabytes. There is no warm/cold split because there is no keygen.

For detailed usage examples and high-level architecture, please refer to the [main README](../../README.md) in the project root.

## Core Components

### Multi-Party Inference (MPI)
- `base_model_user_mpi/`: Client-side implementation for base model users (User B)
  - Handles remote LoRA module communication
  - Manages model patching and inference
- `lora_contributor_mpi/`: Server-side implementation for LoRA providers (User A)
  - Manages LoRA module serving
  - Handles proof generation requests

### Zero-Knowledge Components
- `zk_proof_generator.py`: Core proof generation and verification
- `proof_contract.py`: Canonical transcript, statement, metadata, and artifact schemas
- `activations_commit.py`: Merkle tree interface for model commitments

### Build & Distribution
- `pyproject.toml`: Package metadata and build configuration
- `requirements.txt`: Project dependencies

### Sample Scripts
- `scripts/base_model_user_sample_script.py`: Example client usage
- `scripts/lora_contributor_sample_script.py`: Example server usage
- `scripts/verify_proofs.py`: Proof verification utility

## Key Interfaces

1. **Base Model User (B)**
```python
from zklora import BaseModelClient

client = BaseModelClient(base_model="distilgpt2")
client.init_and_patch()
loss = client.forward_loss("input text")
```

2. **LoRA Provider (A)**
```python
from zklora import LoRAServer

server = LoRAServer(base_model_name="distilgpt2", 
                   lora_model_id="path/to/lora")
server.list_lora_injection_points()
```

3. **Proof Verification**
```python
from zklora import batch_verify_proofs

verify_time, num_proofs = batch_verify_proofs(
    proof_dir="proof_artifacts",
    transcript="b-transcript.json",
    expected_adapters="adapter-manifest.json",
)
```

In this example, `adapter-manifest.json` is the verifier's pre-inference pinned copy or digest-matched file, not a manifest first generated after inference.

For detailed implementation information, please refer to the individual module documentation. 
