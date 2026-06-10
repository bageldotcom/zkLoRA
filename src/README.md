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
├── src/lib.rs                  # Halo2/PyO3 native prover
├── scripts/                    # Sample usage scripts
├── pyproject.toml             # Build configuration
└── requirements.txt           # Dependencies
```

## Implementation Details

### Zero-Knowledge Architecture

The zero-knowledge proof system in zkLoRA is built on transcript-bound LoRA delta statements and native Halo2 proofs. The `zk_proof_generator.py` module orchestrates the proof generation process by:

1. Capturing the base user's local transcript of activations and returned LoRA deltas
2. Binding each proof to a verifier-pinned pre-inference adapter manifest with a Poseidon adapter commitment
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

Native Halo2 performance should be measured for the specific LoRA shapes being proven. The v3 backend keeps the v2 proof contract (same statement format, same Poseidon adapter-commitment scheme) while making proving and verification dramatically faster:

- **Lookup-based range checks**: signed interval checks decompose values into window-sized limbs via a running sum constrained against a lookup table, instead of one boolean row per bit. The interval semantics are exact (the top limb is scaled to its residual width), and the circuit drops provably redundant per-product checks whose bounds already follow from the range-checked operands. At a 768×4×2304 LoRA shape this reduces the circuit from k≈26 to k=21 (32× fewer rows).
- **Keyed SRS/proving-key/verifying-key caches**: params and keys are derived deterministically from the statement shape (dims, fixed-point bits, scaling) and reused across invocations of the same module. First proof per shape pays keygen; subsequent proofs and all verifications are keygen-free. Cache sizes are tunable via `ZKLORA_PARAMS_CACHE_CAP`, `ZKLORA_PK_CACHE_CAP`, and `ZKLORA_VK_CACHE_CAP`.
- **Parallel batch operations**: the PyO3 bindings release the GIL, and `generate_proofs` / `batch_verify_proofs` fan out across a thread pool (`ZKLORA_PROVE_WORKERS`, `ZKLORA_VERIFY_WORKERS`).
- **Native fast paths with exact fallbacks**: quantized delta computation and the hiding Merkle commitment have Rust implementations that are value-identical to the Python reference paths (covered by parity tests); the Python implementations remain as exact fallbacks.

Proofs and verifying keys from the v2 backend are not compatible with v3 (the `backend` field in statements changed to `zklora-halo2-v3`), but pinned adapter manifests remain valid: the adapter commitment scheme is unchanged.

Measured on a 4-core machine (warm key cache): a 2×1×2 relation proves in ~0.7s and verifies in ~0.02s (previously ~23s / ~18s), an 8×2×8 relation proves in ~1.4s (previously >5 minutes), and the real 768×4×2304 c_attn shape becomes feasible at k=21.

Memory note: proving memory is dominated by halo2's extended-domain evaluations (the Poseidon chip's gate degree implies an 8× extended domain). Expect roughly 7–8 GB peak at k=20 (e.g. 768×4×768) and >15 GB at k=21 (768×4×2304); size the prover host accordingly. Verification is lightweight once the verifying key is cached.

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
