# ZKLoRA Source Code Structure

This directory contains the core implementation of ZKLoRA. Here's a detailed overview of the key components and their interactions.

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

The zero-knowledge proof system in ZKLoRA is built on transcript-bound LoRA delta statements and native Halo2 proofs. The `zk_proof_generator.py` module orchestrates the proof generation process by:

1. Capturing the base user's local transcript of activations and returned LoRA deltas
2. Generating native `.zklora.*` proof artifacts for contributor-side LoRA invocations
3. Verifying proof artifacts against the base user's transcript before accepting a module

### Multi-Party Inference Protocol

The MPI system enables secure interaction between the base model user (B) and LoRA provider (A) through:

- Encrypted communication channels for activation exchange
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

Native Halo2 performance should be measured for the specific LoRA shapes being proven. The v1 implementation prioritizes proof-contract correctness and transcript binding before publishing benchmark claims.

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
)
```

For detailed implementation information, please refer to the individual module documentation. 
