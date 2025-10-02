<p align="center">
  <img src="bagel-logo.png" alt="Bagel Logo" width="200"/>
</p>

<p align="center">
  <a href="https://twitter.com/bagelopenAI">
    <img src="https://img.shields.io/twitter/follow/bagelopenAI?style=flat-square" alt="Twitter Follow"/>
  </a>
  
  <a href="https://blog.bagel.net">
    <img src="https://img.shields.io/badge/Follow%20on-Substack-orange?style=flat-square&logo=substack" alt="Substack Follow"/>
  </a>
  
  <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">
    <img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg?style=flat-square" alt="License"/>
  </a>
</p>

<h1 align="center">ZKLoRA</h1>
<h3 align="center">Efficient Zero-Knowledge Proofs for LoRA Verification</h3>

<hr>

## ZKLoRA: Efficient Zero-Knowledge Proofs for LoRA Verification

Low-Rank Adaptation (LoRA) is a widely adopted method for customizing large-scale language models. In distributed, untrusted training environments, an open source base model user may want to use LoRA weights created by an external contributor, leading to two requirements:

1. **Base Model User Verification**: The user must confirm that the LoRA weights are effective when paired with the intended base model.
2. **LoRA Contributor Protection**: The contributor must keep their proprietary LoRA weights private until compensation is assured.

To solve this, we created **ZKLoRA** a zero-knowledge verification protocol that relies on polynomial commitments, succinct proofs, and multi-party inference to verify LoRA–base model compatibility without exposing LoRA weights. With ZKLoRA, verification of LoRA modules takes just 1-2 seconds, even for state-of-the-art language models with tens of billions of parameters.

For detailed information about this research, please refer to [our paper](https://arxiv.org/abs/2501.13965).

<h2 align="center">Quick Usage Instructions</h2>

### 1. LoRA Contributor Side (User A)

First, install ZKLoRA using pip:
```bash
pip install zklora
```

Use `src/scripts/lora_contributor_sample_script.py` to:
- Host LoRA submodules
- Handle inference requests
- Generate proof artifacts

```python
import argparse
import threading
import time

from zklora import LoRAServer, AServerTCP

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port_a", type=int, default=30000)
    parser.add_argument("--base_model", default="distilgpt2")
    parser.add_argument("--lora_model_id", default="ng0-k1/distilgpt2-finetuned-es")
    parser.add_argument("--out_dir", default="a-out")
    args = parser.parse_args()

    stop_event = threading.Event()
    server_obj = LoRAServer(args.base_model, args.lora_model_id, args.out_dir)
    t = AServerTCP(args.host, args.port_a, server_obj, stop_event)
    t.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[A-Server] stopping.")
    stop_event.set()
    t.join()

if __name__ == "__main__":
    main()
```

### 2. Base Model User Side (User B)

Use `src/scripts/base_model_user_sample_script.py` to:
- Load and patch the base model
- Connect to A's submodules
- Perform inference
- Trigger proof generation

```python
import argparse

from zklora import BaseModelClient

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host_a", default="127.0.0.1")
    parser.add_argument("--port_a", type=int, default=30000)
    parser.add_argument(
        "--contributors",
        nargs="*",
        help="Additional LoRA contributors as host:port",
    )
    parser.add_argument("--base_model", default="distilgpt2")
    parser.add_argument("--combine_mode", choices=["replace","add_delta"], default="add_delta")
    args = parser.parse_args()

    contributors = [(args.host_a, args.port_a)]
    if args.contributors:
        for item in args.contributors:
            host, port = item.split(":")
            contributors.append((host, int(port)))

    client = BaseModelClient(
        base_model=args.base_model,
        combine_mode=args.combine_mode,
        contributors=contributors,
    )
    client.init_and_patch()

    # Run inference => triggers remote LoRA calls on A
    text = "Hello World, this is a LoRA test."
    loss_val = client.forward_loss(text)
    print(f"[B] final loss => {loss_val:.4f}")

    # End inference => A finalizes proofs offline
    client.end_inference()
    print("[B] done. B can now fetch proof files from A and verify them offline.")

if __name__=="__main__":
    main()
```

### 3. Proof Verification

Use `src/scripts/verify_proofs.py` to validate the proof artifacts:

```python
#!/usr/bin/env python3
"""
Verify LoRA proof artifacts in a given directory.

Example usage:
  python verify_proofs.py --proof_dir a-out --verbose
"""

import argparse
from zklora import batch_verify_proofs

def main():
    parser = argparse.ArgumentParser(
        description="Verify LoRA proof artifacts in a given directory."
    )
    parser.add_argument(
        "--proof_dir",
        type=str,
        default="proof_artifacts",
        help="Directory containing proof files (.pf), plus settings, vk, srs."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print more details during verification."
    )
    args = parser.parse_args()

    total_verify_time, num_proofs = batch_verify_proofs(
        proof_dir=args.proof_dir,
        verbose=args.verbose
    )
    print(f"Done verifying {num_proofs} proofs. Total time: {total_verify_time:.2f}s")

if __name__ == "__main__":
    main()
```

### 4. Polynomial Commitment of Activations

ZKLoRA includes a robust polynomial commitment scheme for securely committing to neural network activations without revealing the underlying data. This cryptographic primitive enables privacy-preserving verification of computations.

#### Basic Usage

```python
from zklora import commit_activations, verify_commitment

# Commit to activation data stored in JSON format
commitment = commit_activations("activations.json")

# Verify the commitment against original data
is_valid = verify_commitment("activations.json", commitment)
assert is_valid
```

#### Commitment Features

The polynomial commitment scheme provides several key properties:

- **Zero-Knowledge**: Commitments reveal no information about the underlying activation data
- **Binding**: Once created, commitments cannot be changed to refer to different data
- **Deterministic Verification**: Given the same data and nonce, verification is consistent
- **Cryptographic Security**: Uses BLAKE3 hashing and polynomial arithmetic over finite fields

#### Advanced Usage Examples

**Committing to Different Data Types:**

```python
import json
from zklora import commit_activations, verify_commitment

# Example with floating point activations
activation_data = {
    "input_data": [1.5, 2.7, -3.14, 0.0, 42.8]
}
with open("float_activations.json", "w") as f:
    json.dump(activation_data, f)

commitment = commit_activations("float_activations.json")
assert verify_commitment("float_activations.json", commitment)

# Example with nested activation structures (automatically flattened)
nested_data = {
    "input_data": [[1, 2], [3, [4, 5]], 6]
}
with open("nested_activations.json", "w") as f:
    json.dump(nested_data, f)

nested_commitment = commit_activations("nested_activations.json")
assert verify_commitment("nested_activations.json", nested_commitment)
```

**Batch Processing for Multiple Modules:**

```python
import os
from zklora import commit_activations, verify_commitment

# Commit to activations from multiple LoRA modules
module_commitments = {}
activation_files = ["module1_acts.json", "module2_acts.json", "module3_acts.json"]

for file_path in activation_files:
    if os.path.exists(file_path):
        commitment = commit_activations(file_path)
        module_commitments[file_path] = commitment
        print(f"Committed to {file_path}: {commitment[:50]}...")

# Verify all commitments
for file_path, commitment in module_commitments.items():
    is_valid = verify_commitment(file_path, commitment)
    print(f"Verification for {file_path}: {'✓ VALID' if is_valid else '✗ INVALID'}")
```

**Understanding Commitment Structure:**

```python
import json
from zklora import commit_activations

# Create a commitment and examine its structure
commitment_str = commit_activations("activations.json")
commitment_data = json.loads(commitment_str)

print("Commitment structure:")
print(f"Root hash: {commitment_data['root']}")     # Merkle tree root
print(f"Nonce: {commitment_data['nonce']}")        # Cryptographic nonce
print(f"Root length: {len(commitment_data['root'])}")  # 66 chars (0x + 64 hex)
print(f"Nonce length: {len(commitment_data['nonce'])}")  # 66 chars (0x + 64 hex)
```

#### Security Properties

1. **Collision Resistance**: Different activation datasets produce different commitments
2. **Hiding Property**: Commitments reveal no information about the committed data
3. **Non-Malleability**: Cannot modify commitments without detection
4. **Efficient Verification**: Verification scales logarithmically with data size

#### Use Cases in Multi-Party LoRA

- **Activation Integrity**: Ensure base model activations haven't been tampered with
- **Privacy-Preserving Audits**: Allow verification without revealing sensitive data
- **Multi-Contributor Scenarios**: Enable secure collaboration between multiple LoRA providers
- **Proof Generation**: Create verifiable evidence of correct computation

### 5. Running Tests

Run unit tests with:

```bash
pytest
```

<hr>

<h2 align="center">Code Structure</h2>

For detailed information about the codebase organization and implementation details, see [Code Structure](src/zklora/README.md).

<h2 align="center">Summary</h2>

<table>
<tr>
<td>✓</td><td><strong>Trust-Minimized Verification:</strong> Zero-knowledge proofs enable secure LoRA validation</td>
</tr>
<tr>
<td>✓</td><td><strong>Rapid Verification:</strong> 1-2 second processing per module, even for billion-parameter models</td>
</tr>
<tr>
<td>✓</td><td><strong>Multi-Party Inference:</strong> Protected activation exchange between parties</td>
</tr>
<tr>
<td>✓</td><td><strong>Complete Privacy:</strong> LoRA weights remain confidential while ensuring compatibility</td>
</tr>
<tr>
<td>✓</td><td><strong>Production Ready:</strong> Efficiently scales to handle multiple LoRA modules</td>
</tr>
</table>

Polynomial commitments for base model activations and multi-contributor LoRA scenarios are supported starting in version 0.1.2.

<h2 align="center">Credits</h2>

ZKLoRA is built upon these outstanding open source projects:

| Project | Description |
|---------|-------------|
| [PEFT](https://github.com/huggingface/peft) | Parameter-Efficient Fine-Tuning library by Hugging Face |
| [Transformers](https://github.com/huggingface/transformers) | State-of-the-art Natural Language Processing |
| [dusk-merkle](https://github.com/dusk-network/dusk-merkle) | Merkle tree implementation in Rust |
| [BLAKE3](https://github.com/BLAKE3-team/BLAKE3) | Cryptographic hash function |
| [EZKL](https://github.com/zkonduit/ezkl) | Zero-knowledge proof system for neural networks |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | Cross-platform ML model inference |

<hr>

<p align="center">
<sub>Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International</sub>
</p>
