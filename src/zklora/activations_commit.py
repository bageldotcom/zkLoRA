import merkle
import json
import numpy as np


def get_merkle_root(activations_path: str) -> str:
    """
    Calculate the Merkle root hash of model activations stored in a JSON file.

    Args:
        activations_path: Path to JSON file containing model activations under "input_data" key

    Returns:
        str: Hexadecimal string of the Merkle root hash, prefixed with "0x"
    """
    # Load the intermediate activations from JSON file
    with open(activations_path, "r") as f:
        activations = json.load(f)

    # Convert nested data to numpy array and flatten
    flattened_np = np.array(activations["input_data"]).reshape(-1)

    # Get and return the Merkle root hash
    return merkle.insert_values(flattened_np.tolist())


if __name__ == "__main__":
    activations_path = "intermediate_activations/base_model_model_lm_head.json"
    merkle_root = get_merkle_root(activations_path)
    print("Merkle root:", merkle_root)
