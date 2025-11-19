# zklora/mpi_lora_onnx_exporter.py
"""
New code specifically for 'split inference' (MPI) scenario,
similar to lora_onnx_exporter but with different approach or naming to avoid collisions.
"""

import os
import json
import torch
import numpy as np
import torch.nn as nn


def normalize_lora_matrices_mpi(
    A: torch.Tensor, B: torch.Tensor, x_data: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
    """
    Same shape logic as the older function, but with a new name
    to avoid collisions with the old version.
    x_data => (batch, seq_len, hidden_dim).
    """
    in_dim = x_data.shape[-1]
    a0, a1 = A.shape
    if a0 == in_dim:
        r = a1
    elif a1 == in_dim:
        A = A.transpose(0, 1)
        r = A.shape[1]
    else:
        raise ValueError(f"A shape {A.shape} doesn't match x_data last dim {in_dim}.")

    b0, b1 = B.shape
    if b0 == r:
        out_dim = b1
    elif b1 == r:
        B = B.transpose(0, 1)
        out_dim = B.shape[1]
    else:
        raise ValueError(f"B shape {B.shape} doesn't match rank={r}.")
    return A, B, in_dim, r, out_dim


class LoraShapeTransformerMPI(nn.Module):
    """
    Variation of LoraShapeTransformer used specifically for
    the split-inference approach, with a new class name to avoid collisions.
    """

    def __init__(self, A, B, batch_size, seq_len, hidden_dim):
        super().__init__()
        self.register_buffer("A", A)
        self.register_buffer("B", B)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

    def forward(self, x_1d: torch.Tensor) -> torch.Tensor:
        x_3d = x_1d.view(self.batch_size, self.seq_len, self.hidden_dim)
        out_3d = (x_3d @ self.A) @ self.B
        # Possibly any other modifications unique to your MPI scenario
        out_3d = out_3d + x_3d.mean() + self.A.sum() + self.B.sum()
        out_2d = out_3d.view(1, -1)
        return out_2d


def export_lora_onnx_json_mpi(
    sub_name: str,
    x_data: np.ndarray,
    submodule: nn.Module,
    output_dir: str,
    verbose: bool = False,
):
    """
    The 'split inference' version of the ONNX+JSON exporter.
    Similar logic but a different name to avoid collisions with the old function.
    """
    import torch.onnx
    from torch.onnx import TrainingMode

    batch_size, seq_len, hidden_dim = x_data.shape
    total_size = batch_size * seq_len * hidden_dim
    x_1d = torch.from_numpy(x_data.reshape(1, total_size))

    # If the submodule doesn't have lora_A/lora_B, skip
    if not (hasattr(submodule, "lora_A") and hasattr(submodule, "lora_B")):
        if verbose:
            print(
                f"[export_lora_onnx_json_mpi] No lora_A/B in submodule '{sub_name}', skipping."
            )
        return

    a_keys = list(submodule.lora_A.keys()) if hasattr(submodule.lora_A, "keys") else []
    if not a_keys:
        if verbose:
            print(
                f"[export_lora_onnx_json_mpi] No adapter keys in submodule.lora_A for '{sub_name}'."
            )
        return

    A_mod = submodule.lora_A[a_keys[0]]
    B_mod = submodule.lora_B[a_keys[0]]

    A = A_mod.weight.detach().cpu().float()
    B = B_mod.weight.detach().cpu().float()

    try:
        from .mpi_lora_onnx_exporter import normalize_lora_matrices_mpi

        A_fixed, B_fixed, in_dim, rank, out_dim = normalize_lora_matrices_mpi(
            A, B, x_data
        )
    except ValueError as e:
        if verbose:
            print(f"Shape fix error for '{sub_name}': {e}")
        return

    # Build the shape-transformer
    lora_transformer = LoraShapeTransformerMPI(
        A_fixed, B_fixed, batch_size, seq_len, hidden_dim
    ).eval()

    safe_name = sub_name.replace(".", "_").replace("/", "_")
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, f"{safe_name}.onnx")

    try:
        torch.onnx.export(
            lora_transformer,
            x_1d,
            onnx_path,
            export_params=True,
            do_constant_folding=False,
            opset_version=11,
            input_names=["input_x"],
            output_names=["output"],
            training=TrainingMode.TRAINING,
            keep_initializers_as_inputs=False,
        )
        if verbose:
            print(f"[A-mpi] ONNX => {onnx_path}")
    except Exception as e:
        if verbose:
            print(f"Export error for '{sub_name}': {e}")

    # Save JSON
    json_path = os.path.join(output_dir, f"{safe_name}.json")
    with open(json_path, "w") as f:
        row_data = x_1d.numpy().tolist()
        json.dump({"input_data": row_data}, f)

    if verbose:
        print(f"[A-mpi] JSON => {json_path}, shape => (1, {total_size})")
