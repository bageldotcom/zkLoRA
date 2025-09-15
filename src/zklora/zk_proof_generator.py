import os
import glob
import json
import time
import asyncio
from typing import NamedTuple, Optional, Literal

import numpy as np
import onnx
import onnxruntime
import ezkl


class ProofPaths(NamedTuple):
    circuit: str
    settings: str
    srs: str
    verification_key: str
    proving_key: str
    witness: str
    proof: str


def resolve_proof_paths(proof_dir: str, base_name: str) -> Optional[ProofPaths]:
    """Retrieves paths for all required proof-related files given a directory and base name."""
    return ProofPaths(
        circuit=os.path.join(proof_dir, f"{base_name}.ezkl"),
        settings=os.path.join(proof_dir, f"{base_name}_settings.json"),
        srs=os.path.join(proof_dir, "kzg.srs"),
        verification_key=os.path.join(proof_dir, f"{base_name}.vk"),
        proving_key=os.path.join(proof_dir, f"{base_name}.pk"),
        witness=os.path.join(proof_dir, f"{base_name}_witness.json"),
        proof=os.path.join(proof_dir, f"{base_name}.pf"),
    )


def batch_verify_proofs(
    proof_dir: str = "proof_artifacts", verbose: bool = False
) -> tuple[float, int]:
    """Batch verifies proofs for all ONNX models in the specified directory.

    Args:
        onnx_dir: Directory containing ONNX model files
        proof_dir: Directory containing proof artifacts (proofs, verification keys, etc.)

    ## Returns:
        tuple[float, int]: Total time spent verifying proofs, number of proofs verified
    """
    proof_files = glob.glob(os.path.join(proof_dir, "*.pf"))
    if not proof_files:
        print(f"No proof files found in {proof_dir}.")
        return 0.0, 0  # or return None

    total_verify_time = 0.0

    for proof_file in proof_files:
        base_name = os.path.splitext(os.path.basename(proof_file))[0]
        names = resolve_proof_paths(proof_dir, base_name)
        if names is None:
            continue
        # Only unpack the variables we need
        paths = names  # more descriptive variable name

        print(f"Verifying proof for {base_name}...")
        start_time = time.time()
        verify_ok = ezkl.verify(
            paths.proof, paths.settings, paths.verification_key, paths.srs
        )
        end_time = time.time()

        duration = end_time - start_time
        total_verify_time += duration
        if verbose:
            print(f"Verification took {duration:.2f} seconds")

        if verify_ok:
            if verbose:
                print(f"Proof verified successfully for {base_name}!\n")
        else:
            if verbose:
                print(f"Verification failed for {base_name}.\n")

    print(f"Total proofs verified: {len(proof_files)}")
    return total_verify_time, len(proof_files)

ZKBackend = Literal["ezkl", "plonky3"]

async def generate_proofs(
    onnx_dir: str = "lora_onnx_params",
    json_dir: str = "intermediate_activations",
    output_dir: str = "proof_artifacts",
    verbose: bool = False,
    zk_backend: ZKBackend = "ezkl",
) -> Optional[tuple[float, float, float, int, int]]:
    """Asynchronously scans onnx_dir for .onnx files and json_dir for .json files.
    For each matching pair, runs:
    1) gen_settings + compile_circuit
    2) gen_srs + setup
    3) gen_witness (async)
    4) prove

    Args:
        onnx_dir: Directory containing ONNX model files
        json_dir: Directory containing input JSON files
        output_dir: Directory to store proof artifacts (default: current directory)

    Returns:
        - total_settings_time: Total time spent on settings/setup
        - total_witness_time: Total time spent generating witnesses
        - total_prove_time: Total time spent generating proofs
        - count_onnx_files: Number of ONNX files successfully processed
    """

    os.makedirs(output_dir, exist_ok=True)

    onnx_files = glob.glob(os.path.join(onnx_dir, "*.onnx"))
    if not onnx_files:
        print(f"No ONNX files found in {onnx_dir}.")
        return

    if verbose:
        print(f"Found {len(onnx_files)} ONNX files in {onnx_dir}.")

    total_settings_time = 0
    total_witness_time = 0
    total_prove_time = 0
    count_onnx_files = 0
    total_params = 0
    print("Processing ONNX files for proof generation...")
    for onnx_path in onnx_files:
        base_name = os.path.splitext(os.path.basename(onnx_path))[0]
        json_path = os.path.join(json_dir, base_name + ".json")
        if not os.path.isfile(json_path):
            print(f"No matching JSON for {onnx_path}, skipping.")
            continue

        if verbose:
            print("==========================================")
            print(f"Preparing to prove with ONNX: {onnx_path}")
            print(f"Matching JSON: {json_path}")

        onnx_model = onnx.load(onnx_path)
        param_count = sum(np.prod(param.dims) for param in onnx_model.graph.initializer)
        if verbose:
            print(f"Number of parameters: {param_count:,}")
        total_params += param_count

        names = resolve_proof_paths(output_dir, base_name)
        if names is None:
            continue
        (
            circuit_name,
            settings_file,
            srs_file,
            vk_file,
            pk_file,
            witness_file,
            proof_file,
        ) = names

        if zk_backend == "ezkl":
            py_args = ezkl.PyRunArgs()
            py_args.input_visibility = "public"
            py_args.output_visibility = "public"
            py_args.param_visibility = "private"
            py_args.logrows = 20

            if verbose:
                print("Generating settings & compiling circuit...")
            start_time = time.time()

            # 1) gen_settings + compile_circuit
            ezkl.gen_settings(onnx_path, settings_file, py_run_args=py_args)
            ezkl.compile_circuit(onnx_path, circuit_name, settings_file)

            # 2) SRS + setup
            if not os.path.isfile(srs_file):
                ezkl.gen_srs(srs_file, py_args.logrows)
            ezkl.setup(circuit_name, vk_file, pk_file, srs_file)
            end_time = time.time()
            if verbose:
                print(f"Setup for {base_name} took {end_time - start_time:.2f} sec")
            total_settings_time += end_time - start_time

            # Local check
            with open(json_path, "r") as f:
                data = json.load(f)
            input_array = np.array(data["input_data"], dtype=np.float32)
            if verbose:
                print("Input shape from JSON:", input_array.shape)
            session = onnxruntime.InferenceSession(onnx_path)
            out = session.run(None, {"input_x": input_array})
            if verbose:
                print("Local ONNX output shape:", out[0].shape)

            # 3) gen_witness (async)
            if verbose:
                print("Generating witness (async)...")
            start_time = time.time()
            try:
                await ezkl.gen_witness(
                    data=json_path, model=circuit_name, output=witness_file
                )
            except RuntimeError as e:
                print(f"Failed to generate witness: {e}")
                continue

            if not ezkl.mock(witness_file, circuit_name):
                print("Mock run failed, skipping.")
                continue

            end_time = time.time()
            if verbose:
                print(f"Witness gen took {end_time - start_time:.2f} sec")
            total_witness_time += end_time - start_time
            # 4) prove
            if verbose:
                print("Generating proof...")
            start_time = time.time()
            prove_ok = ezkl.prove(
                witness_file, circuit_name, pk_file, proof_file, "single", srs_file
            )
            end_time = time.time()
            if verbose:
                print(f"Proof gen took {end_time - start_time:.2f} sec")
            total_prove_time += end_time - start_time

            if not prove_ok:
                print(f"Proof generation failed for {base_name}")
                continue

            if verbose:
                print(f"Done with {base_name}.\n")
            os.remove(pk_file)
            count_onnx_files += 1
        elif zk_backend == "plonky3":
            pass
        else:
            raise ValueError(f"Invalid ZK backend: {zk_backend}")

    return (
        total_settings_time,
        total_witness_time,
        total_prove_time,
        total_params,
        count_onnx_files,
    )
