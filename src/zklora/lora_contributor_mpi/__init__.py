import socket
import threading
import os
import math
from fractions import Fraction

import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

from ..zk_proof_generator import _prover_backend, generate_proofs
from ..base_model_user_mpi import _recv_json_message, _send_json_message
from ..proof_contract import (
    FixedPointConfig,
    InvocationWitness,
    adapter_manifest_entry,
    compute_delta_quantized,
    flatten,
    quantize_nested,
    write_adapter_manifest,
)
from ..proof_v3 import (
    adapter_manifest_entry_v3,
    adapter_manifest_payload_v3,
    ensure_secret_outside_artifacts,
    load_or_create_contributor_secret,
    resolve_contributor_secret_path,
    write_adapter_manifest_v3,
)


def read_file_as_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def strip_prefix(raw_name: str) -> str:
    """
    Remove 'base_model.model.', 'base_model.', 'model.' from the submodule name.
    Example:
      'base_model.model.transformer.h.0.attn.c_attn' => 'transformer.h.0.attn.c_attn'
    """
    name2 = raw_name
    for pfx in ["base_model.model.", "base_model.", "model."]:
        if name2.startswith(pfx):
            name2 = name2[len(pfx) :]
    return name2.strip()


class LoRAServer:
    def __init__(
        self,
        base_model_name: str,
        lora_model_id: str,
        out_dir: str,
        fixed_point: FixedPointConfig | None = None,
        manifest_secret_path: str | None = None,
    ):
        self.out_dir = out_dir
        self.fixed_point = fixed_point or FixedPointConfig()
        os.makedirs(self.out_dir, exist_ok=True)
        # Contributor secret seed for hiding adapter commitments. It must never
        # live inside out_dir, which is handed to the verifier as-is.
        self.manifest_secret_path = resolve_contributor_secret_path(
            manifest_secret_path
        )
        ensure_secret_outside_artifacts(self.manifest_secret_path, self.out_dir)
        self._manifest_entries: list | None = None
        self._manifest_payload: dict | None = None

        # 1) Load model, disable cache => no 'past_key_values'
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        base_model.config.use_cache = False

        base_model.eval()

        # 2) Load LoRA
        self.peft_model = PeftModel.from_pretrained(base_model, lora_model_id)
        self.peft_model.eval()

        # 3) Build submodule dict for actual LoRA submodules, e.g. 'transformer.h.0.attn.c_attn'
        self.submodules = {}
        for raw_name, module in self.peft_model.named_modules():
            if any("lora" in pname.lower() for pname, _ in module.named_parameters()):
                sname = strip_prefix(raw_name)
                # skip if empty or doesn't contain '.' or doesn't end in c_attn
                if not sname or "." not in sname:
                    continue
                if not sname.endswith("c_attn"):
                    continue
                self.submodules[sname] = module

        self.session_data: dict[str, list[InvocationWitness]] = {}
        self._invocation_counts: dict[tuple[str, str], int] = {}
        self.last_scaling: tuple[int, int] = (1, 1)
        self.last_q_delta: list[list[int]] = []

    def list_lora_injection_points(self):
        return list(self.submodules.keys())

    def adapter_manifest_entries(self):
        legacy = _prover_backend() == "legacy-halo2"
        if not legacy and self._manifest_entries is not None:
            # Schema-3 entries carry a fresh commitment nonce; the same entries
            # must back both the pinned manifest and every later proof, so they
            # are built once per server run.
            return self._manifest_entries
        secret = None
        if not legacy:
            secret = load_or_create_contributor_secret(self.manifest_secret_path)
        entries = []
        for sub_name, module in self.submodules.items():
            a_matrix, b_matrix, scaling_num, scaling_den = lora_matrices_and_scaling(
                module
            )
            a_quantized = quantize_nested(
                a_matrix.detach().cpu().numpy().tolist(), self.fixed_point
            )
            b_quantized = quantize_nested(
                b_matrix.detach().cpu().numpy().tolist(), self.fixed_point
            )
            if legacy:
                entries.append(
                    adapter_manifest_entry(
                        sub_name,
                        a_quantized,
                        b_quantized,
                        scaling_num,
                        scaling_den,
                        self.fixed_point,
                    )
                )
            else:
                entries.append(
                    adapter_manifest_entry_v3(
                        sub_name,
                        a_quantized,
                        b_quantized,
                        scaling_num,
                        scaling_den,
                        self.fixed_point,
                        secret,
                    )
                )
        if not legacy:
            self._manifest_entries = entries
        return entries

    def write_adapter_manifest(self, path: str):
        entries = self.adapter_manifest_entries()
        if _prover_backend() == "legacy-halo2":
            write_adapter_manifest(path, entries)
            self._manifest_payload = None
        else:
            self._manifest_payload = write_adapter_manifest_v3(path, entries)

    def apply_lora(
        self,
        sub_name: str,
        input_tensor: torch.Tensor,
        session_id: str | None = None,
    ):
        if sub_name not in self.submodules:
            raise ValueError(f"[LoRAServer] submodule '{sub_name}' not recognized.")
        mod = self.submodules[sub_name]
        print(f"[A] apply_lora on '{sub_name}', shape={list(input_tensor.shape)}")
        with torch.no_grad():
            delta_float, a_matrix, b_matrix, scaling_num, scaling_den = (
                compute_lora_delta(mod, input_tensor)
            )
        self.last_scaling = (int(scaling_num), int(scaling_den))
        self.last_q_delta = []

        sid = session_id or "default-session"
        key = (sid, sub_name)
        x_rows = input_tensor.detach().cpu().float().reshape(-1, int(a_matrix.shape[1]))
        delta_rows = []
        a_quantized = quantize_nested(
            a_matrix.detach().cpu().numpy().tolist(), self.fixed_point
        )
        b_quantized = quantize_nested(
            b_matrix.detach().cpu().numpy().tolist(), self.fixed_point
        )
        for x_row in x_rows:
            q_x = flatten(
                quantize_nested(x_row.cpu().numpy().tolist(), self.fixed_point)
            )
            q_delta = compute_delta_quantized(
                a_quantized,
                b_quantized,
                q_x,
                scaling_num,
                scaling_den,
                self.fixed_point,
            )
            delta_rows.append([int(v) for v in q_delta])
            invocation_index = self._invocation_counts.get(key, 0)
            self._invocation_counts[key] = invocation_index + 1
            witness = InvocationWitness(
                session_id=sid,
                module_name=sub_name,
                invocation_index=invocation_index,
                input_shape=[int(a_matrix.shape[1])],
                output_shape=[int(b_matrix.shape[0])],
                x=q_x,
                delta=q_delta,
                a=a_quantized,
                b=b_quantized,
                scaling_num=scaling_num,
                scaling_den=scaling_den,
                adapter_metadata={
                    "rank": int(a_matrix.shape[0]),
                    "in_dim": int(a_matrix.shape[1]),
                    "out_dim": int(b_matrix.shape[0]),
                    "source": "peft-linear-lora",
                },
                fixed_point=self.fixed_point,
            )
            self.session_data.setdefault(sid, []).append(witness)
        self.last_q_delta = [row[:] for row in delta_rows]
        quantized_delta = torch.tensor(delta_rows, dtype=torch.float64) / float(
            self.fixed_point.scale
        )
        return quantized_delta.reshape(delta_float.shape)

    def finalize_proofs_and_collect(self, session_id: str | None = None):
        """
        Generates native zkLoRA proof artifacts for captured LoRA invocations.
        """
        print(f"[A] finalize_proofs_and_collect => native artifacts => {self.out_dir}")
        if session_id is None:
            records = [
                record for values in self.session_data.values() for record in values
            ]
            self.session_data.clear()
        else:
            records = self.session_data.pop(session_id, [])

        if _prover_backend() == "legacy-halo2":
            proof_res = generate_proofs(
                records=records,
                output_dir=self.out_dir,
                verbose=True,
            )
        else:
            if self._manifest_payload is None:
                # Pin the same entries the verifier will receive out-of-band.
                self._manifest_payload = adapter_manifest_payload_v3(
                    self.adapter_manifest_entries()
                )
            proof_res = generate_proofs(
                records=records,
                output_dir=self.out_dir,
                verbose=True,
                adapter_manifest=self._manifest_payload,
                manifest_secret_path=self.manifest_secret_path,
            )

        if not proof_res:
            print("[A] No proofs generated or something went wrong.")
        else:
            print("[A] Proof generation done.")

        return


class LoRAServerSocket(threading.Thread):
    def __init__(
        self,
        host,
        port,
        lora_server: LoRAServer,
        stop_event,
        stop_timeout: float = 1200.0,
    ):
        super().__init__()
        self.host = host
        self.port = port
        self.lora_server = lora_server
        self.stop_event = stop_event
        self.stop_timeout = stop_timeout

    def run(self):
        print(f"[A-Server] listening on {self.host}:{self.port}")
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind((self.host, self.port))
        srv.listen(5)
        srv.settimeout(self.stop_timeout)

        print(
            f"[A-Server] Running on {self.host}:{self.port}, local artifacts in '{self.lora_server.out_dir}'"
        )
        try:
            while not self.stop_event.is_set():
                try:
                    conn, addr = srv.accept()
                except socket.timeout:
                    continue
                self.handle_conn(conn, addr)
        finally:
            srv.close()
            print("[A-Server] shutting down...")

    def handle_conn(self, conn, addr):
        try:
            req = self.recv_message(conn)
            if req is None:
                return
            rtype = req.get("request_type", "lora_forward")

            if rtype == "init_request":
                submods = self.lora_server.list_lora_injection_points()
                resp = {"response_type": "init_response", "injection_points": submods}

            elif rtype == "lora_forward":
                sname = req["submodule_name"]
                arr = req["input_array"]
                session_id = req.get("session_id")
                tin = torch.tensor(arr, dtype=torch.float32)
                out = self.lora_server.apply_lora(sname, tin, session_id=session_id)
                resp = {
                    "response_type": "lora_forward_response",
                    "output_array": out.cpu().numpy(),
                    "q_delta": self.lora_server.last_q_delta,
                    "scaling_num": int(self.lora_server.last_scaling[0]),
                    "scaling_den": int(self.lora_server.last_scaling[1]),
                }

            elif rtype == "end_inference":
                self.lora_server.finalize_proofs_and_collect(
                    session_id=req.get("session_id")
                )
                resp = {
                    "response_type": "end_inference_ack",
                    "message": "A finished native zkLoRA proof generation locally.",
                }

            else:
                resp = {"error": f"Unknown request_type {rtype}"}

            _send_json_message(conn, resp)
        except Exception as e:
            print(f"[A-Server] error: {e}")
        finally:
            conn.close()

    def recv_message(self, conn):
        conn.settimeout(1200.0)
        return _recv_json_message(conn)


def compute_lora_delta(module, input_tensor: torch.Tensor):
    """Return the canonical LoRA delta and matrices for a PEFT linear LoRA module."""

    a_matrix, b_matrix, scaling_num, scaling_den = lora_matrices_and_scaling(module)
    scaling = scaling_num / scaling_den

    x = input_tensor.float()
    delta = torch.matmul(torch.matmul(x, a_matrix.t()), b_matrix.t()) * scaling
    return delta, a_matrix, b_matrix, scaling_num, scaling_den


def lora_matrices_and_scaling(module):
    if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
        raise ValueError("module does not expose lora_A/lora_B")

    adapter_names = list(module.lora_A.keys()) if hasattr(module.lora_A, "keys") else []
    if not adapter_names:
        raise ValueError("module has no LoRA adapter weights")
    adapter = adapter_names[0]

    a_mod = module.lora_A[adapter]
    b_mod = module.lora_B[adapter]
    a_matrix = a_mod.weight.detach().cpu().float()
    b_matrix = b_mod.weight.detach().cpu().float()

    scaling_num, scaling_den = scaling_rational(module, adapter, int(a_matrix.shape[0]))
    return a_matrix, b_matrix, scaling_num, scaling_den


def scaling_rational(module, adapter: str, rank: int) -> tuple[int, int]:
    if hasattr(module, "lora_alpha"):
        alpha_source = module.lora_alpha
        alpha = (
            alpha_source.get(adapter, None)
            if isinstance(alpha_source, dict)
            else alpha_source
        )
        if alpha is not None:
            numerator, denominator = int(alpha), int(rank)
            divisor = math.gcd(numerator, denominator)
            return numerator // divisor, denominator // divisor

    scaling = 1.0
    if hasattr(module, "scaling"):
        if isinstance(module.scaling, dict):
            scaling = float(module.scaling.get(adapter, 1.0))
        else:
            scaling = float(module.scaling)
    fraction = Fraction(str(scaling)).limit_denominator()
    return fraction.numerator, fraction.denominator
