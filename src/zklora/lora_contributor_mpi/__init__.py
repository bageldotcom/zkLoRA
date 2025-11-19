import socket
import threading
import pickle
import os
import asyncio

import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

# from zklora with the MPI exporter & proof generator
from ..zk_proof_generator import generate_proofs
from ..mpi_lora_onnx_exporter import export_lora_onnx_json_mpi


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
    def __init__(self, base_model_name: str, lora_model_id: str, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

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

        self.session_data = {}  # {sub_name => [np arrays]}

    def list_lora_injection_points(self):
        return list(self.submodules.keys())

    def apply_lora(self, sub_name: str, input_tensor: torch.Tensor):
        if sub_name not in self.submodules:
            raise ValueError(f"[LoRAServer] submodule '{sub_name}' not recognized.")
        mod = self.submodules[sub_name]
        print(f"[A] apply_lora on '{sub_name}', shape={list(input_tensor.shape)}")
        with torch.no_grad():
            out = mod(input_tensor)
        x_np = input_tensor.cpu().numpy()
        self.session_data.setdefault(sub_name, []).append(x_np)
        return out

    def finalize_proofs_and_collect(self):
        """
        Exports ONNX + JSON for each submodule used, then runs proofs, reads .pf + .vk, etc.
        """
        print(f"[A] finalize_proofs_and_collect => exporting => {self.out_dir}")
        for sname, arr_list in self.session_data.items():
            if not arr_list:
                continue
            last_in = arr_list[-1]
            mod = self.submodules[sname]
            export_lora_onnx_json_mpi(
                sub_name=sname,
                x_data=last_in,
                submodule=mod,
                output_dir=self.out_dir,
                verbose=True,
            )
        self.session_data.clear()

        # generate proofs synchronously
        print(
            "[A] Running generate_proofs(...) via asyncio.run(...) in the same thread."
        )
        proof_res = asyncio.run(
            generate_proofs(
                onnx_dir=self.out_dir,
                json_dir=self.out_dir,
                output_dir=self.out_dir,
                verbose=True,
            )
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
        import socket

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
            data = self.recv_all(conn)
            if not data:
                return
            req = pickle.loads(data)
            rtype = req.get("request_type", "lora_forward")

            if rtype == "init_request":
                submods = self.lora_server.list_lora_injection_points()
                resp = {"response_type": "init_response", "injection_points": submods}

            elif rtype == "lora_forward":
                sname = req["submodule_name"]
                arr = req["input_array"]
                tin = torch.tensor(arr, dtype=torch.float32)
                out = self.lora_server.apply_lora(sname, tin)
                resp = {
                    "response_type": "lora_forward_response",
                    "output_array": out.cpu().numpy(),
                }

            elif rtype == "end_inference":
                # generate proofs locally
                self.lora_server.finalize_proofs_and_collect()
                resp = {
                    "response_type": "end_inference_ack",
                    "message": "A finished proof generation locally. B can close.",
                }

            else:
                resp = {"error": f"Unknown request_type {rtype}"}

            conn.sendall(pickle.dumps(resp))
        except Exception as e:
            print(f"[A-Server] error: {e}")
        finally:
            conn.close()

    def recv_all(self, conn, chunk_size=4096):
        buffer = b""
        conn.settimeout(1200.0)
        while True:
            try:
                chunk = conn.recv(chunk_size)
            except socket.timeout:
                break
            if not chunk:
                break
            buffer += chunk
        return buffer
