import socket
import pickle
import uuid

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..proof_contract import (
    FixedPointConfig,
    TranscriptEntry,
    flatten,
    quantize_nested,
    write_transcript,
)


class BaseModelToLoRAComm:
    def __init__(self, host_a="127.0.0.1", port_a=30000):
        self.host_a = host_a
        self.port_a = port_a

    def init_request(self):
        data = {"request_type": "init_request"}
        resp = self.send_and_recv(data)
        return resp.get("injection_points", [])

    def lora_forward(self, sub_name, arr, session_id=None, include_metadata=False):
        req = {
            "request_type": "lora_forward",
            "submodule_name": sub_name,
            "input_array": arr,
            "session_id": session_id,
        }
        resp = self.send_and_recv(req)
        if include_metadata:
            return resp
        return resp.get("output_array", None)

    def end_inference(self, session_id=None):
        req = {"request_type": "end_inference", "session_id": session_id}
        resp = self.send_and_recv(
            req
        )  # , timeout=600.0)  # might be slower if proof gen is big
        return resp

    def send_and_recv(self, data_dict):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.host_a, self.port_a))
        bin_req = pickle.dumps(data_dict)
        s.sendall(bin_req)
        s.shutdown(socket.SHUT_WR)

        buffer = b""
        s.settimeout(1200.0)  # give more time if proof generation is slow
        while True:
            try:
                chunk = s.recv(4096)
            except socket.timeout:
                break
            if not chunk:
                break
            buffer += chunk
        s.close()

        if not buffer:
            raise RuntimeError(
                "[B] No data from A (EOF). Possibly A took too long or closed early."
            )

        resp = pickle.loads(buffer)
        return resp


class RemoteLoRAWrappedModule(nn.Module):
    def __init__(
        self,
        sub_name,
        local_sub,
        comm: BaseModelToLoRAComm,
        combine_mode="replace",
        transcript_recorder=None,
    ):
        super().__init__()
        self.sub_name = sub_name
        self.local_sub = local_sub
        self.comm = comm
        self.combine_mode = combine_mode
        self.transcript_recorder = transcript_recorder

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            base_out = self.local_sub(x)
        arr = x.cpu().numpy()
        session_id = (
            self.transcript_recorder.session_id
            if self.transcript_recorder is not None
            else None
        )
        try:
            remote_resp = self.comm.lora_forward(
                self.sub_name, arr, session_id=session_id, include_metadata=True
            )
        except TypeError:
            remote_resp = self.comm.lora_forward(self.sub_name, arr)
        if isinstance(remote_resp, dict):
            remote_out = remote_resp.get("output_array")
            scaling_num = int(remote_resp.get("scaling_num", 1))
            scaling_den = int(remote_resp.get("scaling_den", 1))
        else:
            remote_out = remote_resp
            scaling_num = (
                self.transcript_recorder.scaling_num if self.transcript_recorder else 1
            )
            scaling_den = (
                self.transcript_recorder.scaling_den if self.transcript_recorder else 1
            )
        if remote_out is None:
            raise RuntimeError(f"[B] submodule '{self.sub_name}' => no output from A.")
        out_t = torch.tensor(remote_out, dtype=torch.float32)
        if self.transcript_recorder is not None:
            self.transcript_recorder.record(
                self.sub_name,
                arr,
                remote_out,
                scaling_num=scaling_num,
                scaling_den=scaling_den,
            )
        if self.combine_mode == "add_delta":
            return base_out + out_t
        return out_t


class ProofTranscriptRecorder:
    def __init__(
        self,
        session_id: str | None = None,
        fixed_point: FixedPointConfig | None = None,
        scaling_num: int = 1,
        scaling_den: int = 1,
    ):
        self.session_id = session_id or str(uuid.uuid4())
        self.fixed_point = fixed_point or FixedPointConfig()
        self.scaling_num = scaling_num
        self.scaling_den = scaling_den
        self.entries: list[TranscriptEntry] = []
        self._counts: dict[str, int] = {}

    def record(
        self,
        module_name,
        x_values,
        delta_values,
        scaling_num: int | None = None,
        scaling_den: int | None = None,
    ):
        x_rows = _canonical_rows(x_values)
        delta_rows = _canonical_rows(delta_values)
        if len(x_rows) != len(delta_rows):
            raise ValueError(
                f"transcript row mismatch for {module_name}: "
                f"{len(x_rows)} inputs vs {len(delta_rows)} deltas"
            )
        entries: list[TranscriptEntry] = []
        for x_row, delta_row in zip(x_rows, delta_rows):
            invocation_index = self._counts.get(module_name, 0)
            self._counts[module_name] = invocation_index + 1
            q_x = flatten(quantize_nested(x_row, self.fixed_point))
            q_delta = flatten(quantize_nested(delta_row, self.fixed_point))
            entry = TranscriptEntry(
                session_id=self.session_id,
                module_name=module_name,
                invocation_index=invocation_index,
                input_shape=[len(q_x)],
                output_shape=[len(q_delta)],
                x=q_x,
                delta=q_delta,
                fixed_point=self.fixed_point,
                scaling_num=int(
                    scaling_num if scaling_num is not None else self.scaling_num
                ),
                scaling_den=int(
                    scaling_den if scaling_den is not None else self.scaling_den
                ),
            )
            self.entries.append(entry)
            entries.append(entry)
        return entries[-1] if len(entries) == 1 else entries

    def write(self, path: str):
        write_transcript(path, self.entries)


def _to_list(values):
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    if hasattr(values, "tolist"):
        return values.tolist()
    return values


def _canonical_rows(values):
    tensor = torch.as_tensor(_to_list(values), dtype=torch.float32)
    if tensor.ndim == 0:
        return [[float(tensor.item())]]
    if tensor.ndim == 1:
        return [tensor.cpu().numpy().tolist()]
    return tensor.reshape(-1, tensor.shape[-1]).cpu().numpy().tolist()


class BaseModelClient:
    def __init__(
        self,
        base_model: str = "distilgpt2",
        host_a: str = "127.0.0.1",
        port_a: int = 30000,
        combine_mode: str = "replace",
        contributors: list[tuple[str, int]] | None = None,
        session_id: str | None = None,
        fixed_point: FixedPointConfig | None = None,
    ):
        """Client for interacting with one or more LoRA contributors."""
        self.model = AutoModelForCausalLM.from_pretrained(base_model)

        self.model.config.use_cache = False
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

        if contributors is None:
            contributors = [(host_a, port_a)]

        self.comms = [BaseModelToLoRAComm(h, p) for h, p in contributors]
        self.combine_mode = combine_mode
        self.transcript = ProofTranscriptRecorder(
            session_id=session_id, fixed_point=fixed_point
        )

    def _navigate(self, mod: nn.Module, parts: list[str]) -> nn.Module:
        """
        If a part is digits => mod=mod[int], else mod=getattr(mod, part).
        E.g. 'transformer','h','0','attn','c_attn' => indexing for '0'.
        """
        for p in parts:
            if p.isdigit():
                idx = int(p)
                mod = mod[idx]
            else:
                mod = getattr(mod, p)
        return mod

    def init_and_patch(self):
        """Query all contributors for injection points and patch the model."""
        for comm in self.comms:
            submods = comm.init_request()
            print("[B] injection points =>", submods)
            for full_name in submods:
                if not full_name.strip():
                    print("[B] skipping empty submodule name.")
                    continue
                try:
                    path_parts = full_name.split(".")
                    *parents, child = path_parts
                    m = self._navigate(self.model, parents)
                    orig_sub = getattr(m, child)
                    wrapped = RemoteLoRAWrappedModule(
                        full_name,
                        orig_sub,
                        comm,
                        self.combine_mode,
                        transcript_recorder=self.transcript,
                    )
                    setattr(m, child, wrapped)
                    print(
                        f"[B] Patched submodule '{full_name}' from {comm.host_a}:{comm.port_a}."
                    )
                except Exception as e:
                    print(f"[B] Could not patch '{full_name}': {e}")

    def forward_loss(self, text: str) -> float:
        enc = self.tokenizer(text, return_tensors="pt")
        in_ids = enc["input_ids"]
        with torch.no_grad():
            out = self.model(in_ids, labels=in_ids)
        return out.loss.item()

    def end_inference(self):
        """Notify all contributors that inference is finished."""
        for comm in self.comms:
            try:
                resp = comm.end_inference(session_id=self.transcript.session_id)
            except TypeError:
                resp = comm.end_inference()
            print(
                "[B] end_inference => got ack from", comm.host_a, comm.port_a, ":", resp
            )
