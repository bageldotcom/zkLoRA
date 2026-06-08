import json
import math
import socket
import uuid
from typing import Any

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


_JSON_LENGTH_BYTES = 8
_MAX_JSON_MESSAGE_BYTES = 64 * 1024 * 1024


def _json_ready(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        return _json_ready(value.tolist())
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _send_json_message(sock: socket.socket, message: dict[str, Any]) -> None:
    payload = json.dumps(
        _json_ready(message),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(payload) > _MAX_JSON_MESSAGE_BYTES:
        raise RuntimeError("JSON message exceeds maximum frame size.")
    sock.sendall(len(payload).to_bytes(_JSON_LENGTH_BYTES, "big") + payload)


def _recv_exact(
    sock: socket.socket, size: int, allow_empty: bool = False
) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        try:
            chunk = sock.recv(remaining)
        except socket.timeout as exc:
            raise RuntimeError("Timed out while reading JSON message.") from exc
        if not chunk:
            if allow_empty and remaining == size:
                return None
            raise RuntimeError("Unexpected EOF while reading JSON message.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_json_message(sock: socket.socket) -> dict[str, Any] | None:
    header = _recv_exact(sock, _JSON_LENGTH_BYTES, allow_empty=True)
    if header is None:
        return None
    size = int.from_bytes(header, "big")
    if size <= 0:
        raise RuntimeError("Received empty JSON message.")
    if size > _MAX_JSON_MESSAGE_BYTES:
        raise RuntimeError("Received JSON message exceeds maximum frame size.")
    payload = _recv_exact(sock, size)
    if payload is None:
        raise RuntimeError("Unexpected EOF while reading JSON payload.")
    try:
        return json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("Received invalid JSON message.") from exc


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
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.host_a, self.port_a))
            _send_json_message(s, data_dict)
            s.shutdown(socket.SHUT_WR)

            s.settimeout(1200.0)  # give more time if proof generation is slow
            resp = _recv_json_message(s)

        if resp is None:
            raise RuntimeError(
                "[B] No data from A (EOF). Possibly A took too long or closed early."
            )
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
            q_delta = remote_resp.get("q_delta")
            scaling_num = int(remote_resp.get("scaling_num", 1))
            scaling_den = int(remote_resp.get("scaling_den", 1))
        else:
            remote_out = remote_resp
            q_delta = None
            scaling_num = (
                self.transcript_recorder.scaling_num if self.transcript_recorder else 1
            )
            scaling_den = (
                self.transcript_recorder.scaling_den if self.transcript_recorder else 1
            )
        if q_delta is not None and self.transcript_recorder is not None:
            out_t = _dequantize_q_delta(
                q_delta,
                self.transcript_recorder.fixed_point,
                tuple(base_out.shape),
                base_out.device,
                base_out.dtype if torch.is_floating_point(base_out) else torch.float32,
            )
            remote_out_for_record = out_t.detach().cpu().numpy()
        elif self.transcript_recorder is not None:
            raise RuntimeError(
                f"[B] submodule '{self.sub_name}' => proof-bound response missing q_delta."
            )
        else:
            if remote_out is None:
                raise RuntimeError(
                    f"[B] submodule '{self.sub_name}' => no output from A."
                )
            out_t = torch.tensor(
                remote_out, dtype=torch.float32, device=base_out.device
            )
            remote_out_for_record = remote_out
        if self.transcript_recorder is not None:
            self.transcript_recorder.record(
                self.sub_name,
                arr,
                remote_out_for_record,
                scaling_num=scaling_num,
                scaling_den=scaling_den,
                q_delta_values=q_delta,
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
        q_delta_values=None,
    ):
        x_rows = _canonical_rows(x_values)
        delta_rows = _canonical_rows(delta_values)
        if len(x_rows) != len(delta_rows):
            raise ValueError(
                f"transcript row mismatch for {module_name}: "
                f"{len(x_rows)} inputs vs {len(delta_rows)} deltas"
            )
        if q_delta_values is None:
            q_delta_rows = [
                flatten(quantize_nested(delta_row, self.fixed_point))
                for delta_row in delta_rows
            ]
        else:
            q_delta_rows = _canonical_int_rows(q_delta_values)
        if len(x_rows) != len(q_delta_rows):
            raise ValueError(
                f"transcript row mismatch for {module_name}: "
                f"{len(x_rows)} inputs vs {len(q_delta_rows)} q_deltas"
            )
        entries: list[TranscriptEntry] = []
        for x_row, q_delta in zip(x_rows, q_delta_rows):
            invocation_index = self._counts.get(module_name, 0)
            self._counts[module_name] = invocation_index + 1
            q_x = flatten(quantize_nested(x_row, self.fixed_point))
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


def _canonical_int_rows(values):
    values = _to_list(values)
    _assert_exact_int_values(values)
    tensor = torch.as_tensor(values, dtype=torch.int64)
    if tensor.numel() == 0:
        return []
    if tensor.ndim == 0:
        return [[int(tensor.item())]]
    if tensor.ndim == 1:
        return [[int(v) for v in tensor.cpu().numpy().tolist()]]
    return [
        [int(v) for v in row]
        for row in tensor.reshape(-1, tensor.shape[-1]).cpu().numpy().tolist()
    ]


def _assert_exact_int_values(values):
    if isinstance(values, bool):
        raise ValueError("q_delta values must be integers, not booleans")
    if isinstance(values, int):
        return
    if isinstance(values, (list, tuple)):
        for value in values:
            _assert_exact_int_values(value)
        return
    raise ValueError(f"q_delta values must be integers, got {type(values).__name__}")


def _dequantize_q_delta(
    q_delta_values,
    fixed_point: FixedPointConfig,
    target_shape: tuple[int, ...],
    device,
    dtype,
):
    q_delta_rows = _canonical_int_rows(q_delta_values)
    if not q_delta_rows:
        raise RuntimeError("Received empty q_delta for proof-bound LoRA response.")
    q_delta = torch.tensor(q_delta_rows, dtype=torch.float64)
    expected_rows = math.prod(target_shape[:-1]) if target_shape else 1
    expected_cols = target_shape[-1] if target_shape else 1
    if list(q_delta.shape) != [expected_rows, expected_cols]:
        raise RuntimeError(
            "q_delta shape does not match local module output rows: "
            f"{list(q_delta.shape)} != {[expected_rows, expected_cols]}"
        )
    expected = math.prod(target_shape)
    if q_delta.numel() != expected:
        raise RuntimeError(
            "q_delta shape does not match local module output: "
            f"{list(q_delta.shape)} cannot reshape to {list(target_shape)}"
        )
    return (
        (q_delta / float(fixed_point.scale))
        .reshape(target_shape)
        .to(device=device, dtype=dtype)
    )


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
        patched_modules: set[str] = set()
        for comm in self.comms:
            submods = comm.init_request()
            print("[B] injection points =>", submods)
            for full_name in submods:
                if not full_name.strip():
                    print("[B] skipping empty submodule name.")
                    continue
                if full_name in patched_modules:
                    raise RuntimeError(
                        "duplicate LoRA injection point from contributors is unsupported: "
                        f"{full_name}"
                    )
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
                    patched_modules.add(full_name)
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
