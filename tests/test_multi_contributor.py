import json
import socket
import threading
import types
from unittest.mock import patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from zklora.base_model_user_mpi import (  # noqa: E402
    BaseModelClient,
    BaseModelToLoRAComm,
    ProofTranscriptRecorder,
    RemoteLoRAWrappedModule,
)
from zklora.proof_contract import FixedPointConfig  # noqa: E402


def _read_exact(conn, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = conn.recv(remaining)
        if not chunk:
            raise RuntimeError("unexpected EOF in test socket")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _write_json(conn, message):
    payload = json.dumps(message, ensure_ascii=True, separators=(",", ":")).encode(
        "utf-8"
    )
    conn.sendall(len(payload).to_bytes(8, "big") + payload)


def _read_json(conn):
    header = _read_exact(conn, 8)
    payload = _read_exact(conn, int.from_bytes(header, "big"))
    return json.loads(payload.decode("utf-8"))


class DummySub(nn.Module):
    def forward(self, x):
        return x * 2


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sub1 = DummySub()
        self.sub2 = DummySub()
        self.config = types.SimpleNamespace(use_cache=True)

    def eval(self):
        pass

    def forward(self, input_ids, labels=None):
        out = types.SimpleNamespace(loss=torch.tensor(0.0))
        return out


class DummyTokenizer:
    def __call__(self, text, return_tensors=None):
        return {"input_ids": torch.tensor([[1]])}


class FakeComm(BaseModelToLoRAComm):
    def __init__(self, name):
        super().__init__("127.0.0.1", 0)
        self.name = name

    def init_request(self):
        return [self.name]

    def lora_forward(self, sub_name, arr):
        return arr + 1

    def end_inference(self):
        return {"ok": True}


class MetadataComm(BaseModelToLoRAComm):
    def __init__(self):
        super().__init__("127.0.0.1", 0)

    def lora_forward(self, sub_name, arr, session_id=None, include_metadata=False):
        return {
            "output_array": arr + 2,
            "scaling_num": 3,
            "scaling_den": 2,
        }


class QuantizedMetadataComm(MetadataComm):
    def __init__(self, q_delta=None, output_offset=2):
        super().__init__()
        self.q_delta = q_delta
        self.output_offset = output_offset

    def lora_forward(self, sub_name, arr, session_id=None, include_metadata=False):
        resp = super().lora_forward(
            sub_name, arr, session_id=session_id, include_metadata=include_metadata
        )
        resp["output_array"] = arr + self.output_offset
        resp["q_delta"] = (
            self.q_delta
            if self.q_delta is not None
            else (torch.as_tensor(arr, dtype=torch.int64) + 2).tolist()
        )
        return resp


def test_multi_contributor_patch():
    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained", return_value=DummyModel()
    ):
        with patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=DummyTokenizer()
        ):
            comm1 = FakeComm("sub1")
            comm2 = FakeComm("sub2")
            client = BaseModelClient(
                base_model="dummy",
                contributors=[("h1", 1), ("h2", 2)],
            )
            # replace created comms with our fake ones
            client.comms = [comm1, comm2]
            client.init_and_patch()

            assert isinstance(client.model.sub1, RemoteLoRAWrappedModule)
            assert isinstance(client.model.sub2, RemoteLoRAWrappedModule)
            assert client.model.sub1.comm is comm1
            assert client.model.sub2.comm is comm2


def test_remote_module_records_per_vector_transcript_with_scaling():
    recorder = ProofTranscriptRecorder(
        session_id="s1",
        fixed_point=FixedPointConfig(scale_bits=0, value_bits=16, intermediate_bits=32),
    )
    wrapped = RemoteLoRAWrappedModule(
        "sub1",
        DummySub(),
        QuantizedMetadataComm(),
        combine_mode="add_delta",
        transcript_recorder=recorder,
    )

    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    out = wrapped(x)

    assert torch.equal(out, x * 2 + x + 2)
    assert len(recorder.entries) == 2
    assert [entry.invocation_index for entry in recorder.entries] == [0, 1]
    assert all(entry.scaling_num == 3 for entry in recorder.entries)
    assert all(entry.scaling_den == 2 for entry in recorder.entries)
    assert recorder.entries[0].input_shape == [2]
    assert recorder.entries[0].output_shape == [2]


def test_base_comm_uses_length_prefixed_json_protocol():
    received = {}
    errors = []

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.settimeout(2.0)
    listener.listen(1)
    host, port = listener.getsockname()

    def serve_once():
        try:
            conn, _ = listener.accept()
            with conn:
                conn.settimeout(2.0)
                header = _read_exact(conn, 8)
                payload = _read_exact(conn, int.from_bytes(header, "big"))
                received["header"] = header
                received["payload"] = payload
                received["request"] = json.loads(payload.decode("utf-8"))
                _write_json(
                    conn,
                    {
                        "response_type": "ok",
                        "output_array": [1.0, 2.0],
                        "q_delta": [[11, 22]],
                    },
                )
        except Exception as exc:
            errors.append(exc)
        finally:
            listener.close()

    thread = threading.Thread(target=serve_once, daemon=True)
    thread.start()

    try:
        resp = BaseModelToLoRAComm(host, port).send_and_recv(
            {
                "request_type": "lora_forward",
                "submodule_name": "sub1",
                "input_array": torch.tensor([1.25, 2.5]),
            }
        )
    finally:
        listener.close()
        thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert not errors
    assert int.from_bytes(received["header"], "big") == len(received["payload"])
    assert received["payload"].startswith(b"{")
    assert not received["payload"].startswith(b"\x80")
    assert received["request"]["input_array"] == [1.25, 2.5]
    assert resp["q_delta"] == [[11, 22]]


def test_remote_module_records_transported_q_delta_exactly():
    q_delta = [[2**52 + 101, -(2**52 + 202)], [303, -404]]
    recorder = ProofTranscriptRecorder(
        session_id="s1",
        fixed_point=FixedPointConfig(
            scale_bits=0, value_bits=63, intermediate_bits=127
        ),
    )
    wrapped = RemoteLoRAWrappedModule(
        "sub1",
        DummySub(),
        QuantizedMetadataComm(q_delta=q_delta, output_offset=999),
        transcript_recorder=recorder,
    )

    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    out = wrapped(x)

    assert torch.equal(out, torch.tensor(q_delta, dtype=torch.float32).reshape_as(x))
    assert [entry.delta for entry in recorder.entries] == q_delta
    assert [entry.output_shape for entry in recorder.entries] == [[2], [2]]


def test_remote_module_requires_q_delta_for_proof_bound_forward():
    recorder = ProofTranscriptRecorder(session_id="s1")
    wrapped = RemoteLoRAWrappedModule(
        "sub1",
        DummySub(),
        MetadataComm(),
        transcript_recorder=recorder,
    )

    with pytest.raises(RuntimeError, match="missing q_delta"):
        wrapped(torch.tensor([[1.0, 2.0]]))


def test_remote_module_rejects_malformed_q_delta_before_recording():
    recorder = ProofTranscriptRecorder(
        session_id="s1",
        fixed_point=FixedPointConfig(scale_bits=0, value_bits=16, intermediate_bits=32),
    )
    wrapped = RemoteLoRAWrappedModule(
        "sub1",
        DummySub(),
        QuantizedMetadataComm(q_delta=[[1, 2, 3]]),
        transcript_recorder=recorder,
    )

    with pytest.raises(RuntimeError, match="q_delta shape"):
        wrapped(torch.tensor([[1.0, 2.0]]))
    assert recorder.entries == []

    wrapped.comm = QuantizedMetadataComm(q_delta=[[1.0, 2.0]])
    with pytest.raises(ValueError, match="q_delta values must be integers"):
        wrapped(torch.tensor([[1.0, 2.0]]))
    assert recorder.entries == []


def test_lora_server_socket_returns_json_q_delta():
    pytest.importorskip("peft")
    from zklora.lora_contributor_mpi import LoRAServerSocket  # noqa: E402

    class FakeLoRAServer:
        out_dir = "out"
        last_scaling = (5, 7)
        last_q_delta = [[123, -456]]

        def apply_lora(self, sub_name, input_tensor, session_id=None):
            self.seen = (sub_name, input_tensor.cpu().numpy().tolist(), session_id)
            return input_tensor + 0.5

    fake_server = FakeLoRAServer()
    client_sock, server_sock = socket.socketpair()
    client_sock.settimeout(2.0)
    stop_event = threading.Event()
    handler = LoRAServerSocket("127.0.0.1", 0, fake_server, stop_event)
    thread = threading.Thread(
        target=handler.handle_conn, args=(server_sock, "local"), daemon=True
    )
    thread.start()

    try:
        _write_json(
            client_sock,
            {
                "request_type": "lora_forward",
                "submodule_name": "sub1",
                "input_array": [[1.0, 2.0]],
                "session_id": "sess-1",
            },
        )
        client_sock.shutdown(socket.SHUT_WR)
        resp = _read_json(client_sock)
    finally:
        thread.join(timeout=2.0)
        client_sock.close()

    assert not thread.is_alive()
    assert fake_server.seen == ("sub1", [[1.0, 2.0]], "sess-1")
    assert resp["output_array"] == [[1.5, 2.5]]
    assert resp["q_delta"] == [[123, -456]]
    assert resp["scaling_num"] == 5
    assert resp["scaling_den"] == 7
