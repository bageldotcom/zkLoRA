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
    recorder = ProofTranscriptRecorder(session_id="s1")
    wrapped = RemoteLoRAWrappedModule(
        "sub1",
        DummySub(),
        MetadataComm(),
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
