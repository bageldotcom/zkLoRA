from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable


BACKEND_ID = "zklora-halo2-v1"
SCHEMA_VERSION = 1
COMMITMENT_SCHEME = "sha256-canonical-lora-matrices-v1"
NATIVE_COMMITMENT_SCHEME = "field-linear-lora-matrices-v1"
INVOCATION_STRATEGY = "one-proof-per-module-invocation-v1"


class ProofContractError(ValueError):
    """Raised when a transcript, witness, or proof artifact violates the contract."""


@dataclass(frozen=True)
class FixedPointConfig:
    scale_bits: int = 20
    value_bits: int = 63
    intermediate_bits: int = 127

    @property
    def scale(self) -> int:
        return 1 << self.scale_bits

    @property
    def value_bound(self) -> int:
        return (1 << (self.value_bits - 1)) - 1

    @property
    def intermediate_bound(self) -> int:
        return (1 << (self.intermediate_bits - 1)) - 1


@dataclass(frozen=True)
class TranscriptEntry:
    session_id: str
    module_name: str
    invocation_index: int
    input_shape: list[int]
    output_shape: list[int]
    x: list[int]
    delta: list[int]
    fixed_point: FixedPointConfig
    scaling_num: int
    scaling_den: int

    def key(self) -> tuple[str, str, int]:
        return (self.session_id, self.module_name, self.invocation_index)

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["fixed_point"] = asdict(self.fixed_point)
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "TranscriptEntry":
        fp = FixedPointConfig(**data["fixed_point"])
        return cls(
            session_id=data["session_id"],
            module_name=data["module_name"],
            invocation_index=int(data["invocation_index"]),
            input_shape=[int(v) for v in data["input_shape"]],
            output_shape=[int(v) for v in data["output_shape"]],
            x=[int(v) for v in data["x"]],
            delta=[int(v) for v in data["delta"]],
            fixed_point=fp,
            scaling_num=int(data["scaling_num"]),
            scaling_den=int(data["scaling_den"]),
        )


@dataclass(frozen=True)
class InvocationWitness:
    session_id: str
    module_name: str
    invocation_index: int
    input_shape: list[int]
    output_shape: list[int]
    x: list[int]
    delta: list[int]
    a: list[list[int]]
    b: list[list[int]]
    scaling_num: int = 1
    scaling_den: int = 1
    adapter_metadata: dict[str, Any] | None = None
    fixed_point: FixedPointConfig = FixedPointConfig()

    @property
    def rank(self) -> int:
        return len(self.a)

    @property
    def in_dim(self) -> int:
        return len(self.a[0]) if self.a else 0

    @property
    def out_dim(self) -> int:
        return len(self.b)


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_json_bytes(data: Any) -> bytes:
    return canonical_json(data).encode("utf-8")


def digest_hex(data: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(data)).hexdigest()


def _round_decimal_away_from_zero(value: Decimal) -> int:
    if value.is_zero():
        return 0
    sign = -1 if value.is_signed() else 1
    magnitude = abs(value).to_integral_value(rounding=ROUND_HALF_UP)
    return sign * int(magnitude)


def quantize_scalar(
    value: int | float | str | Decimal, config: FixedPointConfig
) -> int:
    scaled = Decimal(str(value)) * Decimal(config.scale)
    result = _round_decimal_away_from_zero(scaled)
    _check_range(result, config.value_bound, "quantized value")
    return result


def quantize_nested(values: Any, config: FixedPointConfig) -> Any:
    if isinstance(values, (list, tuple)):
        return [quantize_nested(v, config) for v in values]
    return quantize_scalar(values, config)


def flatten(values: Any) -> list[int]:
    if isinstance(values, (list, tuple)):
        out: list[int] = []
        for value in values:
            out.extend(flatten(value))
        return out
    return [int(values)]


def shape_of(values: Any) -> list[int]:
    shape: list[int] = []
    cursor = values
    while isinstance(cursor, (list, tuple)):
        shape.append(len(cursor))
        cursor = cursor[0] if cursor else []
    return shape


def _check_range(value: int, bound: int, label: str) -> None:
    if value < -bound or value > bound:
        raise ProofContractError(f"{label} {value} exceeds signed bound +/-{bound}")


def _div_round_away_from_zero(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ProofContractError("denominator must be positive")
    sign = -1 if numerator < 0 else 1
    magnitude = abs(numerator)
    quotient, remainder = divmod(magnitude, denominator)
    if remainder * 2 >= denominator:
        quotient += 1
    return sign * quotient


def _rescale(value: int, config: FixedPointConfig) -> int:
    _check_range(value, config.intermediate_bound, "intermediate")
    return _div_round_away_from_zero(value, config.scale)


def validate_matrix(matrix: list[list[int]], rows: int, cols: int, label: str) -> None:
    if len(matrix) != rows:
        raise ProofContractError(f"{label} expected {rows} rows, got {len(matrix)}")
    for row in matrix:
        if len(row) != cols:
            raise ProofContractError(f"{label} expected {cols} columns, got {len(row)}")


def compute_delta_quantized(
    a: list[list[int]],
    b: list[list[int]],
    x: list[int],
    scaling_num: int,
    scaling_den: int,
    config: FixedPointConfig,
) -> list[int]:
    if scaling_den <= 0:
        raise ProofContractError("scaling_den must be positive")
    rank = len(a)
    if rank == 0:
        raise ProofContractError("rank must be positive")
    in_dim = len(a[0])
    out_dim = len(b)
    validate_matrix(a, rank, in_dim, "A")
    validate_matrix(b, out_dim, rank, "B")
    if len(x) != in_dim:
        raise ProofContractError(f"x expected length {in_dim}, got {len(x)}")

    for label, values in (("x", x), ("A", flatten(a)), ("B", flatten(b))):
        for value in values:
            _check_range(int(value), config.value_bound, label)

    intermediate: list[int] = []
    for row in a:
        raw = sum(int(weight) * int(x_i) for weight, x_i in zip(row, x))
        intermediate.append(_rescale(raw, config))

    delta: list[int] = []
    for row in b:
        raw = sum(int(weight) * int(value) for weight, value in zip(row, intermediate))
        scaled = _rescale(raw, config)
        scaled = _div_round_away_from_zero(scaled * int(scaling_num), int(scaling_den))
        _check_range(scaled, config.value_bound, "delta")
        delta.append(scaled)
    return delta


def lora_commitment(
    a: list[list[int]], b: list[list[int]], adapter_metadata: dict[str, Any] | None
) -> str:
    payload = {
        "scheme": COMMITMENT_SCHEME,
        "a": a,
        "b": b,
        "adapter_metadata": adapter_metadata or {},
    }
    return digest_hex(payload)


def native_lora_commitment(a: list[list[int]], b: list[list[int]]) -> int:
    accumulator = 0
    coefficient = 1
    for value in flatten(a) + flatten(b):
        accumulator += coefficient * int(value)
        coefficient += 1
    return accumulator


def circuit_id(
    in_dim: int,
    rank: int,
    out_dim: int,
    fixed_point: FixedPointConfig,
) -> str:
    payload = {
        "backend": BACKEND_ID,
        "commitment_scheme": COMMITMENT_SCHEME,
        "invocation_strategy": INVOCATION_STRATEGY,
        "in_dim": in_dim,
        "rank": rank,
        "out_dim": out_dim,
        "fixed_point": asdict(fixed_point),
    }
    return digest_hex(payload)


def vk_fingerprint(circuit: str) -> str:
    return digest_hex({"backend": BACKEND_ID, "vk_for_circuit": circuit})


def statement_from_witness(witness: InvocationWitness) -> dict[str, Any]:
    expected = compute_delta_quantized(
        witness.a,
        witness.b,
        witness.x,
        witness.scaling_num,
        witness.scaling_den,
        witness.fixed_point,
    )
    if expected != witness.delta:
        raise ProofContractError(
            f"witness delta does not match fixed-point LoRA relation: {expected} != {witness.delta}"
        )

    circuit = circuit_id(
        witness.in_dim, witness.rank, witness.out_dim, witness.fixed_point
    )
    adapter_metadata = dict(witness.adapter_metadata or {})
    adapter_metadata.setdefault("rank", witness.rank)
    adapter_metadata.setdefault("in_dim", witness.in_dim)
    adapter_metadata.setdefault("out_dim", witness.out_dim)

    statement = {
        "schema_version": SCHEMA_VERSION,
        "backend": BACKEND_ID,
        "session_id": witness.session_id,
        "module_name": witness.module_name,
        "invocation_index": witness.invocation_index,
        "input_shape": witness.input_shape,
        "output_shape": witness.output_shape,
        "x": witness.x,
        "delta": witness.delta,
        "scaling": {"num": witness.scaling_num, "den": witness.scaling_den},
        "fixed_point": asdict(witness.fixed_point),
        "adapter_metadata": adapter_metadata,
        "lora_commitment": lora_commitment(witness.a, witness.b, adapter_metadata),
        "native_lora_commitment": {
            "scheme": NATIVE_COMMITMENT_SCHEME,
            "value": native_lora_commitment(witness.a, witness.b),
        },
        "circuit_id": circuit,
        "vk_fingerprint": vk_fingerprint(circuit),
        "commitment_scheme": COMMITMENT_SCHEME,
        "invocation_strategy": INVOCATION_STRATEGY,
    }
    return statement


def transcript_entry_from_statement(statement: dict[str, Any]) -> TranscriptEntry:
    return TranscriptEntry(
        session_id=statement["session_id"],
        module_name=statement["module_name"],
        invocation_index=int(statement["invocation_index"]),
        input_shape=[int(v) for v in statement["input_shape"]],
        output_shape=[int(v) for v in statement["output_shape"]],
        x=[int(v) for v in statement["x"]],
        delta=[int(v) for v in statement["delta"]],
        fixed_point=FixedPointConfig(**statement["fixed_point"]),
        scaling_num=int(statement["scaling"]["num"]),
        scaling_den=int(statement["scaling"]["den"]),
    )


def statement_matches_transcript(
    statement: dict[str, Any], transcript_entry: TranscriptEntry
) -> bool:
    return (
        transcript_entry_from_statement(statement).to_json()
        == transcript_entry.to_json()
    )


def module_slug(module_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", module_name).strip("._")
    return slug or "module"


def artifact_prefix(
    output_dir: str | os.PathLike[str], statement: dict[str, Any]
) -> Path:
    return (
        Path(output_dir)
        / f"{module_slug(statement['module_name'])}.{int(statement['invocation_index']):04d}"
    )


def _native_statement_json(statement: dict[str, Any]) -> str:
    payload = {
        "x": statement["x"],
        "delta": statement["delta"],
        "fixed_point": statement["fixed_point"],
        "rank": int(statement["adapter_metadata"]["rank"]),
        "scaling_num": int(statement["scaling"]["num"]),
        "scaling_den": int(statement["scaling"]["den"]),
        "lora_commitment": int(statement["native_lora_commitment"]["value"]),
    }
    return canonical_json(payload)


def _native_witness_json(witness: InvocationWitness) -> str:
    return canonical_json({"a": witness.a, "b": witness.b})


def _native_module():
    try:
        from zklora import _native_prover  # type: ignore

        return _native_prover
    except Exception:
        return None


def write_invocation_artifacts(
    output_dir: str | os.PathLike[str], witness: InvocationWitness
) -> dict[str, str]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    statement = statement_from_witness(witness)
    prefix = artifact_prefix(output_dir, statement)
    proof_path = Path(f"{prefix}.zklora.proof")
    vk_path = Path(f"{prefix}.zklora.vk")
    pk_path = Path(f"{prefix}.zklora.pk")
    statement_path = Path(f"{prefix}.zklora.statement.json")
    meta_path = Path(f"{prefix}.zklora.meta.json")

    native = _native_module()
    if native is None:
        raise ProofContractError(
            "native Halo2 prover is unavailable; build/install zklora with maturin"
        )
    proof_bytes = native.prove(
        _native_statement_json(statement), _native_witness_json(witness)
    )
    proof_kind = "native-halo2"
    vk = {
        "backend": BACKEND_ID,
        "circuit_id": statement["circuit_id"],
        "vk_fingerprint": statement["vk_fingerprint"],
    }
    pk = {
        "backend": BACKEND_ID,
        "circuit_id": statement["circuit_id"],
        "pk_fingerprint": digest_hex({"pk_for_circuit": statement["circuit_id"]}),
    }
    meta = {
        "schema_version": SCHEMA_VERSION,
        "backend": BACKEND_ID,
        "proof_file": proof_path.name,
        "statement_file": statement_path.name,
        "vk_file": vk_path.name,
        "pk_file": pk_path.name,
        "statement_digest": digest_hex(statement),
        "proof_digest": hashlib.sha256(proof_bytes).hexdigest(),
        "proof_kind": proof_kind,
        "vk_digest": digest_hex(vk),
        "circuit_id": statement["circuit_id"],
        "vk_fingerprint": statement["vk_fingerprint"],
    }

    proof_path.write_bytes(proof_bytes)
    vk_path.write_text(canonical_json(vk) + "\n", encoding="utf-8")
    pk_path.write_text(canonical_json(pk) + "\n", encoding="utf-8")
    statement_path.write_text(canonical_json(statement) + "\n", encoding="utf-8")
    meta_path.write_text(canonical_json(meta) + "\n", encoding="utf-8")
    return {
        "proof": str(proof_path),
        "vk": str(vk_path),
        "pk": str(pk_path),
        "statement": str(statement_path),
        "meta": str(meta_path),
    }


def load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_transcript(
    path_or_entries: str | os.PathLike[str] | Iterable[Any],
) -> list[TranscriptEntry]:
    if isinstance(path_or_entries, (str, os.PathLike)):
        with open(path_or_entries, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = list(path_or_entries)

    if isinstance(data, dict) and "entries" in data:
        data = data["entries"]
    entries: list[TranscriptEntry] = []
    for item in data:
        if isinstance(item, TranscriptEntry):
            entries.append(item)
        else:
            entries.append(TranscriptEntry.from_json(item))
    return entries


def write_transcript(
    path: str | os.PathLike[str], entries: Iterable[TranscriptEntry]
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "entries": [entry.to_json() for entry in entries],
    }
    Path(path).write_text(canonical_json(payload) + "\n", encoding="utf-8")


def verify_artifact_set(
    statement_path: str | os.PathLike[str],
    transcript_entries: Iterable[TranscriptEntry],
) -> None:
    statement = load_json(statement_path)
    statement_path = Path(statement_path)
    prefix_text = str(statement_path)
    suffix = ".zklora.statement.json"
    if not prefix_text.endswith(suffix):
        raise ProofContractError(
            f"unexpected statement artifact name: {statement_path}"
        )
    prefix = prefix_text[: -len(suffix)]
    proof_path = Path(f"{prefix}.zklora.proof")
    vk_path = Path(f"{prefix}.zklora.vk")
    meta_path = Path(f"{prefix}.zklora.meta.json")

    transcript_index = {entry.key(): entry for entry in transcript_entries}
    entry = transcript_index.get(
        (
            statement["session_id"],
            statement["module_name"],
            int(statement["invocation_index"]),
        )
    )
    if entry is None:
        raise ProofContractError("statement is missing from verifier transcript")
    if not statement_matches_transcript(statement, entry):
        raise ProofContractError("statement does not match verifier transcript")

    expected_circuit = circuit_id(
        len(statement["x"]),
        int(statement["adapter_metadata"].get("rank", 0)),
        len(statement["delta"]),
        FixedPointConfig(**statement["fixed_point"]),
    )
    if statement["circuit_id"] != expected_circuit:
        raise ProofContractError("statement circuit_id does not match expected circuit")
    if statement["vk_fingerprint"] != vk_fingerprint(expected_circuit):
        raise ProofContractError(
            "statement vk_fingerprint does not match expected circuit"
        )
    native_commitment = statement.get("native_lora_commitment", {})
    if native_commitment.get("scheme") != NATIVE_COMMITMENT_SCHEME:
        raise ProofContractError("statement native LoRA commitment scheme mismatch")

    vk = load_json(vk_path)
    if vk.get("circuit_id") != expected_circuit:
        raise ProofContractError("verification key circuit_id mismatch")
    if vk.get("vk_fingerprint") != statement["vk_fingerprint"]:
        raise ProofContractError("verification key fingerprint mismatch")

    meta = load_json(meta_path)
    proof_bytes = proof_path.read_bytes()
    if meta.get("proof_kind") != "native-halo2":
        raise ProofContractError("unsupported proof kind; expected native-halo2")
    native = _native_module()
    if native is None:
        raise ProofContractError("native Halo2 verifier is unavailable")
    if not native.verify(_native_statement_json(statement), proof_bytes):
        raise ProofContractError("proof bytes failed native Halo2 verification")

    if meta.get("statement_digest") != digest_hex(statement):
        raise ProofContractError("metadata statement digest mismatch")
    if meta.get("proof_digest") != hashlib.sha256(proof_bytes).hexdigest():
        raise ProofContractError("metadata proof digest mismatch")
    if meta.get("vk_fingerprint") != statement["vk_fingerprint"]:
        raise ProofContractError("metadata vk fingerprint mismatch")


def verify_artifacts(
    proof_dir: str | os.PathLike[str],
    transcript: str | os.PathLike[str] | Iterable[Any],
) -> tuple[float, int]:
    import time

    start = time.time()
    entries = load_transcript(transcript)
    statement_files = sorted(Path(proof_dir).glob("*.zklora.statement.json"))
    seen = set()
    for statement_file in statement_files:
        statement = load_json(statement_file)
        key = (
            statement["session_id"],
            statement["module_name"],
            int(statement["invocation_index"]),
        )
        if key in seen:
            raise ProofContractError(f"duplicate proof statement for {key}")
        seen.add(key)
        verify_artifact_set(statement_file, entries)

    expected = {entry.key() for entry in entries}
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ProofContractError(
            f"proof transcript coverage mismatch missing={missing} extra={extra}"
        )
    return time.time() - start, len(statement_files)
