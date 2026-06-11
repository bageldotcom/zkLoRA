from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import secrets
import threading
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable


# v4: commit-and-prove sigma backend over ristretto255. The adapter is bound
# once per manifest through salted Pedersen row commitments plus a one-time
# exact weight range proof; each invocation proof checks the same exact
# quantized LoRA relation as the v3 circuit through Fiat-Shamir random
# projections and aggregated range proofs, so per-proof work no longer scales
# with rank*in + out*rank weights. Statement semantics, transcript binding,
# and the verifier trust boundary are unchanged. v3 halo2 artifacts (schema
# version 2) remain verifiable through the legacy path below.
BACKEND_ID = "zklora-sigma-v4"
SCHEMA_VERSION = 3
PROOF_KIND = "native-sigma-v4"
COMMITMENT_SCHEME = "sha256-canonical-lora-matrices-v1"
ADAPTER_COMMITMENT_SCHEME = "pedersen-ristretto255-salted-v1"
INVOCATION_STRATEGY = "one-proof-per-module-invocation-v1"

LEGACY_BACKEND_ID = "zklora-halo2-v3"
LEGACY_SCHEMA_VERSION = 2
LEGACY_PROOF_KIND = "native-halo2"
LEGACY_ADAPTER_COMMITMENT_SCHEME = "poseidon-pasta-fp-adapter-v1"


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


def quantize_rows(rows: list[list[float]], config: FixedPointConfig) -> list[list[int]]:
    """Quantize flat float rows with exact quantize_scalar semantics.

    The native implementation reproduces Decimal(str(value)) round-half-up
    rounding using exact integer arithmetic over the shortest round-trip
    decimal representation; the Python path is the reference fallback.
    """
    if not rows:
        return []
    native = _native_module()
    if native is not None and hasattr(native, "quantize_rows"):
        try:
            # str(float) is the semantic anchor of quantize_scalar; passing the
            # strings keeps the native path bit-identical to Decimal(str(v)).
            return [
                [int(v) for v in row]
                for row in native.quantize_rows(
                    [[str(float(v)) for v in row] for row in rows],
                    config.scale_bits,
                    config.value_bits,
                )
            ]
        except (OverflowError, TypeError, ValueError):
            pass
    return [[quantize_scalar(v, config) for v in row] for row in rows]


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


def _div_round_to_canonical_interval(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ProofContractError("denominator must be positive")
    return (numerator + denominator // 2) // denominator


def _rescale(value: int, config: FixedPointConfig) -> int:
    _check_range(value, config.intermediate_bound, "intermediate")
    return _div_round_to_canonical_interval(value, config.scale)


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
    native = _native_module()
    if native is not None and hasattr(native, "compute_delta_quantized"):
        try:
            return [
                int(v)
                for v in native.compute_delta_quantized(
                    a,
                    b,
                    x,
                    int(scaling_num),
                    int(scaling_den),
                    config.scale_bits,
                    config.value_bits,
                    config.intermediate_bits,
                )
            ]
        except (OverflowError, TypeError, ValueError):
            # Fall back to the exact arbitrary-precision path below; it either
            # produces the result or raises the canonical contract error.
            pass
    return _compute_delta_quantized_python(a, b, x, scaling_num, scaling_den, config)


def compute_delta_quantized_rows(
    a: list[list[int]],
    b: list[list[int]],
    xs: list[list[int]],
    scaling_num: int,
    scaling_den: int,
    config: FixedPointConfig,
) -> list[list[int]]:
    """Batched delta computation for multiple input rows of one adapter.

    Uses the parallel native fast path when available (validating the adapter
    once instead of per row), with the exact per-row implementation as
    fallback. Results are identical to mapping compute_delta_quantized.
    """
    if not xs:
        return []
    native = _native_module()
    if native is not None and hasattr(native, "compute_delta_rows"):
        try:
            return [
                [int(v) for v in row]
                for row in native.compute_delta_rows(
                    a,
                    b,
                    xs,
                    int(scaling_num),
                    int(scaling_den),
                    config.scale_bits,
                    config.value_bits,
                    config.intermediate_bits,
                )
            ]
        except (OverflowError, TypeError, ValueError):
            pass
    return [
        compute_delta_quantized(a, b, x, scaling_num, scaling_den, config) for x in xs
    ]


def _compute_delta_quantized_python(
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
        scaled = _div_round_to_canonical_interval(
            scaled * int(scaling_num), int(scaling_den)
        )
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


def adapter_commitment_payload(
    a: list[list[int]],
    b: list[list[int]],
    scaling_num: int,
    scaling_den: int,
    fixed_point: FixedPointConfig,
) -> dict[str, Any]:
    rank = len(a)
    in_dim = len(a[0]) if a else 0
    out_dim = len(b)
    return {
        "schema_version": SCHEMA_VERSION,
        "in_dim": in_dim,
        "rank": rank,
        "out_dim": out_dim,
        "fixed_point": asdict(fixed_point),
        "scaling_num": int(scaling_num),
        "scaling_den": int(scaling_den),
        "a": a,
        "b": b,
    }


_ADAPTER_COMMITMENT_CACHE: dict[tuple, str] = {}
_ADAPTER_COMMITMENT_CACHE_MAX = 64
_ADAPTER_COMMITMENT_LOCK = threading.Lock()

_ADAPTER_SALT_LOCK = threading.Lock()
_ADAPTER_SALT: str | None = None


def adapter_salt() -> str:
    """Secret salt for the contributor's adapter commitments (64 hex chars).

    The salt only affects hiding, never binding: commitments stay binding
    under discrete log even if the salt leaks, and the salted commitment is
    strictly more hiding than an unsalted one. It must stay stable across
    manifest writing and proving so the pinned commitment matches the proofs;
    set ZKLORA_ADAPTER_SALT_FILE to persist it across processes (LoRAServer
    defaults this to a file in its output directory). The salt is contributor
    secret material: it never appears in manifests or proof artifacts.
    """
    global _ADAPTER_SALT
    with _ADAPTER_SALT_LOCK:
        if _ADAPTER_SALT is None:
            path = os.environ.get("ZKLORA_ADAPTER_SALT_FILE")
            if path and os.path.exists(path):
                value = Path(path).read_text(encoding="utf-8").strip()
                if len(value) != 64 or any(
                    c not in "0123456789abcdefABCDEF" for c in value
                ):
                    raise ProofContractError(
                        f"adapter salt file {path} must hold 64 hex characters"
                    )
                _ADAPTER_SALT = value.lower()
            else:
                _ADAPTER_SALT = secrets.token_hex(32)
                if path:
                    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        f.write(_ADAPTER_SALT + "\n")
        return _ADAPTER_SALT


def _freeze_matrix(matrix: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in matrix)


def _require_native():
    native = _native_module()
    if native is None:
        raise ProofContractError(
            "native zkLoRA prover is unavailable; build/install zklora with maturin"
        )
    return native


def adapter_commitment(
    a: list[list[int]],
    b: list[list[int]],
    scaling_num: int,
    scaling_den: int,
    fixed_point: FixedPointConfig,
) -> str:
    # The commitment is recomputed for every invocation statement of a module,
    # over the same (large) adapter matrices. Content-keyed memoisation keeps
    # the commitment derivation to one pass per adapter. The backend identity
    # is part of the key so monkeypatched/fake backends never share entries.
    native = _require_native()
    salt = adapter_salt()
    key = (
        id(native),
        salt,
        _freeze_matrix(a),
        _freeze_matrix(b),
        int(scaling_num),
        int(scaling_den),
        fixed_point,
    )
    with _ADAPTER_COMMITMENT_LOCK:
        cached = _ADAPTER_COMMITMENT_CACHE.get(key)
    if cached is not None:
        return cached
    value = str(
        native.adapter_commitment_v4(
            canonical_json(
                adapter_commitment_payload(a, b, scaling_num, scaling_den, fixed_point)
            ),
            salt,
        )
    )
    with _ADAPTER_COMMITMENT_LOCK:
        if len(_ADAPTER_COMMITMENT_CACHE) >= _ADAPTER_COMMITMENT_CACHE_MAX:
            _ADAPTER_COMMITMENT_CACHE.pop(next(iter(_ADAPTER_COMMITMENT_CACHE)))
        _ADAPTER_COMMITMENT_CACHE[key] = value
    return value


def adapter_setup(
    a: list[list[int]],
    b: list[list[int]],
    scaling_num: int,
    scaling_den: int,
    fixed_point: FixedPointConfig,
) -> dict[str, Any]:
    """Public adapter setup for the verifier: salted Pedersen row commitments
    plus the one-time proofs that every committed weight lies in the exact
    value-bound interval. Ships inside the pinned adapter manifest; contains
    no weight or salt material."""
    native = _require_native()
    return json.loads(
        native.adapter_setup_v4(
            canonical_json(
                adapter_commitment_payload(a, b, scaling_num, scaling_den, fixed_point)
            ),
            adapter_salt(),
        )
    )


def circuit_id(
    in_dim: int,
    rank: int,
    out_dim: int,
    fixed_point: FixedPointConfig,
    backend: str = BACKEND_ID,
) -> str:
    payload = {
        "backend": backend,
        "commitment_scheme": COMMITMENT_SCHEME,
        "invocation_strategy": INVOCATION_STRATEGY,
        "in_dim": in_dim,
        "rank": rank,
        "out_dim": out_dim,
        "fixed_point": asdict(fixed_point),
    }
    return digest_hex(payload)


def vk_fingerprint(circuit: str, backend: str = BACKEND_ID) -> str:
    return digest_hex({"backend": backend, "vk_for_circuit": circuit})


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
    commitment_value = adapter_commitment(
        witness.a,
        witness.b,
        witness.scaling_num,
        witness.scaling_den,
        witness.fixed_point,
    )

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
        "adapter_commitment": {
            "scheme": ADAPTER_COMMITMENT_SCHEME,
            "value": commitment_value,
        },
        "circuit_id": circuit,
        "vk_fingerprint": vk_fingerprint(circuit),
        "commitment_scheme": COMMITMENT_SCHEME,
        "invocation_strategy": INVOCATION_STRATEGY,
    }
    statement["statement_digest"] = digest_hex(_statement_digest_payload(statement))
    return statement


def _statement_digest_payload(statement: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in statement.items() if key != "statement_digest"}


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
    return Path(output_dir) / (
        f"{module_slug(statement['session_id'])}."
        f"{module_slug(statement['module_name'])}."
        f"{int(statement['invocation_index']):04d}"
    )


def _native_statement_json(statement: dict[str, Any]) -> str:
    payload = {
        "x": statement["x"],
        "delta": statement["delta"],
        "fixed_point": statement["fixed_point"],
        "rank": int(statement["adapter_metadata"]["rank"]),
        "scaling_num": int(statement["scaling"]["num"]),
        "scaling_den": int(statement["scaling"]["den"]),
        "adapter_commitment": str(statement["adapter_commitment"]["value"]),
        "statement_digest": statement["statement_digest"],
    }
    return canonical_json(payload)


def _native_witness_json(witness: InvocationWitness) -> str:
    return canonical_json({"a": witness.a, "b": witness.b})


def _native_module():
    try:
        return importlib.import_module("zklora._native_prover")
    except ModuleNotFoundError as exc:
        if exc.name == "zklora._native_prover":
            return None
        raise
    except ImportError as exc:
        raise ProofContractError(
            f"failed to import native Halo2 prover: {exc}"
        ) from exc
    except Exception as exc:
        raise ProofContractError(
            f"failed to import native Halo2 prover: {exc}"
        ) from exc


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

    native = _require_native()
    proof_bytes = native.prove_v4(
        _native_statement_json(statement), _native_witness_json(witness), adapter_salt()
    )
    proof_kind = PROOF_KIND
    vk = {
        "schema_version": SCHEMA_VERSION,
        "backend": BACKEND_ID,
        "circuit_id": statement["circuit_id"],
        "vk_fingerprint": statement["vk_fingerprint"],
    }
    pk = {
        "schema_version": SCHEMA_VERSION,
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
        "statement_digest": statement["statement_digest"],
        "statement_file_digest": digest_hex(statement),
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


def _worker_count(env_var: str) -> int:
    try:
        configured = int(os.environ.get(env_var, ""))
    except ValueError:
        configured = 0
    if configured > 0:
        return configured
    return max(os.cpu_count() or 1, 1)


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


def adapter_manifest_entry(
    module_name: str,
    a: list[list[int]],
    b: list[list[int]],
    scaling_num: int,
    scaling_den: int,
    fixed_point: FixedPointConfig,
) -> dict[str, Any]:
    rank = len(a)
    in_dim = len(a[0]) if a else 0
    out_dim = len(b)
    validate_matrix(a, rank, in_dim, "A")
    validate_matrix(b, out_dim, rank, "B")
    return {
        "module_name": module_name,
        "rank": rank,
        "in_dim": in_dim,
        "out_dim": out_dim,
        "fixed_point": asdict(fixed_point),
        "scaling": {"num": int(scaling_num), "den": int(scaling_den)},
        "adapter_commitment": {
            "scheme": ADAPTER_COMMITMENT_SCHEME,
            "value": adapter_commitment(
                a, b, int(scaling_num), int(scaling_den), fixed_point
            ),
        },
        # Public verification material: row commitments plus the one-time
        # weight range/link proofs. The commitment value above is the SHA-256
        # of the deterministic core, so pinning the manifest pins the setup.
        "adapter_setup": adapter_setup(
            a, b, int(scaling_num), int(scaling_den), fixed_point
        ),
    }


def write_adapter_manifest(
    path: str | os.PathLike[str], entries: Iterable[dict[str, Any]]
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "backend": BACKEND_ID,
        "commitment_scheme": ADAPTER_COMMITMENT_SCHEME,
        "adapters": list(entries),
    }
    Path(path).write_text(canonical_json(payload) + "\n", encoding="utf-8")


def load_expected_adapters(
    expected_adapters: str
    | os.PathLike[str]
    | dict[str, Any]
    | Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if isinstance(expected_adapters, (str, os.PathLike)):
        data: Any = load_json(expected_adapters)
    elif isinstance(expected_adapters, dict):
        data = expected_adapters
    else:
        data = {"adapters": list(expected_adapters)}

    adapters = data.get("adapters") if isinstance(data, dict) else None
    if adapters is None:
        raise ProofContractError("expected adapter manifest must contain adapters")

    index: dict[str, dict[str, Any]] = {}
    for entry in adapters:
        module_name = entry["module_name"]
        if module_name in index:
            raise ProofContractError(f"duplicate expected adapter for {module_name}")
        index[module_name] = entry
    return index


def statement_matches_expected_adapter(
    statement: dict[str, Any], expected: dict[str, Any]
) -> bool:
    return (
        statement["module_name"] == expected["module_name"]
        and int(statement["adapter_metadata"]["rank"]) == int(expected["rank"])
        and len(statement["x"]) == int(expected["in_dim"])
        and len(statement["delta"]) == int(expected["out_dim"])
        and statement["fixed_point"] == expected["fixed_point"]
        and statement["scaling"] == expected["scaling"]
        and statement["adapter_commitment"] == expected["adapter_commitment"]
    )


def verify_artifact_set(
    statement_path: str | os.PathLike[str],
    transcript_entries: Iterable[TranscriptEntry],
    expected_adapters: dict[str, dict[str, Any]],
) -> None:
    statement = load_json(statement_path)
    if (
        statement.get("schema_version") == LEGACY_SCHEMA_VERSION
        and statement.get("backend") == LEGACY_BACKEND_ID
    ):
        backend = LEGACY_BACKEND_ID
        proof_kind = LEGACY_PROOF_KIND
        adapter_scheme = LEGACY_ADAPTER_COMMITMENT_SCHEME
    elif (
        statement.get("schema_version") == SCHEMA_VERSION
        and statement.get("backend") == BACKEND_ID
    ):
        backend = BACKEND_ID
        proof_kind = PROOF_KIND
        adapter_scheme = ADAPTER_COMMITMENT_SCHEME
    else:
        raise ProofContractError("unsupported proof artifact schema/backend")
    if statement.get("statement_digest") != digest_hex(
        _statement_digest_payload(statement)
    ):
        raise ProofContractError("statement digest mismatch")
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
    expected_adapter = expected_adapters.get(statement["module_name"])
    if expected_adapter is None:
        raise ProofContractError(
            "statement module is missing from expected adapter manifest"
        )
    if not statement_matches_expected_adapter(statement, expected_adapter):
        raise ProofContractError("statement does not match expected adapter manifest")

    expected_circuit = circuit_id(
        len(statement["x"]),
        int(statement["adapter_metadata"].get("rank", 0)),
        len(statement["delta"]),
        FixedPointConfig(**statement["fixed_point"]),
        backend=backend,
    )
    if statement["circuit_id"] != expected_circuit:
        raise ProofContractError("statement circuit_id does not match expected circuit")
    if statement["vk_fingerprint"] != vk_fingerprint(expected_circuit, backend=backend):
        raise ProofContractError(
            "statement vk_fingerprint does not match expected circuit"
        )
    adapter = statement.get("adapter_commitment", {})
    if adapter.get("scheme") != adapter_scheme:
        raise ProofContractError("statement adapter commitment scheme mismatch")

    vk = load_json(vk_path)
    if (
        vk.get("schema_version") != statement["schema_version"]
        or vk.get("backend") != backend
    ):
        raise ProofContractError("unsupported verification key schema/backend")
    if vk.get("circuit_id") != expected_circuit:
        raise ProofContractError("verification key circuit_id mismatch")
    if vk.get("vk_fingerprint") != statement["vk_fingerprint"]:
        raise ProofContractError("verification key fingerprint mismatch")

    meta = load_json(meta_path)
    proof_bytes = proof_path.read_bytes()
    if (
        meta.get("schema_version") != statement["schema_version"]
        or meta.get("backend") != backend
    ):
        raise ProofContractError("unsupported metadata schema/backend")
    if meta.get("proof_kind") != proof_kind:
        raise ProofContractError(f"unsupported proof kind; expected {proof_kind}")
    native = _native_module()
    if native is None:
        raise ProofContractError("native zkLoRA verifier is unavailable")
    if backend == BACKEND_ID:
        setup = expected_adapter.get("adapter_setup")
        if not isinstance(setup, dict):
            raise ProofContractError(
                "expected adapter manifest entry lacks adapter_setup for sigma-v4"
            )
        # verify_v4 checks that SHA-256 of the setup core equals the
        # statement's pinned adapter commitment, verifies the one-time
        # weight range/link proofs (cached per adapter), and verifies the
        # invocation proof against the row commitments.
        if not native.verify_v4(
            _native_statement_json(statement), proof_bytes, canonical_json(setup)
        ):
            raise ProofContractError("proof bytes failed native sigma-v4 verification")
    else:
        if not native.verify(_native_statement_json(statement), proof_bytes):
            raise ProofContractError("proof bytes failed native Halo2 verification")

    if meta.get("statement_digest") != statement["statement_digest"]:
        raise ProofContractError("metadata statement digest mismatch")
    if meta.get("statement_file_digest") != digest_hex(statement):
        raise ProofContractError("metadata statement file digest mismatch")
    if meta.get("proof_digest") != hashlib.sha256(proof_bytes).hexdigest():
        raise ProofContractError("metadata proof digest mismatch")
    if meta.get("vk_fingerprint") != statement["vk_fingerprint"]:
        raise ProofContractError("metadata vk fingerprint mismatch")


def verify_artifacts(
    proof_dir: str | os.PathLike[str],
    transcript: str | os.PathLike[str] | Iterable[Any],
    expected_adapters: str
    | os.PathLike[str]
    | dict[str, Any]
    | Iterable[dict[str, Any]],
) -> tuple[float, int]:
    import time
    from concurrent.futures import ThreadPoolExecutor

    start = time.time()
    entries = load_transcript(transcript)
    adapter_index = load_expected_adapters(expected_adapters)
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

    # Every artifact is verified independently; the native verifier releases
    # the GIL, so a small thread pool overlaps proof verification across files.
    max_workers = min(len(statement_files), _worker_count("ZKLORA_VERIFY_WORKERS"))
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(verify_artifact_set, statement_file, entries, adapter_index)
                for statement_file in statement_files
            ]
            for future in futures:
                future.result()
    else:
        for statement_file in statement_files:
            verify_artifact_set(statement_file, entries, adapter_index)

    expected = {entry.key() for entry in entries}
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ProofContractError(
            f"proof transcript coverage mismatch missing={missing} extra={extra}"
        )
    return time.time() - start, len(statement_files)
