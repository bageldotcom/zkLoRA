"""Schema-3 projection-backend artifact layer.

One artifact set covers a contiguous batch of module invocations (rows) instead
of a single row. Statements are digest-only: full ``x``/``delta`` rows live in
the verifier-recorded transcript and are bound through per-row digests plus a
batch transcript digest. The cryptographic relation is proven by the native
``pedersen-projection-v1`` backend; this module owns statements, batching,
manifests, coverage, and dispatch between schema versions.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .proof_contract import (
    FixedPointConfig,
    InvocationWitness,
    ProofContractError,
    TranscriptEntry,
    canonical_json,
    compute_delta_quantized,
    digest_hex,
    load_json,
    load_transcript,
    module_slug,
)
from . import proof_contract as _v2

SCHEMA_VERSION_V3 = 3
BACKEND_ID_V3 = "zklora-projection-v1"
PROOF_KIND_V3 = "pedersen-projection-v1"
PROOF_SYSTEM_V3 = "pedersen-projection-ipa-bp"
COMMITMENT_SCHEME_V3 = "pedersen-vector-vesta-v1"
RANGE_ARGUMENT_V3 = "bp-aggregate-linked-v1"
GENERATOR_SEED_ID = "zklora/v3/gen-seed/v1"
ENCODING_V3 = "row-major-exact-len-v1"
FIAT_SHAMIR_V3 = "merlin-strobe128-v1"
INVOCATION_STRATEGY_V3 = "one-proof-per-row-batch-v1"
SECURITY_LEVEL_BITS_V3 = 128
AUDIT_STATUS_V3 = "unaudited"

DEFAULT_TARGET_CHUNK_ROWS = 256
MAX_BATCH_ROWS = 4096
# Python mirror of the Rust-enforced DoS caps (Rust verify_v3 is sovereign).
MAX_V3_DIM = 65_536
MAX_V3_RANK = 1_024
FIELD_SAFE_BITS = 250

_CHUNK_ROWS_ENV = "ZKLORA_CHUNK_ROWS"
_SECRET_ENV = "ZKLORA_CONTRIBUTOR_SECRET"
_SECRET_DEFAULT = Path("~") / ".zklora" / "contributor.secret.json"
_SECRET_GLOB = "*.secret.json"

_STATEMENT_SUFFIX = ".zklora.statement.json"


def _native_v3():
    # Resolved through the module attribute so test fakes that patch
    # proof_contract._native_module cover both schema paths.
    native = _v2._native_module()
    if native is None:
        raise ProofContractError(
            "native projection prover is unavailable; build/install zklora with "
            "maturin, or unset ZKLORA_PROVER_BACKEND to use the default legacy "
            "backend"
        )
    missing = [
        name
        for name in (
            "prove_v3",
            "verify_v3",
            "adapter_commit_v3",
            "verify_adapter_manifest_v3",
        )
        if not hasattr(native, name)
    ]
    if missing:
        raise ProofContractError(
            "installed native module does not support the opt-in projection "
            f"backend (missing {', '.join(missing)}); rebuild zklora with a "
            "native module that ships v3 support, or unset "
            "ZKLORA_PROVER_BACKEND to use the default legacy backend"
        )
    return native


def _ceil_log2(value: int) -> int:
    if value <= 1:
        return 0
    return (value - 1).bit_length()


def target_chunk_rows() -> int:
    raw = os.environ.get(_CHUNK_ROWS_ENV)
    if raw is None:
        return DEFAULT_TARGET_CHUNK_ROWS
    try:
        value = int(raw)
    except ValueError as exc:
        raise ProofContractError(f"invalid {_CHUNK_ROWS_ENV}: {raw!r}") from exc
    if value < 1 or value > MAX_BATCH_ROWS:
        raise ProofContractError(
            f"{_CHUNK_ROWS_ENV} must be within [1, {MAX_BATCH_ROWS}], got {value}"
        )
    return value


# ---------------------------------------------------------------------------
# Contributor secret handling
# ---------------------------------------------------------------------------


def resolve_contributor_secret_path(
    explicit: str | os.PathLike[str] | None = None,
) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser()
    env = os.environ.get(_SECRET_ENV)
    if env:
        return Path(env).expanduser()
    return _SECRET_DEFAULT.expanduser()


def ensure_secret_outside_artifacts(
    secret_path: Path, output_dir: str | os.PathLike[str]
) -> None:
    secret = secret_path.expanduser().resolve()
    artifacts = Path(output_dir).expanduser().resolve()
    if secret == artifacts or secret.is_relative_to(artifacts):
        raise ProofContractError(
            "contributor secret must not live inside the shared artifact "
            f"directory: {secret} is inside {artifacts}; the artifact directory "
            "is handed to the verifier and a leaked seed destroys hiding of the "
            "adapter commitments"
        )


def _read_contributor_secret(path: Path) -> str:
    data = load_json(path)
    seed = str(data.get("seed", ""))
    if len(seed) != 64 or any(c not in "0123456789abcdef" for c in seed.lower()):
        raise ProofContractError(f"malformed contributor secret at {path}")
    return seed


def load_or_create_contributor_secret(path: Path) -> str:
    path = path.expanduser()
    if path.exists():
        return _read_contributor_secret(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    seed = secrets.token_hex(32)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        # Lost a concurrent-creation race; the winner's seed is authoritative.
        return _read_contributor_secret(path)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(canonical_json({"seed": seed}) + "\n")
    return seed


def reject_secrets_in_artifact_dir(proof_dir: str | os.PathLike[str]) -> None:
    leaked = sorted(Path(proof_dir).glob(_SECRET_GLOB))
    if leaked:
        names = ", ".join(p.name for p in leaked)
        raise ProofContractError(
            f"refusing to verify: contributor secret file(s) found among proof "
            f"artifacts ({names}); the seed must never be shared with the verifier"
        )


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchWitness:
    session_id: str
    module_name: str
    start_invocation_index: int
    rows: list[InvocationWitness] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.rows)

    @property
    def first(self) -> InvocationWitness:
        return self.rows[0]

    @property
    def x_rows(self) -> list[list[int]]:
        return [[int(v) for v in row.x] for row in self.rows]

    @property
    def delta_rows(self) -> list[list[int]]:
        return [[int(v) for v in row.delta] for row in self.rows]


def _group_key(record: InvocationWitness):
    return (
        record.session_id,
        record.module_name,
        record.in_dim,
        record.rank,
        record.out_dim,
        record.fixed_point,
        int(record.scaling_num),
        int(record.scaling_den),
    )


def build_batches(
    records: Iterable[InvocationWitness], target_rows: int | None = None
) -> list[BatchWitness]:
    target = target_chunk_rows() if target_rows is None else int(target_rows)
    if target < 1 or target > MAX_BATCH_ROWS:
        raise ProofContractError(
            f"target chunk rows must be within [1, {MAX_BATCH_ROWS}], got {target}"
        )

    groups: dict[Any, list[InvocationWitness]] = {}
    order: list[Any] = []
    for record in records:
        if len(record.x) != record.in_dim:
            raise ProofContractError(
                f"record x length {len(record.x)} does not match in_dim "
                f"{record.in_dim} for {record.module_name}"
            )
        if len(record.delta) != record.out_dim:
            raise ProofContractError(
                f"record delta length {len(record.delta)} does not match out_dim "
                f"{record.out_dim} for {record.module_name}"
            )
        key = _group_key(record)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(record)

    batches: list[BatchWitness] = []
    for key in order:
        rows = sorted(groups[key], key=lambda r: int(r.invocation_index))
        reference = rows[0]
        for row in rows[1:]:
            if row.a != reference.a or row.b != reference.b:
                raise ProofContractError(
                    "batch group has inconsistent adapter weights for "
                    f"{reference.module_name}; one batch must cover one adapter"
                )
        indices = [int(row.invocation_index) for row in rows]
        for prev, cur in zip(indices, indices[1:]):
            if cur == prev:
                raise ProofContractError(
                    f"duplicate invocation index {cur} for {reference.module_name}"
                )
            if cur != prev + 1:
                raise ProofContractError(
                    "non-contiguous invocation indices for "
                    f"{reference.module_name}: {prev} -> {cur}"
                )
        for offset in range(0, len(rows), target):
            chunk = rows[offset : offset + target]
            batches.append(
                BatchWitness(
                    session_id=reference.session_id,
                    module_name=reference.module_name,
                    start_invocation_index=int(chunk[0].invocation_index),
                    rows=list(chunk),
                )
            )
    return batches


# ---------------------------------------------------------------------------
# Digests, identifiers, statements
# ---------------------------------------------------------------------------


def row_digest(
    *,
    session_id: str,
    module_name: str,
    invocation_index: int,
    input_shape: list[int],
    output_shape: list[int],
    x_row: list[int],
    delta_row: list[int],
    fixed_point: FixedPointConfig,
    scaling_num: int,
    scaling_den: int,
    rank: int,
    in_dim: int,
    out_dim: int,
    adapter_commitment: dict[str, Any],
    manifest_commitment: str,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION_V3,
        "backend": BACKEND_ID_V3,
        "proof_kind": PROOF_KIND_V3,
        "session_id": session_id,
        "module_name": module_name,
        "invocation_index": int(invocation_index),
        "input_shape": [int(v) for v in input_shape],
        "output_shape": [int(v) for v in output_shape],
        "x_row": [int(v) for v in x_row],
        "delta_row": [int(v) for v in delta_row],
        "fixed_point": asdict(fixed_point),
        "scaling": {"num": int(scaling_num), "den": int(scaling_den)},
        "rank": int(rank),
        "in_dim": int(in_dim),
        "out_dim": int(out_dim),
        "adapter_commitment": adapter_commitment,
        "manifest_commitment": manifest_commitment,
    }
    return digest_hex(payload)


def batch_transcript_digest(row_digests: list[str]) -> str:
    return digest_hex(list(row_digests))


def circuit_id_v3(
    in_dim: int,
    rank: int,
    out_dim: int,
    fixed_point: FixedPointConfig,
    scaling_num: int,
    scaling_den: int,
    chunk_rows: int,
) -> str:
    payload = {
        "backend": BACKEND_ID_V3,
        "proof_kind": PROOF_KIND_V3,
        "commitment_scheme": COMMITMENT_SCHEME_V3,
        "generator_seed_id": GENERATOR_SEED_ID,
        "encoding": ENCODING_V3,
        "range_argument": RANGE_ARGUMENT_V3,
        "fiat_shamir": FIAT_SHAMIR_V3,
        "in_dim": int(in_dim),
        "rank": int(rank),
        "out_dim": int(out_dim),
        "fixed_point": asdict(fixed_point),
        "scaling": {"num": int(scaling_num), "den": int(scaling_den)},
        "target_chunk_rows": int(chunk_rows),
    }
    return digest_hex(payload)


def vk_fingerprint_v3(circuit: str) -> str:
    return digest_hex({"backend": BACKEND_ID_V3, "vk_for_circuit": circuit})


def _statement_digest_payload(statement: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in statement.items() if key != "statement_digest"}


def derive_bounds(
    x_rows: list[list[int]],
    delta_rows: list[list[int]],
    fixed_point: FixedPointConfig,
    scaling_num: int,
    scaling_den: int,
) -> dict[str, int]:
    """Public per-batch bounds and range widths, identical on both sides.

    The composition checks below are stated against the *proved* bounds
    ``P = 2^n - 1 - shift`` (what a one-sided ``[0, 2^n)`` range proof actually
    establishes), never against the nominal bounds.
    """

    scale = fixed_point.scale
    value_bound = fixed_point.value_bound
    b_u = 0
    for row in x_rows:
        raw = sum(abs(int(v)) for v in row) * value_bound
        b_u = max(b_u, (raw + scale // 2) // scale)
    d_max = max((abs(int(v)) for row in delta_rows for v in row), default=0)
    num = int(scaling_num)
    den = int(scaling_den)
    if den <= 0 or num == 0:
        raise ProofContractError("scaling must have positive den and nonzero num")
    b_m = (den * d_max + den // 2) // abs(num)

    n_u = max(1, (2 * b_u).bit_length())
    n_m = max(1, (2 * b_m).bit_length())
    proved_a = 1 << (fixed_point.value_bits - 1)
    proved_u = (1 << n_u) - 1 - b_u
    proved_m = (1 << n_m) - 1 - b_m
    return {
        "value_bound": value_bound,
        "b_u": b_u,
        "b_m": b_m,
        "n_u": n_u,
        "n_m": n_m,
        "n_rem": int(fixed_point.scale_bits),
        "n_rd2": max(1, _ceil_log2(den)),
        "proved_a": proved_a,
        "proved_u": proved_u,
        "proved_m": proved_m,
    }


def check_bounds_composition(
    bounds: dict[str, int],
    in_dim: int,
    rank: int,
    fixed_point: FixedPointConfig,
    scaling_num: int,
    scaling_den: int,
) -> None:
    proved_a = bounds["proved_a"]
    proved_u = bounds["proved_u"]
    proved_m = bounds["proved_m"]
    checks = [
        2 * proved_a.bit_length() + _ceil_log2(max(1, in_dim)) + 1,
        int(fixed_point.scale_bits) + proved_u.bit_length() + 1,
        proved_u.bit_length() + proved_a.bit_length() + _ceil_log2(max(1, rank)) + 1,
        abs(int(scaling_num)).bit_length() + proved_m.bit_length() + 1,
        int(scaling_den).bit_length() + bounds["value_bound"].bit_length() + 1,
    ]
    if any(bits > FIELD_SAFE_BITS for bits in checks):
        raise ProofContractError(
            "fixed-point config and batch dimensions exceed Pasta field-safe "
            "integer bounds for the projection relation"
        )


def _check_v3_dims(in_dim: int, rank: int, out_dim: int, count: int) -> None:
    if in_dim < 1 or out_dim < 1 or in_dim > MAX_V3_DIM or out_dim > MAX_V3_DIM:
        raise ProofContractError(
            f"projection artifact exceeds verification caps: dims {in_dim}x{out_dim}"
        )
    if rank < 1 or rank > MAX_V3_RANK:
        raise ProofContractError(
            f"projection artifact exceeds verification caps: rank {rank}"
        )
    if count < 1 or count > MAX_BATCH_ROWS:
        raise ProofContractError(
            f"projection artifact exceeds verification caps: row count {count}"
        )


def statement_from_batch(
    batch: BatchWitness,
    manifest_entry: dict[str, Any],
    manifest_commitment: str,
    chunk_rows: int,
) -> dict[str, Any]:
    first = batch.first
    rank = first.rank
    in_dim = first.in_dim
    out_dim = first.out_dim
    _check_v3_dims(in_dim, rank, out_dim, batch.count)
    if batch.count > int(chunk_rows):
        raise ProofContractError(
            f"batch of {batch.count} rows exceeds target chunk {chunk_rows}"
        )
    _check_manifest_entry_matches(
        manifest_entry,
        module_name=batch.module_name,
        rank=rank,
        in_dim=in_dim,
        out_dim=out_dim,
        fixed_point=first.fixed_point,
        scaling_num=first.scaling_num,
        scaling_den=first.scaling_den,
    )

    for row in batch.rows:
        # The statement pins input_shape/output_shape to the flat [in_dim] /
        # [out_dim] form and verification requires every transcript row to
        # match it exactly; reject other recorded shapes here so a drifting
        # capture path fails at generation time with a clear error instead of
        # producing artifacts that can never verify.
        if list(row.input_shape) != [in_dim] or list(row.output_shape) != [out_dim]:
            raise ProofContractError(
                "schema-3 statements require per-row shapes "
                f"[{in_dim}]/[{out_dim}]; {batch.module_name}"
                f"#{row.invocation_index} recorded input_shape="
                f"{row.input_shape} output_shape={row.output_shape}"
            )
        expected = compute_delta_quantized(
            first.a,
            first.b,
            row.x,
            first.scaling_num,
            first.scaling_den,
            first.fixed_point,
        )
        if expected != row.delta:
            raise ProofContractError(
                "witness delta does not match fixed-point LoRA relation for "
                f"{batch.module_name}#{row.invocation_index}"
            )

    adapter_commitment = manifest_entry["adapter_commitment"]
    digests = [
        row_digest(
            session_id=batch.session_id,
            module_name=batch.module_name,
            invocation_index=row.invocation_index,
            input_shape=row.input_shape,
            output_shape=row.output_shape,
            x_row=row.x,
            delta_row=row.delta,
            fixed_point=first.fixed_point,
            scaling_num=first.scaling_num,
            scaling_den=first.scaling_den,
            rank=rank,
            in_dim=in_dim,
            out_dim=out_dim,
            adapter_commitment=adapter_commitment,
            manifest_commitment=manifest_commitment,
        )
        for row in batch.rows
    ]
    bounds = derive_bounds(
        batch.x_rows,
        batch.delta_rows,
        first.fixed_point,
        first.scaling_num,
        first.scaling_den,
    )
    check_bounds_composition(
        bounds, in_dim, rank, first.fixed_point, first.scaling_num, first.scaling_den
    )

    circuit = circuit_id_v3(
        in_dim,
        rank,
        out_dim,
        first.fixed_point,
        first.scaling_num,
        first.scaling_den,
        chunk_rows,
    )
    statement = {
        "schema_version": SCHEMA_VERSION_V3,
        "backend": BACKEND_ID_V3,
        "proof_kind": PROOF_KIND_V3,
        "session_id": batch.session_id,
        "module_name": batch.module_name,
        "start_invocation_index": batch.start_invocation_index,
        "count": batch.count,
        "target_chunk_rows": int(chunk_rows),
        "input_shape": [in_dim],
        "output_shape": [out_dim],
        "row_digests": digests,
        "batch_transcript_digest": batch_transcript_digest(digests),
        "fixed_point": asdict(first.fixed_point),
        "scaling": {"num": int(first.scaling_num), "den": int(first.scaling_den)},
        "adapter_commitment": adapter_commitment,
        "manifest_commitment": manifest_commitment,
        "circuit_id": circuit,
        "vk_fingerprint": vk_fingerprint_v3(circuit),
    }
    statement["statement_digest"] = digest_hex(_statement_digest_payload(statement))
    return statement


def _check_manifest_entry_matches(
    entry: dict[str, Any],
    *,
    module_name: str,
    rank: int,
    in_dim: int,
    out_dim: int,
    fixed_point: FixedPointConfig,
    scaling_num: int,
    scaling_den: int,
) -> None:
    if entry.get("module_name") != module_name:
        raise ProofContractError(
            f"manifest entry module {entry.get('module_name')!r} != {module_name!r}"
        )
    if "a_commitment" not in entry or "commitment_nonce" not in entry:
        raise ProofContractError(
            f"module {module_name} requires a schema-3 manifest entry with "
            "pedersen commitments"
        )
    if (
        int(entry.get("rank", -1)) != rank
        or int(entry.get("in_dim", -1)) != in_dim
        or int(entry.get("out_dim", -1)) != out_dim
        or entry.get("fixed_point") != asdict(fixed_point)
        or entry.get("scaling") != {"num": int(scaling_num), "den": int(scaling_den)}
    ):
        raise ProofContractError(
            f"statement does not match pinned manifest entry for {module_name}"
        )
    scheme = entry.get("adapter_commitment", {}).get("scheme")
    if scheme != COMMITMENT_SCHEME_V3:
        raise ProofContractError(
            f"manifest adapter commitment scheme mismatch for {module_name}"
        )


# ---------------------------------------------------------------------------
# Adapter manifest (schema 3)
# ---------------------------------------------------------------------------


def adapter_manifest_entry_v3(
    module_name: str,
    a: list[list[int]],
    b: list[list[int]],
    scaling_num: int,
    scaling_den: int,
    fixed_point: FixedPointConfig,
    secret_seed_hex: str,
) -> dict[str, Any]:
    rank = len(a)
    in_dim = len(a[0]) if a else 0
    out_dim = len(b)
    _v2.validate_matrix(a, rank, in_dim, "A")
    _v2.validate_matrix(b, out_dim, rank, "B")
    _check_v3_dims(in_dim, rank, out_dim, 1)
    native = _native_v3()
    adapter_json = canonical_json(
        {
            "schema_version": SCHEMA_VERSION_V3,
            "module_name": module_name,
            "in_dim": in_dim,
            "rank": rank,
            "out_dim": out_dim,
            "fixed_point": asdict(fixed_point),
            "scaling_num": int(scaling_num),
            "scaling_den": int(scaling_den),
            "a": a,
            "b": b,
        }
    )
    commit = json.loads(native.adapter_commit_v3(adapter_json, secret_seed_hex))
    for key in ("a_commitment", "b_commitment", "commitment_nonce", "range_proof"):
        if key not in commit:
            raise ProofContractError(f"native adapter_commit_v3 omitted {key}")
    public = {
        "module_name": module_name,
        "rank": rank,
        "in_dim": in_dim,
        "out_dim": out_dim,
        "fixed_point": asdict(fixed_point),
        "scaling": {"num": int(scaling_num), "den": int(scaling_den)},
        "a_commitment": list(commit["a_commitment"]),
        "b_commitment": list(commit["b_commitment"]),
        "commitment_nonce": str(commit["commitment_nonce"]),
    }
    value = digest_hex(public)
    entry = dict(public)
    entry["adapter_commitment"] = {"scheme": COMMITMENT_SCHEME_V3, "value": value}
    entry["range_proof"] = str(commit["range_proof"])
    return entry


def adapter_manifest_payload_v3(entries: Iterable[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION_V3,
        "backend": BACKEND_ID_V3,
        "commitment_scheme": COMMITMENT_SCHEME_V3,
        "adapters": list(entries),
    }


def write_adapter_manifest_v3(
    path: str | os.PathLike[str], entries: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    payload = adapter_manifest_payload_v3(entries)
    Path(path).write_text(canonical_json(payload) + "\n", encoding="utf-8")
    return payload


def manifest_commitment_of(payload: dict[str, Any]) -> str:
    return digest_hex(payload)


_MANIFEST_META_KEY = "__zklora_manifest__"


def _index_expected_adapter(index: dict[str, Any], entry: dict[str, Any]) -> None:
    module_name = entry["module_name"]
    if module_name in index:
        raise ProofContractError(f"duplicate expected adapter for {module_name}")
    if "a_commitment" in entry:
        scheme = entry.get("adapter_commitment", {}).get("scheme")
        if scheme != COMMITMENT_SCHEME_V3:
            raise ProofContractError(
                f"unsupported v3 adapter commitment scheme for {module_name}"
            )
        expected_value = digest_hex(
            {
                "module_name": entry["module_name"],
                "rank": int(entry["rank"]),
                "in_dim": int(entry["in_dim"]),
                "out_dim": int(entry["out_dim"]),
                "fixed_point": entry["fixed_point"],
                "scaling": entry["scaling"],
                "a_commitment": list(entry["a_commitment"]),
                "b_commitment": list(entry["b_commitment"]),
                "commitment_nonce": str(entry["commitment_nonce"]),
            }
        )
        if entry["adapter_commitment"].get("value") != expected_value:
            raise ProofContractError(
                f"manifest adapter commitment mismatch for {module_name}"
            )
        _check_v3_dims(
            int(entry["in_dim"]), int(entry["rank"]), int(entry["out_dim"]), 1
        )
        native = _native_v3()
        if not native.verify_adapter_manifest_v3(canonical_json(entry)):
            raise ProofContractError(
                f"manifest adapter range proof failed for {module_name}"
            )
    index[module_name] = entry


def load_expected_adapters_any(
    expected_adapters: str
    | os.PathLike[str]
    | dict[str, Any]
    | Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Index adapter entries by module, schema-aware.

    Schema-3 entries (those carrying pedersen commitments) are verified at pin
    time via the native one-time A/B range proof; the manifest commitment over
    the full pinned payload is stashed under a reserved key for statement
    checks. v2 entries pass through untouched so legacy artifacts keep
    verifying against them.
    """

    if isinstance(expected_adapters, (str, os.PathLike)):
        data: Any = load_json(expected_adapters)
    elif isinstance(expected_adapters, dict):
        data = expected_adapters
    else:
        data = {"adapters": list(expected_adapters)}

    adapters = data.get("adapters") if isinstance(data, dict) else None
    if adapters is None:
        raise ProofContractError("expected adapter manifest must contain adapters")

    index: dict[str, Any] = {}
    has_v3 = False
    for position, entry in enumerate(adapters):
        try:
            _index_expected_adapter(index, entry)
        except ProofContractError:
            raise
        except (KeyError, TypeError, IndexError, ValueError, AttributeError) as exc:
            raise ProofContractError(
                f"malformed expected adapter entry at position {position}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if "a_commitment" in entry:
            has_v3 = True

    manifest_commitment = None
    if has_v3:
        if not isinstance(data, dict) or data.get("schema_version") not in (
            SCHEMA_VERSION_V3,
        ):
            # Entries supplied without a pinned schema-3 manifest envelope still
            # get a commitment over a canonical envelope so statements can bind.
            data = adapter_manifest_payload_v3(adapters)
        manifest_commitment = manifest_commitment_of(
            {k: v for k, v in data.items() if k != _MANIFEST_META_KEY}
        )
    index[_MANIFEST_META_KEY] = {"manifest_commitment": manifest_commitment}
    return index


def _adapters_index_for_v2(index: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in index.items() if k != _MANIFEST_META_KEY}


# ---------------------------------------------------------------------------
# Artifact writing
# ---------------------------------------------------------------------------


def _artifact_prefix(
    output_dir: str | os.PathLike[str], statement: dict[str, Any]
) -> Path:
    return Path(output_dir) / (
        f"{module_slug(statement['session_id'])}."
        f"{module_slug(statement['module_name'])}."
        f"{int(statement['start_invocation_index']):04d}"
    )


def _rows_json(x_rows: list[list[int]], delta_rows: list[list[int]]) -> str:
    return canonical_json({"x_rows": x_rows, "delta_rows": delta_rows})


def write_batch_artifacts(
    output_dir: str | os.PathLike[str],
    batch: BatchWitness,
    manifest_entry: dict[str, Any],
    manifest_commitment: str,
    secret_seed_hex: str,
    chunk_rows: int | None = None,
) -> dict[str, str]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    chunk = target_chunk_rows() if chunk_rows is None else int(chunk_rows)
    statement = statement_from_batch(batch, manifest_entry, manifest_commitment, chunk)
    prefix = _artifact_prefix(output_dir, statement)
    proof_path = Path(f"{prefix}.zklora.proof")
    vk_path = Path(f"{prefix}.zklora.vk")
    pk_path = Path(f"{prefix}.zklora.pk")
    statement_path = Path(f"{prefix}{_STATEMENT_SUFFIX}")
    meta_path = Path(f"{prefix}.zklora.meta.json")

    native = _native_v3()
    first = batch.first
    witness_json = canonical_json(
        {
            "a": first.a,
            "b": first.b,
            "secret_seed": secret_seed_hex,
            "commitment_nonce": manifest_entry["commitment_nonce"],
        }
    )
    proof_bytes = native.prove_v3(
        canonical_json(statement),
        _rows_json(batch.x_rows, batch.delta_rows),
        witness_json,
    )

    vk = {
        "schema_version": SCHEMA_VERSION_V3,
        "backend": BACKEND_ID_V3,
        "proof_kind": PROOF_KIND_V3,
        "commitment_scheme": COMMITMENT_SCHEME_V3,
        "range_argument": RANGE_ARGUMENT_V3,
        "generator_seed_id": GENERATOR_SEED_ID,
        "circuit_id": statement["circuit_id"],
        "vk_fingerprint": statement["vk_fingerprint"],
    }
    pk = {
        "schema_version": SCHEMA_VERSION_V3,
        "backend": BACKEND_ID_V3,
        "circuit_id": statement["circuit_id"],
        "pk_fingerprint": digest_hex({"pk_for_circuit": statement["circuit_id"]}),
    }
    meta = {
        "schema_version": SCHEMA_VERSION_V3,
        "backend": BACKEND_ID_V3,
        "proof_kind": PROOF_KIND_V3,
        "proof_system": PROOF_SYSTEM_V3,
        "commitment_scheme": COMMITMENT_SCHEME_V3,
        "range_argument": RANGE_ARGUMENT_V3,
        "generator_seed_id": GENERATOR_SEED_ID,
        "fiat_shamir": FIAT_SHAMIR_V3,
        "security_level_bits": SECURITY_LEVEL_BITS_V3,
        "audit_status": AUDIT_STATUS_V3,
        "proof_file": proof_path.name,
        "statement_file": statement_path.name,
        "vk_file": vk_path.name,
        "pk_file": pk_path.name,
        "statement_digest": statement["statement_digest"],
        "statement_file_digest": digest_hex(statement),
        "proof_digest": hashlib.sha256(proof_bytes).hexdigest(),
        "vk_digest": digest_hex(vk),
        "pk_digest": digest_hex(pk),
        "circuit_id": statement["circuit_id"],
        "vk_fingerprint": statement["vk_fingerprint"],
        "manifest_commitment": manifest_commitment,
        "batch_transcript_digest": statement["batch_transcript_digest"],
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


def generate_batch_proofs(
    records: Iterable[InvocationWitness],
    output_dir: str | os.PathLike[str],
    adapter_manifest: str | os.PathLike[str] | dict[str, Any],
    manifest_secret_path: str | os.PathLike[str] | None = None,
    verbose: bool = False,
) -> tuple[int, int]:
    """Generate schema-3 batch artifacts; returns (artifact_sets, total_params)."""

    if adapter_manifest is None:
        raise ProofContractError(
            "projection backend requires the pinned adapter manifest "
            "(adapter_manifest=...)"
        )
    if isinstance(adapter_manifest, (str, os.PathLike)):
        manifest_payload = load_json(adapter_manifest)
    else:
        manifest_payload = adapter_manifest
    if manifest_payload.get("schema_version") != SCHEMA_VERSION_V3:
        raise ProofContractError(
            "projection backend requires a schema-3 adapter manifest"
        )
    manifest_commitment = manifest_commitment_of(manifest_payload)
    entries_by_module = {
        entry["module_name"]: entry for entry in manifest_payload["adapters"]
    }

    secret_path = resolve_contributor_secret_path(manifest_secret_path)
    ensure_secret_outside_artifacts(secret_path, output_dir)
    seed = load_or_create_contributor_secret(secret_path)

    chunk = target_chunk_rows()
    batches = build_batches(records, chunk)
    if not batches:
        raise ProofContractError(
            "native zkLoRA proof generation requires captured invocation records"
        )
    total_params = 0
    for batch in batches:
        entry = entries_by_module.get(batch.module_name)
        if entry is None:
            raise ProofContractError(
                f"module {batch.module_name} missing from pinned adapter manifest"
            )
        write_batch_artifacts(
            output_dir, batch, entry, manifest_commitment, seed, chunk
        )
        first = batch.first
        total_params += first.rank * first.in_dim + first.out_dim * first.rank
        if verbose:
            print(
                f"Generated projection batch artifact for {batch.module_name}"
                f"[{batch.start_invocation_index}, "
                f"{batch.start_invocation_index + batch.count})"
            )
    return len(batches), total_params


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _expected_statement_name(statement: dict[str, Any]) -> str:
    return (
        f"{module_slug(statement['session_id'])}."
        f"{module_slug(statement['module_name'])}."
        f"{int(statement['start_invocation_index']):04d}{_STATEMENT_SUFFIX}"
    )


def _transcript_rows_for_statement(
    statement: dict[str, Any],
    transcript_index: dict[tuple[str, str, int], TranscriptEntry],
) -> list[TranscriptEntry]:
    session = statement["session_id"]
    module = statement["module_name"]
    start = int(statement["start_invocation_index"])
    count = int(statement["count"])
    rows: list[TranscriptEntry] = []
    for index in range(start, start + count):
        entry = transcript_index.get((session, module, index))
        if entry is None:
            raise ProofContractError(
                f"statement row {module}#{index} is missing from verifier transcript"
            )
        rows.append(entry)
    return rows


def _unique_transcript_index(
    entries: Iterable[TranscriptEntry],
) -> dict[tuple[str, str, int], TranscriptEntry]:
    index: dict[tuple[str, str, int], TranscriptEntry] = {}
    for entry in entries:
        key = entry.key()
        if key in index:
            raise ProofContractError(f"duplicate transcript row for {key}")
        index[key] = entry
    return index


def verify_v3_artifact_set(
    statement_path: str | os.PathLike[str],
    transcript_index: dict[tuple[str, str, int], TranscriptEntry],
    adapters_index: dict[str, Any],
) -> None:
    # Artifacts are hostile input: structural surprises (missing keys, wrong
    # JSON types, non-numeric strings) must surface as contract errors, not
    # raw KeyError/TypeError crashes. ProofContractError subclasses ValueError
    # and must pass through unwrapped.
    try:
        _verify_v3_artifact_set_checked(
            statement_path, transcript_index, adapters_index
        )
    except ProofContractError:
        raise
    except (KeyError, TypeError, IndexError, ValueError, AttributeError) as exc:
        raise ProofContractError(
            f"malformed schema-3 proof artifact {Path(statement_path).name}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _verify_v3_artifact_set_checked(
    statement_path: str | os.PathLike[str],
    transcript_index: dict[tuple[str, str, int], TranscriptEntry],
    adapters_index: dict[str, Any],
) -> None:
    statement = load_json(statement_path)
    if (
        statement.get("schema_version") != SCHEMA_VERSION_V3
        or statement.get("backend") != BACKEND_ID_V3
        or statement.get("proof_kind") != PROOF_KIND_V3
    ):
        raise ProofContractError("unsupported proof artifact schema/backend")
    for forbidden in ("x", "delta", "lora_commitment", "invocation_index"):
        if forbidden in statement:
            raise ProofContractError(
                f"schema-3 statement must not carry top-level {forbidden!r}"
            )
    if statement.get("statement_digest") != digest_hex(
        _statement_digest_payload(statement)
    ):
        raise ProofContractError("statement digest mismatch")

    statement_path = Path(statement_path)
    if statement_path.name != _expected_statement_name(statement):
        raise ProofContractError(
            f"unexpected statement artifact name: {statement_path.name}"
        )
    prefix = str(statement_path)[: -len(_STATEMENT_SUFFIX)]
    proof_path = Path(f"{prefix}.zklora.proof")
    vk_path = Path(f"{prefix}.zklora.vk")
    pk_path = Path(f"{prefix}.zklora.pk")
    meta_path = Path(f"{prefix}.zklora.meta.json")

    count = int(statement["count"])
    chunk = int(statement.get("target_chunk_rows", 0))
    if chunk < 1 or chunk > MAX_BATCH_ROWS:
        raise ProofContractError("statement target_chunk_rows out of bounds")
    if count < 1 or count > chunk:
        raise ProofContractError("statement row count exceeds its target chunk")
    if len(statement.get("row_digests", [])) != count:
        raise ProofContractError("statement row digest count mismatch")

    module = statement["module_name"]
    entry = adapters_index.get(module)
    if entry is None or _MANIFEST_META_KEY == module:
        raise ProofContractError(
            "statement module is missing from expected adapter manifest"
        )
    fixed_point = FixedPointConfig(**statement["fixed_point"])
    scaling_num = int(statement["scaling"]["num"])
    scaling_den = int(statement["scaling"]["den"])
    in_dim = int(statement["input_shape"][0])
    out_dim = int(statement["output_shape"][0])
    if "a_commitment" not in entry:
        raise ProofContractError(
            f"schema-3 artifact for {module} requires a schema-3 manifest entry"
        )
    rank = int(entry["rank"])
    _check_v3_dims(in_dim, rank, out_dim, count)
    _check_manifest_entry_matches(
        entry,
        module_name=module,
        rank=rank,
        in_dim=in_dim,
        out_dim=out_dim,
        fixed_point=fixed_point,
        scaling_num=scaling_num,
        scaling_den=scaling_den,
    )
    if statement["adapter_commitment"] != entry["adapter_commitment"]:
        raise ProofContractError("statement adapter commitment mismatch")
    manifest_meta = adapters_index.get(_MANIFEST_META_KEY) or {}
    pinned_commitment = manifest_meta.get("manifest_commitment")
    if pinned_commitment is None:
        raise ProofContractError(
            "schema-3 artifacts require a pinned schema-3 manifest commitment"
        )
    if statement["manifest_commitment"] != pinned_commitment:
        raise ProofContractError("statement manifest commitment mismatch")

    expected_circuit = circuit_id_v3(
        in_dim, rank, out_dim, fixed_point, scaling_num, scaling_den, chunk
    )
    if statement["circuit_id"] != expected_circuit:
        raise ProofContractError("statement circuit_id does not match expected circuit")
    if statement["vk_fingerprint"] != vk_fingerprint_v3(expected_circuit):
        raise ProofContractError(
            "statement vk_fingerprint does not match expected circuit"
        )

    rows = _transcript_rows_for_statement(statement, transcript_index)
    x_rows: list[list[int]] = []
    delta_rows: list[list[int]] = []
    for offset, row in enumerate(rows):
        if (
            row.input_shape != statement["input_shape"]
            or row.output_shape != statement["output_shape"]
            or asdict(row.fixed_point) != statement["fixed_point"]
            or row.scaling_num != scaling_num
            or row.scaling_den != scaling_den
        ):
            raise ProofContractError(
                f"transcript row {module}#{row.invocation_index} does not match "
                "statement configuration"
            )
        recomputed = row_digest(
            session_id=row.session_id,
            module_name=row.module_name,
            invocation_index=row.invocation_index,
            input_shape=row.input_shape,
            output_shape=row.output_shape,
            x_row=row.x,
            delta_row=row.delta,
            fixed_point=row.fixed_point,
            scaling_num=row.scaling_num,
            scaling_den=row.scaling_den,
            rank=rank,
            in_dim=in_dim,
            out_dim=out_dim,
            adapter_commitment=statement["adapter_commitment"],
            manifest_commitment=statement["manifest_commitment"],
        )
        if recomputed != statement["row_digests"][offset]:
            raise ProofContractError(
                f"row digest mismatch for {module}#{row.invocation_index}"
            )
        x_rows.append([int(v) for v in row.x])
        delta_rows.append([int(v) for v in row.delta])
    if statement["batch_transcript_digest"] != batch_transcript_digest(
        statement["row_digests"]
    ):
        raise ProofContractError("batch transcript digest mismatch")

    bounds = derive_bounds(x_rows, delta_rows, fixed_point, scaling_num, scaling_den)
    check_bounds_composition(
        bounds, in_dim, rank, fixed_point, scaling_num, scaling_den
    )

    vk = load_json(vk_path)
    if (
        vk.get("schema_version") != SCHEMA_VERSION_V3
        or vk.get("backend") != BACKEND_ID_V3
        or vk.get("circuit_id") != expected_circuit
        or vk.get("vk_fingerprint") != statement["vk_fingerprint"]
    ):
        raise ProofContractError("verification key descriptor mismatch")
    pk = load_json(pk_path)
    if (
        pk.get("schema_version") != SCHEMA_VERSION_V3
        or pk.get("circuit_id") != expected_circuit
    ):
        raise ProofContractError("proving key descriptor mismatch")

    meta = load_json(meta_path)
    proof_bytes = proof_path.read_bytes()
    if (
        meta.get("schema_version") != SCHEMA_VERSION_V3
        or meta.get("backend") != BACKEND_ID_V3
    ):
        raise ProofContractError("unsupported metadata schema/backend")
    if meta.get("proof_kind") != PROOF_KIND_V3:
        raise ProofContractError(f"unsupported proof kind; expected {PROOF_KIND_V3}")
    if meta.get("statement_digest") != statement["statement_digest"]:
        raise ProofContractError("metadata statement digest mismatch")
    if meta.get("statement_file_digest") != digest_hex(statement):
        raise ProofContractError("metadata statement file digest mismatch")
    if meta.get("proof_digest") != hashlib.sha256(proof_bytes).hexdigest():
        raise ProofContractError("metadata proof digest mismatch")
    if meta.get("vk_digest") != digest_hex(vk):
        raise ProofContractError("metadata vk digest mismatch")
    if meta.get("pk_digest") != digest_hex(pk):
        raise ProofContractError("metadata pk digest mismatch")
    if meta.get("manifest_commitment") != statement["manifest_commitment"]:
        raise ProofContractError("metadata manifest commitment mismatch")
    if meta.get("batch_transcript_digest") != statement["batch_transcript_digest"]:
        raise ProofContractError("metadata batch transcript digest mismatch")

    native = _native_v3()
    entry_public = {k: v for k, v in entry.items()}
    if not native.verify_v3(
        canonical_json(statement),
        _rows_json(x_rows, delta_rows),
        canonical_json(entry_public),
        proof_bytes,
    ):
        raise ProofContractError("proof bytes failed projection verification")


@dataclass(frozen=True)
class CoverageClaim:
    session_id: str
    module_name: str
    start: int
    count: int
    source: str

    @property
    def indices(self) -> range:
        return range(self.start, self.start + self.count)


def check_coverage(
    claims: list[CoverageClaim], transcript_entries: list[TranscriptEntry]
) -> None:
    transcript_keys: dict[tuple[str, str], set[int]] = {}
    for entry in transcript_entries:
        key = (entry.session_id, entry.module_name)
        indices = transcript_keys.setdefault(key, set())
        index = int(entry.invocation_index)
        if index in indices:
            duplicate = (entry.session_id, entry.module_name, index)
            raise ProofContractError(f"duplicate transcript row for {duplicate}")
        indices.add(index)

    claim_keys: dict[tuple[str, str], list[CoverageClaim]] = {}
    for claim in claims:
        claim_keys.setdefault((claim.session_id, claim.module_name), []).append(claim)

    # Keys are enumerated from the transcript first so a module with zero
    # artifacts is reported missing, never silently skipped.
    all_keys = sorted(set(transcript_keys) | set(claim_keys))
    for key in all_keys:
        expected = transcript_keys.get(key, set())
        covered: set[int] = set()
        for claim in sorted(claim_keys.get(key, []), key=lambda c: c.start):
            overlap = covered.intersection(claim.indices)
            if overlap:
                raise ProofContractError(
                    f"duplicate proof coverage for {key} rows {sorted(overlap)} "
                    f"(claimed again by {claim.source})"
                )
            covered.update(claim.indices)
        if covered != expected:
            missing = sorted(expected - covered)
            extra = sorted(covered - expected)
            raise ProofContractError(
                f"proof transcript coverage mismatch for {key} "
                f"missing={missing} extra={extra}"
            )


def expand_statement_rows(
    statement: dict[str, Any],
    transcript: str | os.PathLike[str] | Iterable[Any] | None = None,
) -> list[TranscriptEntry]:
    """Resolve a statement (v2 or v3) to its covered transcript rows."""

    schema = statement.get("schema_version")
    if schema == _v2.SCHEMA_VERSION:
        return [_v2.transcript_entry_from_statement(statement)]
    if schema != SCHEMA_VERSION_V3:
        raise ProofContractError(f"unsupported statement schema: {schema}")
    if transcript is None:
        raise ProofContractError(
            "schema-3 statements are digest-only; pass the verifier transcript"
        )
    entries = load_transcript(transcript)
    index = _unique_transcript_index(entries)
    return _transcript_rows_for_statement(statement, index)


def verify_artifacts_mixed(
    proof_dir: str | os.PathLike[str],
    transcript: str | os.PathLike[str] | Iterable[Any],
    expected_adapters: str
    | os.PathLike[str]
    | dict[str, Any]
    | Iterable[dict[str, Any]],
) -> tuple[float, int]:
    import time

    start = time.time()
    reject_secrets_in_artifact_dir(proof_dir)
    entries = load_transcript(transcript)
    transcript_index = _unique_transcript_index(entries)
    adapters_index = load_expected_adapters_any(expected_adapters)
    v2_index = _adapters_index_for_v2(adapters_index)

    statement_files = sorted(Path(proof_dir).glob(f"*{_STATEMENT_SUFFIX}"))
    claims: list[CoverageClaim] = []
    for statement_file in statement_files:
        statement = load_json(statement_file)
        schema = statement.get("schema_version")
        if schema == _v2.SCHEMA_VERSION:
            _v2.verify_artifact_set(statement_file, entries, v2_index)
            claims.append(
                CoverageClaim(
                    session_id=statement["session_id"],
                    module_name=statement["module_name"],
                    start=int(statement["invocation_index"]),
                    count=1,
                    source=str(statement_file),
                )
            )
        elif schema == SCHEMA_VERSION_V3:
            verify_v3_artifact_set(statement_file, transcript_index, adapters_index)
            claims.append(
                CoverageClaim(
                    session_id=statement["session_id"],
                    module_name=statement["module_name"],
                    start=int(statement["start_invocation_index"]),
                    count=int(statement["count"]),
                    source=str(statement_file),
                )
            )
        else:
            raise ProofContractError(
                f"unsupported proof artifact schema: {schema} in {statement_file}"
            )

    check_coverage(claims, entries)
    return time.time() - start, len(statement_files)
