# zkLoRA `pedersen-projection-v1` — Protocol Note

Status: **normative for the v3 backend; unaudited.** No Rust prover code may
change the protocol described here without updating this note first. External
review of §6 (range linkage) and §7 (padding soundness) gates the removal of
the legacy rollback hatch and any public security claims (`audit_status`
flips from `"unaudited"` only after that review).

Notation: `p` is the order of the Pasta base field `Fp` (≈ 2^254);
commitments live in the Vesta group (`EqAffine`), whose scalar field is `Fp`.
`s = 2^scale_bits`. Integers embed into `Fp` as `v ↦ v mod p` with negative
values as `−Fp(|v|)`; `FIELD_SAFE_BITS = 250` is the global headroom bound.

---

## 1. Relation

Per batch — public `X ∈ Z^{rows×in_dim}`, `D ∈ Z^{rows×out_dim}`, scale `s`,
scaling `num/den` (`den > 0`, `num ≠ 0`); private `A ∈ Z^{rank×in_dim}`,
`B ∈ Z^{out_dim×rank}`, `U`, `mid_D`, remainders — the prover demonstrates
knowledge of integer witnesses satisfying, entrywise:

1. `X·Aᵀ = s·U + R_u`, with `R_u[i,j] ∈ [−⌊s/2⌋, ⌊(s−1)/2⌋]`
2. `U·Bᵀ = s·mid_D + R_d1`, with `R_d1` in the same canonical interval of `s`
3. `num·mid_D = den·D + R_d2`, with `R_d2 ∈ [−⌊den/2⌋, ⌊(den−1)/2⌋]`

The canonical interval of width `d` contains exactly `d` integers, so given
the left side, `(q, r)` decompositions are **unique**; identities 1–3 are
therefore exactly the v2 semantics `delta = round_div(round_div(round_div(
X·Aᵀ, s)·Bᵀ, s)·num, den)` per `proof_contract._div_round_to_canonical_interval`.
`raw_U`/`raw_D` never materialize as witnesses.

**Deliberate contract change vs v2:** v2 additionally rejected witnesses whose
raw accumulators exceeded `intermediate_bits` (`_rescale`). v3 replaces that
witness-side rejection with the §8 composition checks; prover-side parity is
preserved because the Python layer self-checks every row with
`compute_delta_quantized` before building a statement.

## 2. Public parameters and generators

- Protocol seed `GENERATOR_SEED_ID = "zklora/v3/gen-seed/v1"` is hardcoded in
  verifier code. Artifacts may never supply generators or seeds.
- `G[label][i] = hash_to_curve("zklora-v3:" + seed_id + ":" + label)(le64(i))`
  using the Vesta `CurveExt::hash_to_curve` (simplified SWU; per the
  pasta_curves implementation the output has no known discrete-log relations
  between distinct inputs).
- Labels: witness generator vectors `w:A, w:B, w:U, w:Ru, w:midD, w:Rd1,
  w:Rd2, w:cu, w:cb` (globally indexed); range bit vectors `rb:g`, `rb:h`
  (length `MAX_RANGE_AGG = 2^22`, shared across aggregates); result generator
  `res` (written `G0`); blinding base `blind` (written `H`).
- Security assumption: discrete log in the Vesta group; all generators are
  mutually independent under the hash-to-curve random-oracle model.

## 3. Commitments

`VectorCommitment(v)` for `v ∈ Fp^n`: chunks of `COMMIT_CHUNK = 2^16`
coordinates; chunk `i` is `C_i = Σ_j v[i·2^16 + j]·G[label][i·2^16 + j] + r_i·H`
with chunk blind `r_i`. `C_total = Σ_i C_i` is binding for the full vector
because every global index has a distinct generator. Chunking only bounds MSM
working sets; all sub-protocols consume `C_total` and `Σ r_i`.

Blinding:
- **Per-proof witnesses** (`U, R_u, mid_D, R_d1, R_d2, c_u, c_b`, all masks):
  fresh `OsRng` scalars per proof. Two proofs over identical witnesses must
  differ in every commitment (tested).
- **Manifest commitments** (`A`, `B`): stateless re-derivation at proof time
  via `r = wide_reduce(blake2b_keyed(seed, "zklora/v3/blind:" + module_name +
  ":" + matrix + ":" + chunk_index + ":" + commitment_nonce))`, where
  `commitment_nonce` is fresh 32-byte randomness at each manifest write,
  stored publicly in the entry and hashed into `adapter_commitment`. Keying
  includes the module name and nonce so blinds never repeat across adapters
  in one manifest nor across manifest rewrites; otherwise `C_A − C_A'` would
  be an unblinded commitment to a weight difference. A leaked seed destroys
  *hiding* only — binding and soundness are unaffected.

## 4. Linear-form openings with committed results (LFO)

Given `C_total(v)` and public weights `w`, the prover sends
`Y = ⟨v, w⟩·G0 + r_Y·H` and proves consistency with a blinded inner-product
argument in the style of Hyrax's zk dot-product (Wahby–Tzialla–shelat–
Thaler–Walfish, S&P 2018, §4.1 / Fig. 6) instantiated with the Bulletproofs
folding IPA (Bünz et al., S&P 2018, Protocol 1 with blinding): 2·⌈log2 n⌉
points plus a constant number of scalars. Knowledge soundness is by the
standard forking extractor for the folded relation; zero-knowledge by the
blinding terms in every round.

**Inviolable rule:** no opening ever reveals a scalar evaluation. All identity
checks operate homomorphically on the `Y` commitments. (Cleartext openings
would hand the verifier one exact linear functional of `A` per batch; across
~`rank·in_dim` batches those functionals form a solvable linear system.)

Committed results per batch (fixed transcript order):

| Symbol | Statement | Vector | Weights |
|---|---|---|---|
| `y_A` | id. 1 | `A` | `γ ⊗ (Xᵀα)` |
| `y_U` | id. 1 | `U` | `α ⊗ γ` |
| `y_Ru` | id. 1 | `R_u` | `α ⊗ γ` |
| `y_cu` / `y_cu'` | id. 2 consistency | `c_u` vector / `U` | `(δ⁰..δ^{R−1})` / `α ⊗ (δ⁰..δ^{rank−1})` |
| `y_cb` / `y_cb'` | id. 2 consistency | `c_b` vector / `B` | `(δ'⁰..δ'^{R−1})` / `β ⊗ (δ'⁰..δ'^{rank−1})` |
| `Y_z` | id. 2 | `⟨u*, b*⟩` | two-vector IPA, length `R` |
| `y_M` | id. 2 & 3 | `mid_D` | `α ⊗ β` |
| `y_Rd1` | id. 2 | `R_d1` | `α ⊗ β` |
| `y_Rd2` | id. 3 | `R_d2` | `α ⊗ β` |

with `α_vec = (1, α, …, α^{rows−1})`, `β_vec` over `out_dim`, `γ_vec` over
`rank`, and `R = next_pow2(rank)`.

## 5. Identity checks and projection soundness

The verifier computes `Xᵀα` and `P₃ = den·(αᵀDβ)` itself and checks,
homomorphically over committed results:

- (1) `y_A − s·y_U − y_Ru = 0`
- (2a) `y_cu − y_cu' = 0` and (2b) `y_cb − y_cb' = 0`
- (2c) `Y_z − s·y_M − y_Rd1 = 0`
- (3) `num·y_M − y_Rd2 − P₃·G0` opens to zero (`y_M` reused from id. 2)

**Zero-check batching:** with challenge `ε`, `C* = Σ_i ε^i·(check_i)`; the
prover gives one Schnorr proof of knowledge of `ρ` with `C* = ρ·H`. If any
single check has a non-`H` component, the batched combination does too except
with probability ≤ (#checks)/p over `ε`; a Schnorr proof on `C*` with a
nonzero `G`-component would break DL between `H` and the other generators.

**Projection soundness.** All witness commitments are absorbed before `α, β,
γ, δ, δ'` are squeezed (§9). Suppose identity 1 fails at some entry: define
the nonzero bivariate polynomial `E(a, c) = Σ_{i,j}(X·Aᵀ − s·U − R_u)[i,j]
a^i c^j` of degree ≤ (rows−1, rank−1); the check passes only if
`E(α, γ) = 0`, which by Schwartz–Zippel happens with probability
≤ (rows + rank − 2)/p. Identities 2 and 3 are bounded analogously by
(rows + out_dim − 2)/p and the id-2 consistency checks by (R−1)/p each. Union
bound over the three identities, the consistency checks, the Schnorr, and the
LFO/IPA knowledge errors stays below 2^−240 at `MAX_BATCH_ROWS`/`MAX_V3_DIM`;
Fiat–Shamir grinding over `< 2^100` transcripts leaves > 128 bits. **One
projection; no repetition** — repetition is a small-field Freivalds artifact.

A note on extraction order: the LFO extractor pins each `Y` to the committed
vector and weights; the binding of `C_total` pins the vectors before the
challenges; so a passing transcript yields integer matrices (after §8 lifts
field values to integers) satisfying the identities mod p entrywise.

## 6. Range argument `bp-aggregate-linked-v1`

Standard aggregated Bulletproofs range proof (BP §4.3) over width classes,
with **one substitution** — the single novel composition point of this
protocol. Textbook BP checks the value-aggregation term `Σ_j z^{2+j}·v_j`
against per-value commitments `Σ_j z^{2+j}·V_j`; v3 has no per-value
commitments, so that term is supplied as a committed result `W` from one LFO
on the `VectorCommitment`s, with weights `z^{2+j}` routed through the class's
index map (weights are zero outside the class's real coordinates). The BP
verification equation uses `W` homomorphically exactly where `Σ z^{2+j}V_j`
would appear.

**Extraction lemma (review target).** From a convincing prover one extracts:
(i) via BP's extractor, bit vectors and a value vector `v̂` with every
`v̂_j ∈ [0, 2^n)` whose `z`-weighted sum matches the value committed in `W`
for many `z`; (ii) via the LFO extractor, that `W` commits to
`Σ_j z^{2+j}·v_j` for the vector `v` inside the (binding) witness
commitments. Equality of the two `z`-polynomials at enough sampled `z` forces
`v̂_j = v_j` coefficient-wise (Schwartz–Zippel over `z`, degree ≤ m+1), hence
every committed coordinate lies in `[0, 2^n)`. The composition is a standard
sequential argument-of-knowledge composition; the forking tree is
polynomial-size for logarithmic-round protocols.

Width classes (values shifted to unsigned by public shift constants):

| Class | Members (shifted) | width n |
|---|---|---|
| `rem` | `R_u + ⌊s/2⌋`, `R_d1 + ⌊s/2⌋` | `scale_bits` (exact: the canonical interval of `s` has exactly `2^scale_bits` values) |
| `u` | `U + B_U` | `bitlen(2·B_U)` |
| `midD` | `mid_D + B_M` | `bitlen(2·B_M)` |
| `rd2` | `R_d2 + ⌊den/2⌋` | `max(1, ⌈log2 den⌉)`, two-sided |
| `ab` (manifest, one-time) | `A + value_bound`, `B + value_bound` | `value_bits` |

- `B_U = max_i ⌊(Σ_k |X[i,k]|·value_bound + ⌊s/2⌋)/s⌋` and
  `B_M = ⌊(den·max|D| + ⌊den/2⌋)/|num|⌋` are derived from **public** data by
  both sides identically (BigInt arithmetic).
- Non-power-of-two interval `[0, den)` for `rd2`: two-sided trick — a second
  aggregate proves `v + (2^n − den) ∈ [0, 2^n)`; its `W'` derives
  homomorphically from `W` (`W' = W + (2^n − den)·(Σ_j z^{2+j})·G0`), so no
  extra commitment or LFO is needed. `den = 1` keeps the uniform path with a
  zero vector at width 1.
- Aggregates split into sub-aggregates of ≤ `2^22` bits; their value sums add
  homomorphically. The `rb:g`/`rb:h` generator vectors are reused across
  aggregates (binding is per-instance; reuse is safe because each aggregate's
  `A, S` are absorbed into the transcript before its challenges).
- The `ab` class is proven once at manifest creation and verified at pin
  time; without it, nothing bounds `A`/`B` and §8 fails.

## 7. Padding soundness

Normative discipline: every padded slot or index map is verifier-derived from
statement dims, never prover-supplied, and each appearance carries an
argument:

- **Rank-IPA padding (forced zero).** `c_u`, `c_b` are committed at full
  length `R = next_pow2(rank)`. The δ-consistency check runs over the full
  padded length: `⟨c_u, (δ⁰..δ^{R−1})⟩ = ⟨U, α⊗(δ⁰..δ^{rank−1})⟩` is a
  polynomial identity in `δ` of degree ≤ R−1; passing at random `δ` forces
  `u*_j = (Uᵀα)_j` for `j < rank` **and `u*_j = 0` for `j ≥ rank`** (error ≤
  (R−1)/p). Truncating the δ-powers at `rank` would leave the padding slots
  unconstrained while they still contribute `Σ_{j≥rank} u*_j·b*_j` to `Y_z` —
  an additive forgery on identity 2 for any `rank < R`.
- **BP aggregate padding (harmless).** Padding slots of a range aggregate
  correspond to no committed coordinate and get weight zero in `W`'s LFO, so
  `W`'s extracted value covers exactly the real coordinates. A prover placing
  nonzero values in padding slots changes `t(x)`'s constant coefficient away
  from `W + δ(y,z)`; since `T1, T2, W` are absorbed before `x`, the degree-2
  polynomial identity in `x` then fails except with probability ≤ 2/p. Honest
  provers zero the padding for completeness.

## 8. Integer-vs-field argument (against proved bounds)

A one-sided range proof `v + shift ∈ [0, 2^n)` establishes
`v ∈ [−shift, 2^n − 1 − shift]`; the composition checks are stated against
the **proved** bounds, never the nominal ones:

- `P_A = 2^{value_bits−1}` (one more than `value_bound`; documented),
  `P_U = 2^{n_u} − 1 − B_U`, `P_M = 2^{n_m} − 1 − B_M`; `rem` is exact at
  `scale_bits`; `rd2` is exact via the two-sided proof.

Verifier asserts (Rust `verify_v3` is sovereign; Python mirrors):

1. `2·bitlen(P_A) + ⌈log2 in_dim⌉ + 1 ≤ 250`
2. `scale_bits + bitlen(P_U) + 1 ≤ 250`
3. `bitlen(P_U) + bitlen(P_A) + ⌈log2 rank⌉ + 1 ≤ 250`
4. `bitlen(num) + bitlen(P_M) + 1 ≤ 250` and
   `bitlen(den) + bitlen(value_bound) + 1 ≤ 250`

Then every entry of both sides of each identity has magnitude `< 2^250 ≪ p/2`
(public `X`, `D` are bounded by `value_bound` by the v2 quantization
contract), so the mod-p entrywise equality from §5 is equality over `Z`, and
uniqueness of canonical remainders gives exactly the v2 rounding semantics.
The range proofs on `U`/`mid_D` exist **only** for this overflow safety; their
values are already pinned by the canonical ranges of `R_u`/`R_d1`.

## 9. Fiat–Shamir schedule (merlin, Strobe-128)

`Transcript::new(b"zklora-projection-v1")`; every prover message is absorbed
before any later challenge; the verifier recomputes the identical schedule
and rejects any deviation:

1. absorb `circuit_id`, `statement_digest`, `manifest_commitment`,
   `batch_transcript_digest`, dims (rows/in/rank/out), fixed-point, scaling,
   derived bounds (`B_U`, `B_M`, all widths)
2. absorb `a_commitment`, `b_commitment` chunks (pinned manifest)
3. absorb `C_U, C_Ru, C_midD, C_Rd1, C_Rd2` (all chunks, fixed label order)
4. challenges `alpha`, `beta`, `gamma` (`Fp::from_uniform_bytes(64)`)
5. absorb `c_u`, `c_b`
6. challenges `delta-u`, `delta-b`
7. absorb all committed results `Y_*` in the §4 table order
8. challenge `epsilon`; Schnorr zero-check (absorb `R`, challenge, response)
9. LFO/IPA sub-protocols under fixed instance labels (`lfo:yA`, …)
10. range classes in fixed order; per aggregate: absorb `A, S` → challenges
    `y, z` → absorb `T1, T2, W` → challenge `x` → IPA rounds.

The ordering invariants that the security argument needs: witness commitments
precede `α/β/γ` (projection soundness, §5); `c_u/c_b` precede `δ/δ'`
(consistency, §7); `C` precedes `z` and `W` precedes `x` (linkage extraction,
§6).

## 10. Zero-knowledge

All verifier-visible values are: hiding commitments, blinded IPA messages,
committed results, and Schnorr transcripts — each simulatable given the
public statement by standard sigma-protocol simulators (commit to random
values, program challenges), composed across the schedule in §9. Masking
randomness is CSPRNG output, never transcript-derived. **Scope (stated
honestly, and repeated in user docs):** `X` and `D` are public, so the
effective map `ΔW = B·A` is recoverable from ~`in_dim` independent public
rows regardless of the proof system, and the `A/B` factorization is
non-unique (`B·A = (BR)(R^{-1}A)`). Hiding buys early-session privacy and
non-disclosure of the exact quantized weights — nothing more. The verifier is
linear-time (MSM-bound), not succinct.

## 11. Accounting (rows=256, in=768, rank=16, out=2304, s=2^20)

| Vector | entries | width | serialized commitments |
|---|---|---|---|
| `A` / `B` | 12,288 / 36,864 | 63 | 1 + 1 pt (manifest, once) |
| `U` / `R_u` | 4,096 / 4,096 | ~35–75 / 20 | 1 + 1 pt |
| `mid_D` / `R_d1` / `R_d2` | 589,824 each | ~31 / 20 / ⌈log2 den⌉ | 9 + 9 + 9 pts |

≈ 30 commitment points ≈ 1 KB per batch; LFO/IPA/Schnorr ≈ 200–300 points;
range aggregates (~31M committed bits in ≤ 8 sub-aggregates) ≈ 60–110 points
each. Expected proof size 25–60 KB. Verifier ≈ 60–70M point-ops (≈ 2–3 s
multicore); prover ≈ 130M point-ops + BigInt witness build (≈ 10–30 s).
Generator cache ≈ 512 MB at full size (uncompressed affine points are 64 B);
first-use derivation of ~8M hash-to-curve points is parallelized and
benchmarked separately. Performance gates: P0 (M3) proof ≤ 500 KB / verify ≤
30 s / prove ≤ 120 s / ≤ 8 GB; P1 (M4) 100 KB / 10 s / 60 s.

## 12. Blocking questions (answered)

1. *Range proof without per-entry commitments?* §6: BP aggregation with the
   `t₀` term supplied by one LFO on the vector commitments — public
   commitment count is O(#vectors).
2. *How are `mid_D`/`R_d1`/`R_d2` committed?* Full chunked vector
   commitments (9 points each at the representative shape); no per-value
   commitments anywhere.
3. *Exact FS order?* §9, with the three ordering invariants named.
4. *Projection soundness error?* §5: ≤ (rows + out_dim)/p per identity, one
   projection, > 128-bit margin after FS grinding.
5. *Domain separation?* Distinct merlin labels per challenge and per
   sub-protocol instance (§9).
6. *No modular wraparound?* §8, stated against proved bounds `P_*`.
7. *What counts as proof size?* All bytes the verifier needs beyond the
   statement, transcript, and pinned manifest — i.e. the `.zklora.proof`
   file including all commitments (§11).
8. *What does the verifier need from the manifest?* The pinned schema-3
   manifest file: per-module dims/config, `a_commitment`, `b_commitment`,
   `commitment_nonce`, `adapter_commitment`, and the one-time `ab` range
   proof, verified at pin time; `manifest_commitment` is recomputed from the
   pinned payload, never taken from artifacts.
9. *Padding soundness?* §7 (added as a blocking question after review).

## 13. Comparison appendix

- **Same projection relation in Halo2 multi-phase.** Would eliminate the
  hand-rolled FS/IPA/BP and the §6 lemma, keeping one audited proof system.
  Killer: the ~1.8M remainder/mid values per batch need ~4–5M lookup rows
  even at 10-bit limbs → k≈23 on IPA-halo2 → minutes-to-tens-of-minutes
  proving and tens of GB — an order of magnitude outside the gates. (For
  scale: v2's per-row circuit with bit-decomposed range checks is already
  unusable at real shapes, and the zcash halo2 0.3 pin has no multi-phase
  challenge API at all.)
- **Forking dalek `bulletproofs` (ristretto255).** Audited A/S/T/IPA
  machinery and merlin-native transcripts, but its aggregated-range API
  requires per-value commitments — exactly the O(entries) serialization this
  design eliminates — so the `W`-linkage would be a fork of audited code:
  the novel part stays novel and we'd own a patched fork instead of an owned
  crate. Staying on Pasta also reuses the existing integer↔field helpers and
  keeps one curve stack in-tree.
- **Sumcheck/MLE track** (previous design iteration): direct multilinear
  sumchecks for the matmuls with a logUp/Lasso-style lookup for ranges.
  Better asymptotic verifier (√N Hyrax structure), but three bespoke
  components (zk-sumcheck masking, MLE-PCS over Pasta, lookup argument)
  versus this design's single novel lemma; range volume — the shared
  dominant cost — is identical. Reconsider if batch sizes grow ~10× or the
  linear verifier MSM becomes the binding constraint (v3.1 may add
  cross-batch MSM aggregation first).
