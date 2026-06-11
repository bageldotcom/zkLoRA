//! zklora-sigma-v4: commit-and-prove backend for the exact quantized LoRA
//! delta statement.
//!
//! The v3 halo2 circuit re-witnesses the whole adapter inside every
//! invocation proof: each weight costs a lookup range check plus ~96 rows of
//! in-circuit Poseidon for the adapter commitment, so proving scales with
//! `rank*in + out*rank` per invocation and real LoRA shapes are minutes to
//! infeasible. v4 splits the statement so that all per-weight work happens
//! once per adapter, outside any invocation proof:
//!
//! * Adapter setup (once, at manifest time): Pedersen row commitments to A
//!   and B over a fixed ristretto255 basis, per-weight Pedersen value
//!   commitments, an aggregated Bulletproofs range proof that every weight
//!   lies in the exact `[-value_bound, value_bound]` interval, and a Schnorr
//!   linking proof that the row commitments and the range-proved value
//!   commitments open to the same integers. The pinned adapter commitment
//!   string is the SHA-256 of the deterministic commitment core.
//!
//! * Invocation proof (per statement): the prover commits to the rounding
//!   quotients and remainders of the three-stage quantized pipeline, then
//!   Fiat-Shamir challenges project each matrix equation onto a single
//!   scalar equation over committed values (Schwartz-Zippel over random
//!   gamma/beta), proven with generalized Schnorr proofs plus one rank-sized
//!   quadratic inner-product sigma protocol. Remainders and quotients are
//!   range-bounded with aggregated Bulletproofs. Per-proof work is
//!   O(in + rank + out) group operations -- independent of `rank*in` --
//!   amortized: the first proof for an adapter additionally derives and
//!   caches the adapter commitments, one O(weights) MSM pass.
//!
//! Statement semantics are identical to the v3 circuit: the same canonical
//! half-up rounding, the same exact remainder intervals, the same value and
//! intermediate bounds. (The only language difference is a deliberate
//! domain extension: for multi-limb quotient caps the accepted interval is
//! `[-I, I+1]` instead of `[-I, I]`; the quotient is uniquely determined by
//! the remainder equation either way, so no false statement is accepted --
//! see `soundness notes` below.)
//!
//! Assumptions: binding reduces to discrete log on ristretto255 and
//! collision resistance of SHA-256/BLAKE3 -- the same assumption class as
//! the v3 backend (halo2-IPA over Pasta is discrete-log based, and the v3
//! transcript already used hash-based Fiat-Shamir). Hiding is improved:
//! invocation commitments use fresh uniform blindings (perfectly hiding),
//! while adapter commitments use deterministic blindings derived from the
//! contributor's secret salt keyed with the adapter content -- so they are
//! computationally hiding while the salt stays secret, degrading to the
//! (still binding) unsalted v3 level if it leaks. The v3 Poseidon chain
//! over raw weights was unsalted to begin with.
//!
//! Soundness notes (the chain from accepted proof to exact delta):
//! 1. All commitments are absorbed into the merlin transcript before any
//!    challenge is squeezed, so beta/gamma/zeta are sound Fiat-Shamir
//!    challenges over the full statement and witness commitments.
//! 2. The projected equations hold mod l (group order). Because beta and
//!    gamma are uniform in F_l and every per-element coefficient was fixed
//!    before the challenge, Schwartz-Zippel gives the per-element equations
//!    mod l except with probability ~(rank+out)/l.
//! 3. `validate_field_safety_v4` bounds every integer magnitude that can
//!    appear in a per-element equation by 2^250 < l/4, and the range proofs
//!    pin each committed quotient/remainder into its exact interval, so the
//!    mod-l equations are equations over Z.
//! 4. Over Z, `raw = s*q + rem` with `rem` in the canonical interval
//!    `[-floor(s/2), ceil(s/2)-1]` has a unique solution (q, rem), which is
//!    exactly the canonical half-up rounding used by the reference pipeline,
//!    so delta is the unique exact function of (x, committed adapter).

use blake3;
use bulletproofs::{BulletproofGens, PedersenGens, RangeProof};
use curve25519_dalek_ng::constants::RISTRETTO_BASEPOINT_TABLE;
use curve25519_dalek_ng::ristretto::{
    CompressedRistretto, RistrettoBasepointTable, RistrettoPoint,
};
use curve25519_dalek_ng::scalar::Scalar;
use curve25519_dalek_ng::traits::{Identity, MultiscalarMul, VartimeMultiscalarMul};
use merlin::Transcript;
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::Signed;
use rand_core::OsRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::{Mutex, OnceLock};

use crate::{
    canonical_remainder_interval_int, div_round_canonical_int, AdapterCommitmentInput,
    BoundedCache, FixedPointConfig, NativeError, NativeStatement, NativeWitness,
};

pub const SIGMA_SCHEME_ID: &str = "zklora-sigma-v4-pedersen-ristretto255";
pub const SIGMA_SCHEMA_VERSION: u64 = 3;
/// Integer magnitudes in any projected equation must stay below 2^250 so
/// that sums of two terms cannot reach l/2 (l ~ 2^252.5 on ristretto255).
const FIELD_SAFE_BITS_V4: usize = 250;
/// Max range-proof entries aggregated into one Bulletproof. Chunks are
/// proven/verified in parallel; smaller chunks parallelize better while
/// costing ~700 bytes each.
const BP_CHUNK: usize = 128;
const TRANSCRIPT_LABEL: &[u8] = b"zklora-sigma-v4";

// ---------------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------------

pub(crate) fn pc_gens() -> &'static PedersenGens {
    static GENS: OnceLock<PedersenGens> = OnceLock::new();
    GENS.get_or_init(PedersenGens::default)
}

/// Precomputed table for the blinding generator: fixed-base multiplication
/// is ~5x faster than the generic path and every commitment pays one.
pub(crate) fn blinding_table() -> &'static RistrettoBasepointTable {
    static TABLE: OnceLock<RistrettoBasepointTable> = OnceLock::new();
    TABLE.get_or_init(|| RistrettoBasepointTable::create(&pc_gens().B_blinding))
}

fn bp_gens() -> &'static BulletproofGens {
    static GENS: OnceLock<BulletproofGens> = OnceLock::new();
    GENS.get_or_init(|| BulletproofGens::new(64, BP_CHUNK))
}

/// Vector-commitment basis, independent of the Pedersen pair (B, B~) by
/// construction: each point is hash-to-group output under a dedicated
/// domain, so no discrete-log relation between any of them is known.
///
/// The lock is never held across the parallel generation: a rayon worker
/// that holds a lock while waiting on stolen subtasks can steal another job
/// that blocks on the same lock, deadlocking the pool. Racing growers may
/// duplicate generation work; the points are deterministic, so the longest
/// result simply wins.
pub(crate) fn g_basis(len: usize) -> std::sync::Arc<Vec<RistrettoPoint>> {
    use std::sync::{Arc, RwLock};
    static CACHE: OnceLock<RwLock<Arc<Vec<RistrettoPoint>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| RwLock::new(Arc::new(Vec::new())));
    let current = {
        let guard = cache.read().expect("basis cache poisoned");
        guard.clone()
    };
    if current.len() >= len {
        return current;
    }
    let extra: Vec<RistrettoPoint> = (current.len()..len)
        .into_par_iter()
        .map(|index| {
            let mut hasher = blake3::Hasher::new();
            hasher.update(b"zklora-sigma-v4 row basis");
            hasher.update(&(index as u64).to_le_bytes());
            let mut bytes = [0u8; 64];
            hasher.finalize_xof().fill(&mut bytes);
            RistrettoPoint::from_uniform_bytes(&bytes)
        })
        .collect();
    let mut grown = current.as_ref().clone();
    grown.extend(extra);
    let grown = Arc::new(grown);
    let mut guard = cache.write().expect("basis cache poisoned");
    if guard.len() < grown.len() {
        *guard = grown.clone();
    }
    guard.clone()
}

// ---------------------------------------------------------------------------
// Scalar / point helpers
// ---------------------------------------------------------------------------

fn scalar_from_bigint(value: &BigInt) -> Result<Scalar, NativeError> {
    let magnitude = value.abs().to_biguint().expect("abs is non-negative");
    if magnitude.bits() as usize > FIELD_SAFE_BITS_V4 {
        return Err(NativeError::InvalidDimensions(
            "integer exceeds sigma-v4 field-safe bound".into(),
        ));
    }
    let mut bytes = [0u8; 32];
    let raw = magnitude.to_bytes_le();
    bytes[..raw.len()].copy_from_slice(&raw);
    let scalar = Scalar::from_bytes_mod_order(bytes);
    Ok(if value.sign() == Sign::Minus {
        -scalar
    } else {
        scalar
    })
}

fn scalar_from_i64(value: i64) -> Scalar {
    if value < 0 {
        -Scalar::from(value.unsigned_abs())
    } else {
        Scalar::from(value as u64)
    }
}

/// Variable-time MSM, split across rayon workers for large inputs. Splitting
/// an MSM into chunks and summing partial results is exact; per-chunk
/// Pippenger loses a little batching efficiency but wall time wins ~cores.
pub(crate) fn par_msm(scalars: &[Scalar], points: &[RistrettoPoint]) -> RistrettoPoint {
    debug_assert_eq!(scalars.len(), points.len());
    if scalars.len() < 1024 {
        return RistrettoPoint::vartime_multiscalar_mul(scalars.iter(), points.iter());
    }
    let chunk = scalars.len().div_ceil(rayon::current_num_threads().max(1));
    scalars
        .par_chunks(chunk)
        .zip(points.par_chunks(chunk))
        .map(|(s, p)| RistrettoPoint::vartime_multiscalar_mul(s.iter(), p.iter()))
        .reduce(RistrettoPoint::identity, |a, b| a + b)
}

pub(crate) fn compress(point: &RistrettoPoint) -> [u8; 32] {
    point.compress().to_bytes()
}

pub(crate) fn decompress(bytes: &[u8; 32]) -> Result<RistrettoPoint, NativeError> {
    CompressedRistretto(*bytes)
        .decompress()
        .ok_or_else(|| NativeError::InvalidDimensions("invalid ristretto point".into()))
}

pub(crate) fn random_scalar() -> Scalar {
    use rand_core::SeedableRng;
    thread_local! {
        static RNG: std::cell::RefCell<rand_chacha::ChaCha12Rng> =
            std::cell::RefCell::new(rand_chacha::ChaCha12Rng::from_rng(OsRng).expect("seed rng"));
    }
    RNG.with(|rng| Scalar::random(&mut *rng.borrow_mut()))
}

/// Deterministic blinding factors for adapter commitments: keyed BLAKE3
/// under a key that binds BOTH the contributor's secret salt AND the full
/// adapter content (see `adapter_blinding_key`). Binding the content is
/// essential: if blindings depended only on (salt, domain, index), two
/// same-shaped adapters published by the same contributor would share
/// blindings at every index, and commitment differences C1_i - C2_i =
/// (w1_i - w2_i)*G would leak exact weight differences from the public
/// manifest. The salt never leaves the prover; if it leaks, hiding degrades
/// to the (still binding) unsalted level of the v3 scheme.
fn derived_blinding(key: &[u8; 32], domain: &str, index: u64) -> Scalar {
    let mut hasher = blake3::Hasher::new_keyed(key);
    hasher.update(domain.as_bytes());
    hasher.update(&index.to_le_bytes());
    let mut bytes = [0u8; 64];
    hasher.finalize_xof().fill(&mut bytes);
    Scalar::from_bytes_mod_order_wide(&bytes)
}

/// Per-adapter blinding key: keyed BLAKE3 of the canonical adapter payload
/// (dims, config, scaling, and every weight) under the contributor salt.
/// Distinct adapters therefore get independent blindings even when they
/// share a shape and a salt, while the derivation stays deterministic so
/// manifest commitments and proof-time commitments always agree.
fn adapter_blinding_key(
    input: &AdapterCommitmentInput,
    salt: &[u8; 32],
) -> Result<[u8; 32], NativeError> {
    let mut hasher = blake3::Hasher::new_keyed(salt);
    hasher.update(b"zklora-sigma-v4 adapter blinding key");
    hasher.update(
        serde_json::to_string(input)
            .map_err(|e| NativeError::Json(e.to_string()))?
            .as_bytes(),
    );
    Ok(*hasher.finalize().as_bytes())
}

pub(crate) fn transcript_scalar(transcript: &mut Transcript, label: &'static [u8]) -> Scalar {
    let mut bytes = [0u8; 64];
    transcript.challenge_bytes(label, &mut bytes);
    Scalar::from_bytes_mod_order_wide(&bytes)
}

pub(crate) fn transcript_scalars(
    transcript: &mut Transcript,
    label: &'static [u8],
    count: usize,
) -> Vec<Scalar> {
    (0..count)
        .map(|_| {
            let mut bytes = [0u8; 64];
            transcript.challenge_bytes(label, &mut bytes);
            Scalar::from_bytes_mod_order_wide(&bytes)
        })
        .collect()
}

pub(crate) fn absorb_points(
    transcript: &mut Transcript,
    label: &'static [u8],
    points: &[[u8; 32]],
) {
    transcript.append_u64(b"count", points.len() as u64);
    for point in points {
        transcript.append_message(label, point);
    }
}

fn parse_salt(salt_hex: &str) -> Result<[u8; 32], NativeError> {
    let cleaned = salt_hex.trim().trim_start_matches("0x");
    if cleaned.len() != 64 || !cleaned.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(NativeError::InvalidDimensions(
            "adapter salt must be 32 bytes of hex".into(),
        ));
    }
    let mut salt = [0u8; 32];
    for (i, chunk) in cleaned.as_bytes().chunks(2).enumerate() {
        salt[i] = u8::from_str_radix(std::str::from_utf8(chunk).expect("hex"), 16)
            .map_err(|_| NativeError::InvalidDimensions("invalid salt hex".into()))?;
    }
    Ok(salt)
}

// ---------------------------------------------------------------------------
// Range planning: exact signed intervals via 64-bit limbs + derived twins
// ---------------------------------------------------------------------------

/// One Bulletproof entry: a committed u64 proven to lie in [0, 2^n).
#[derive(Clone)]
pub(crate) struct RangeEntry {
    pub(crate) n: usize,
    pub(crate) value: u64,       // prover side only (0 on verify)
    pub(crate) blinding: Scalar, // prover side only
    pub(crate) commitment: [u8; 32],
}

fn min_bp_bits(max: u64) -> usize {
    for n in [8usize, 16, 32, 64] {
        if n == 64 || max < (1u64 << n) {
            return n;
        }
    }
    64
}

/// Plan for one committed signed integer `v` in the interval
/// `[lower, upper]`: the prover publishes commitments to the 64-bit limbs of
/// `shifted = v - lower`, full limbs are proven in [0, 2^64) (exact: that IS
/// their full width), and the top limb t (max value T = max >> 64*(L-1)) is
/// proven two-sided: t in [0, 2^n) and T - t in [0, 2^n), which pins
/// t in [0, T] exactly. Single-limb values (the only case for remainders,
/// which require exactness) are pinned with zero slack. For multi-limb
/// values the enforced bound is (T+1)*2^(64*(L-1)) - 1, i.e. a slack of
/// 2^64 - 1 - (max mod 2^64) above `max` in general; `limb_plan` REJECTS
/// any multi-limb width whose slack exceeds 1, so the only admitted
/// multi-limb intervals are the quotient caps `max = 2^k - 2` (slack
/// exactly +1, harmless: the quotient is uniquely determined by the
/// exactly-pinned remainder equation, and field-safety margins already
/// cover max + 1).
struct LimbPlan {
    limb_maxes: Vec<u64>, // per limb: max admissible value
}

fn limb_plan(width: &BigInt) -> Result<LimbPlan, NativeError> {
    // width = upper - lower >= 0; max shifted value.
    if width.sign() == Sign::Minus {
        return Err(NativeError::InvalidDimensions(
            "range interval is empty".into(),
        ));
    }
    let max = width.to_biguint().expect("non-negative");
    let bits = max.bits().max(1) as usize;
    if bits > FIELD_SAFE_BITS_V4 {
        return Err(NativeError::InvalidDimensions(
            "range interval exceeds field-safe bound".into(),
        ));
    }
    let limbs = bits.div_ceil(64);
    if limbs > 1 {
        // Enforce the documented at-most-+1 slack: the low limbs cover
        // [0, 2^(64*(L-1)) - 1] exactly, so the slack above `max` is
        // 2^(64*(L-1)) - 1 - (max mod 2^(64*(L-1))).
        let low_mask = (BigUint::from(1u8) << (64 * (limbs - 1))) - 1u8;
        let slack = &low_mask - (&max & &low_mask);
        if slack > BigUint::from(1u8) {
            return Err(NativeError::InvalidDimensions(
                "multi-limb range interval would exceed the +1 slack bound".into(),
            ));
        }
    }
    let mut limb_maxes = vec![u64::MAX; limbs];
    let top = &max >> (64 * (limbs - 1));
    let top: u64 = top.try_into().expect("top limb fits u64");
    limb_maxes[limbs - 1] = top;
    Ok(LimbPlan { limb_maxes })
}

fn limbs_of(shifted: &BigUint, count: usize) -> Vec<u64> {
    let digits = shifted.to_u64_digits();
    (0..count)
        .map(|i| digits.get(i).copied().unwrap_or(0))
        .collect()
}

/// A committed bounded value class shared between prover and verifier:
/// `values.len()` integers, each in `[lower, lower+width]`.
struct BoundedClass {
    lower: BigInt,
    plan: LimbPlan,
    /// limb commitments, value-major: commitments[value][limb]
    commitments: Vec<Vec<[u8; 32]>>,
    /// prover-side openings (shifted limb values and blindings)
    openings: Vec<Vec<(u64, Scalar)>>,
    /// prover-side: blinding of the derived whole-value commitment
    value_blindings: Vec<Scalar>,
}

/// (limb commitments, limb openings, derived value blinding) for one value.
type CommittedValue = (Vec<[u8; 32]>, Vec<(u64, Scalar)>, Scalar);

impl BoundedClass {
    /// Prover-side construction: commit every limb of every shifted value.
    fn commit(values: &[BigInt], lower: &BigInt, width: &BigInt) -> Result<Self, NativeError> {
        let plan = limb_plan(width)?;
        let _ = pc_gens();
        let results: Vec<CommittedValue> = values
            .par_iter()
            .map(|value| {
                let shifted = (value - lower)
                    .to_biguint()
                    .ok_or_else(|| NativeError::InvalidDimensions("value below range".into()))?;
                let limbs = limbs_of(&shifted, plan.limb_maxes.len());
                for (limb, max) in limbs.iter().zip(plan.limb_maxes.iter()) {
                    if limb > max {
                        return Err(NativeError::InvalidDimensions("value above range".into()));
                    }
                }
                let mut commitments = Vec::with_capacity(limbs.len());
                let mut openings = Vec::with_capacity(limbs.len());
                let mut value_blinding = Scalar::zero();
                let mut base = Scalar::one();
                let shift_64 = Scalar::from(u64::MAX) + Scalar::one();
                for limb in &limbs {
                    let blinding = random_scalar();
                    let commitment = &Scalar::from(*limb) * &RISTRETTO_BASEPOINT_TABLE
                        + blinding_table() * &blinding;
                    commitments.push(compress(&commitment));
                    openings.push((*limb, blinding));
                    value_blinding += base * blinding;
                    base *= shift_64;
                }
                Ok((commitments, openings, value_blinding))
            })
            .collect::<Result<_, _>>()?;
        let mut commitments = Vec::with_capacity(values.len());
        let mut openings = Vec::with_capacity(values.len());
        let mut value_blindings = Vec::with_capacity(values.len());
        for (c, o, b) in results {
            commitments.push(c);
            openings.push(o);
            value_blindings.push(b);
        }
        Ok(Self {
            lower: lower.clone(),
            plan,
            commitments,
            openings,
            value_blindings,
        })
    }

    /// Verifier-side construction from published limb commitments.
    fn from_commitments(
        commitments: Vec<Vec<[u8; 32]>>,
        count: usize,
        lower: &BigInt,
        width: &BigInt,
    ) -> Result<Self, NativeError> {
        let plan = limb_plan(width)?;
        if commitments.len() != count
            || commitments
                .iter()
                .any(|limbs| limbs.len() != plan.limb_maxes.len())
        {
            return Err(NativeError::InvalidDimensions(
                "range commitment shape mismatch".into(),
            ));
        }
        Ok(Self {
            lower: lower.clone(),
            plan,
            commitments,
            openings: Vec::new(),
            value_blindings: Vec::new(),
        })
    }

    /// Derived commitment to the signed value: sum_i 2^(64 i) C_i + lower*B.
    fn value_commitment(&self, index: usize) -> Result<RistrettoPoint, NativeError> {
        let mut scalars = Vec::with_capacity(self.plan.limb_maxes.len() + 1);
        let mut points = Vec::with_capacity(self.plan.limb_maxes.len() + 1);
        let mut base = Scalar::one();
        let shift_64 = Scalar::from(u64::MAX) + Scalar::one();
        for limb in &self.commitments[index] {
            scalars.push(base);
            points.push(decompress(limb)?);
            base *= shift_64;
        }
        scalars.push(scalar_from_bigint(&self.lower)?);
        points.push(pc_gens().B);
        Ok(RistrettoPoint::vartime_multiscalar_mul(
            scalars.iter(),
            points.iter(),
        ))
    }

    fn absorb(&self, transcript: &mut Transcript, label: &'static [u8]) {
        transcript.append_u64(b"class-count", self.commitments.len() as u64);
        for limbs in &self.commitments {
            absorb_points(transcript, label, limbs);
        }
    }

    /// Emit range entries: full limbs one-sided (exact), partial top limbs
    /// two-sided with a derived twin commitment max*B - C.
    fn push_entries(&self, entries: &mut Vec<RangeEntry>) -> Result<(), NativeError> {
        let prover_side = !self.openings.is_empty();
        let per_value: Vec<Vec<RangeEntry>> = (0..self.commitments.len())
            .into_par_iter()
            .map(|index| {
                let mut local = Vec::with_capacity(self.plan.limb_maxes.len() * 2);
                for (limb_index, max) in self.plan.limb_maxes.iter().enumerate() {
                    let n = min_bp_bits(*max);
                    let commitment = self.commitments[index][limb_index];
                    let (value, blinding) = if prover_side {
                        self.openings[index][limb_index]
                    } else {
                        (0, Scalar::zero())
                    };
                    local.push(RangeEntry {
                        n,
                        value,
                        blinding,
                        commitment,
                    });
                    if *max != u64::MAX {
                        // Twin: max - v with blinding -b; commitment derived
                        // so the verifier needs no extra published data.
                        let twin = &Scalar::from(*max) * &RISTRETTO_BASEPOINT_TABLE
                            - decompress(&commitment)?;
                        local.push(RangeEntry {
                            n,
                            value: max.wrapping_sub(value),
                            blinding: -blinding,
                            commitment: compress(&twin),
                        });
                    }
                }
                Ok(local)
            })
            .collect::<Result<_, NativeError>>()?;
        for mut local in per_value {
            entries.append(&mut local);
        }
        Ok(())
    }
}

#[derive(Clone, Serialize, Deserialize)]
enum RangeBundle {
    /// Aggregated Bulletproofs grouped by bit width: (n, chunk proofs) in
    /// the deterministic plan order. Compact (log-size) but ~ms per entry;
    /// always used for the one-time adapter setup, where artifact size
    /// matters more than the amortized-to-zero proving time.
    Bulletproofs { groups: Vec<(usize, Vec<Vec<u8>>)> },
    /// Sumcheck-based LogUp lookup argument: microseconds of field work per
    /// entry plus a few MSMs; the default for per-invocation proofs. Both
    /// engines prove the identical statement (every committed value in its
    /// exact interval) under the same discrete-log + Fiat-Shamir
    /// assumptions, and the verifier accepts either.
    LogUp(Box<crate::logup::LogUpProof>),
}

/// Range engine for invocation proofs: LogUp by default for proving speed;
/// ZKLORA_RANGE_ENGINE=bulletproofs opts into compact proofs instead.
fn invocation_range_engine() -> &'static str {
    static ENGINE: OnceLock<String> = OnceLock::new();
    ENGINE.get_or_init(
        || match std::env::var("ZKLORA_RANGE_ENGINE").ok().as_deref() {
            Some("bulletproofs") => "bulletproofs".to_string(),
            _ => "logup".to_string(),
        },
    )
}

fn prove_ranges_with_engine(
    transcript: &Transcript,
    entries: Vec<RangeEntry>,
    engine: &str,
) -> Result<RangeBundle, NativeError> {
    if engine == "logup" {
        let mut fork = transcript.clone();
        return Ok(RangeBundle::LogUp(Box::new(crate::logup::prove(
            &mut fork, &entries,
        )?)));
    }
    prove_ranges(transcript, entries)
}

fn verify_range_bundle(
    transcript: &Transcript,
    entries: Vec<RangeEntry>,
    bundle: &RangeBundle,
) -> Result<(), NativeError> {
    match bundle {
        RangeBundle::LogUp(proof) => {
            let mut fork = transcript.clone();
            crate::logup::verify(&mut fork, &entries, proof)
        }
        RangeBundle::Bulletproofs { .. } => verify_ranges(transcript, entries, bundle),
    }
}

/// Deterministic chunk plan: Bulletproofs aggregation requires a
/// power-of-two party count, so each width group is partitioned into
/// power-of-two chunks of at most BP_CHUNK entries (the binary decomposition
/// of the remainder). No padding parties are ever needed, and chunks
/// parallelize across cores.
fn chunk_plan(len: usize) -> Vec<(usize, usize)> {
    let mut plan = Vec::new();
    let mut start = 0;
    let mut remaining = len;
    while remaining >= BP_CHUNK {
        plan.push((start, BP_CHUNK));
        start += BP_CHUNK;
        remaining -= BP_CHUNK;
    }
    while remaining > 0 {
        let size = 1usize << (usize::BITS - 1 - remaining.leading_zeros()) as usize;
        plan.push((start, size));
        start += size;
        remaining -= size;
    }
    plan
}

fn group_entries(entries: &[RangeEntry]) -> Vec<(usize, Vec<RangeEntry>)> {
    let mut groups = Vec::new();
    for n in [8usize, 16, 32, 64] {
        let group: Vec<RangeEntry> = entries.iter().filter(|e| e.n == n).cloned().collect();
        if !group.is_empty() {
            groups.push((n, group));
        }
    }
    groups
}

/// Aggregate-prove all entries, grouped by bit width and chunked for
/// parallelism. Each chunk transcript is forked from the caller transcript
/// (which has already absorbed every commitment and sigma response) and
/// domain-separated by group and chunk index, so Fiat-Shamir binding covers
/// the full statement.
fn prove_ranges(
    transcript: &Transcript,
    entries: Vec<RangeEntry>,
) -> Result<RangeBundle, NativeError> {
    let proved = group_entries(&entries)
        .into_par_iter()
        .map(|(n, group)| {
            let chunks: Vec<Vec<u8>> = chunk_plan(group.len())
                .into_par_iter()
                .enumerate()
                .map(|(chunk_index, (start, size))| {
                    let chunk = &group[start..start + size];
                    let values: Vec<u64> = chunk.iter().map(|e| e.value).collect();
                    let blindings: Vec<Scalar> = chunk.iter().map(|e| e.blinding).collect();
                    let mut chunk_transcript = transcript.clone();
                    chunk_transcript.append_u64(b"bp-group-bits", n as u64);
                    chunk_transcript.append_u64(b"bp-chunk-index", chunk_index as u64);
                    let (proof, _commitments) = RangeProof::prove_multiple_with_rng(
                        bp_gens(),
                        pc_gens(),
                        &mut chunk_transcript,
                        &values,
                        &blindings,
                        n,
                        &mut OsRng,
                    )
                    .map_err(|e| NativeError::Halo2(format!("range proof: {e:?}")))?;
                    Ok(proof.to_bytes())
                })
                .collect::<Result<_, NativeError>>()?;
            Ok((n, chunks))
        })
        .collect::<Result<Vec<_>, NativeError>>()?;
    Ok(RangeBundle::Bulletproofs { groups: proved })
}

fn verify_ranges(
    transcript: &Transcript,
    entries: Vec<RangeEntry>,
    bundle: &RangeBundle,
) -> Result<(), NativeError> {
    let RangeBundle::Bulletproofs { groups } = bundle else {
        return Err(NativeError::InvalidDimensions(
            "expected bulletproofs range bundle".into(),
        ));
    };
    let expected = group_entries(&entries);
    if expected.len() != groups.len()
        || expected
            .iter()
            .zip(groups.iter())
            .any(|((n_expected, group), (n_actual, chunks))| {
                n_expected != n_actual || chunk_plan(group.len()).len() != chunks.len()
            })
    {
        return Err(NativeError::InvalidDimensions(
            "range bundle shape mismatch".into(),
        ));
    }
    expected
        .into_par_iter()
        .zip(groups.par_iter())
        .try_for_each(|((n, group), (_, chunks))| {
            chunk_plan(group.len())
                .into_par_iter()
                .zip(chunks.par_iter())
                .enumerate()
                .try_for_each(|(chunk_index, ((start, size), proof_bytes))| {
                    let commitments: Vec<CompressedRistretto> = group[start..start + size]
                        .iter()
                        .map(|e| CompressedRistretto(e.commitment))
                        .collect();
                    let proof = RangeProof::from_bytes(proof_bytes)
                        .map_err(|e| NativeError::Halo2(format!("range proof decode: {e:?}")))?;
                    let mut chunk_transcript = transcript.clone();
                    chunk_transcript.append_u64(b"bp-group-bits", n as u64);
                    chunk_transcript.append_u64(b"bp-chunk-index", chunk_index as u64);
                    proof
                        .verify_multiple_with_rng(
                            bp_gens(),
                            pc_gens(),
                            &mut chunk_transcript,
                            &commitments,
                            n,
                            &mut OsRng,
                        )
                        .map_err(|e| NativeError::Halo2(format!("range verify: {e:?}")))
                })
        })
}

// ---------------------------------------------------------------------------
// Adapter setup
// ---------------------------------------------------------------------------

/// Deterministic commitment core: everything the pinned adapter commitment
/// string covers. Field order is fixed by this struct, and only Rust ever
/// serializes it, so `serde_json::to_string` is canonical.
#[derive(Clone, Serialize, Deserialize)]
pub struct AdapterCore {
    pub scheme: String,
    pub schema_version: u64,
    pub in_dim: usize,
    pub rank: usize,
    pub out_dim: usize,
    pub fixed_point: FixedPointConfig,
    pub scaling_num: i64,
    pub scaling_den: i64,
    pub row_commitments_a: Vec<[u8; 32]>,
    pub row_commitments_b: Vec<[u8; 32]>,
    /// Per-weight commitments to (w + value_bound), A row-major then B
    /// row-major. Only used by the one-time adapter range/link proofs.
    pub weight_commitments: Vec<[u8; 32]>,
}

#[derive(Serialize, Deserialize)]
struct LinkProof {
    r1: [u8; 32],
    r2: [u8; 32],
    sigma_z: Vec<Scalar>,
    sigma_s: Scalar,
    sigma_t: Scalar,
}

#[derive(Serialize, Deserialize)]
pub struct AdapterSetupPub {
    pub core: AdapterCore,
    link_a: LinkProof,
    link_b: LinkProof,
    ranges: RangeBundle,
}

struct AdapterSecrets {
    core: AdapterCore,
    /// Precomputed adapter commitment string (SHA-256 over the serialized
    /// core); serializing multi-MB cores on every prove call is measurable.
    commitment: String,
    row_blindings_a: Vec<Scalar>,
    row_blindings_b: Vec<Scalar>,
    weight_blindings: Vec<Scalar>,
}

fn value_bound_int(config: &FixedPointConfig) -> BigInt {
    (BigInt::from(1) << (config.value_bits - 1)) - 1
}

fn intermediate_bound_int(config: &FixedPointConfig) -> BigInt {
    (BigInt::from(1) << (config.intermediate_bits - 1)) - 1
}

fn validate_adapter_input(input: &AdapterCommitmentInput) -> Result<(), NativeError> {
    let rank = input.a.len();
    let in_dim = input.a.first().map_or(0, |row| row.len());
    let out_dim = input.b.len();
    if rank == 0 || in_dim == 0 || out_dim == 0 {
        return Err(NativeError::InvalidDimensions(
            "adapter dimensions must be positive".into(),
        ));
    }
    if input.in_dim != in_dim || input.rank != rank || input.out_dim != out_dim {
        return Err(NativeError::InvalidDimensions(
            "adapter payload dimensions do not match matrices".into(),
        ));
    }
    if input.a.iter().any(|row| row.len() != in_dim) || input.b.iter().any(|row| row.len() != rank)
    {
        return Err(NativeError::InvalidDimensions(
            "adapter matrices are ragged".into(),
        ));
    }
    if input.scaling_den <= 0 {
        return Err(NativeError::InvalidDimensions(
            "scaling denominator must be positive".into(),
        ));
    }
    let config = &input.fixed_point;
    if config.value_bits == 0
        || config.value_bits > 63
        || config.scale_bits >= config.value_bits
        || config.intermediate_bits == 0
    {
        return Err(NativeError::InvalidDimensions(
            "invalid fixed-point bit widths".into(),
        ));
    }
    let bound = value_bound_int(config);
    for value in input.a.iter().flatten().chain(input.b.iter().flatten()) {
        let value = BigInt::from(*value);
        if value < -&bound || value > bound {
            return Err(NativeError::InvalidDimensions(
                "adapter weight exceeds value bound".into(),
            ));
        }
    }
    validate_field_safety_v4(
        in_dim,
        rank,
        out_dim,
        config,
        input.scaling_num,
        input.scaling_den,
    )
}

/// Bound every integer magnitude appearing in a projected per-element
/// equation: |<x, A_k>| <= in*V^2, |s*u + rem| <= s*(2I+2), |<B_j, u>| <=
/// rank*V*(2I+2), |num*w| <= |num|*(2I+2), |den*delta + rem| <= den*(V+1).
/// Hard dimension caps: orders of magnitude beyond any model layer in use
/// (largest practical LoRA target is an embedding of a few hundred thousand
/// rows), but small enough that proving/verification allocations stay
/// bounded even for hostile inputs.
const MAX_DIM: usize = 1 << 22;
const MAX_RANK: usize = 1 << 16;
const MAX_WEIGHTS: usize = 1 << 26;

fn validate_field_safety_v4(
    in_dim: usize,
    rank: usize,
    out_dim: usize,
    config: &FixedPointConfig,
    scaling_num: i64,
    scaling_den: i64,
) -> Result<(), NativeError> {
    if in_dim > MAX_DIM
        || out_dim > MAX_DIM
        || rank > MAX_RANK
        || rank.saturating_mul(in_dim) + out_dim.saturating_mul(rank) > MAX_WEIGHTS
    {
        return Err(NativeError::InvalidDimensions(
            "adapter dimensions exceed sigma-v4 limits".into(),
        ));
    }
    let value_bits = config.value_bits as usize;
    let intermediate_bits = config.intermediate_bits as usize;
    let log_in = usize::BITS as usize - in_dim.leading_zeros() as usize;
    let log_rank = usize::BITS as usize - rank.leading_zeros() as usize;
    let num_bits = 64 - scaling_num.unsigned_abs().leading_zeros() as usize;
    let den_bits = 64 - (scaling_den as u64).leading_zeros() as usize;
    let candidates = [
        2 * value_bits + log_in,
        config.scale_bits as usize + intermediate_bits + 2,
        value_bits + intermediate_bits + log_rank + 2,
        intermediate_bits + num_bits + 2,
        den_bits + value_bits + 2,
    ];
    if candidates.iter().any(|bits| *bits > FIELD_SAFE_BITS_V4) {
        return Err(NativeError::InvalidDimensions(
            "fixed-point config and dimensions exceed sigma-v4 field-safe bounds".into(),
        ));
    }
    Ok(())
}

fn adapter_secrets(
    input: &AdapterCommitmentInput,
    salt: &[u8; 32],
) -> Result<AdapterSecrets, NativeError> {
    validate_adapter_input(input)?;
    let blinding_key = adapter_blinding_key(input, salt)?;
    let rank = input.a.len();
    let in_dim = input.a[0].len();
    let out_dim = input.b.len();
    let gens = pc_gens();
    let basis = g_basis(in_dim.max(rank));
    let value_bound = value_bound_int(&input.fixed_point);
    let shift = scalar_from_bigint(&value_bound)?;

    let row_blindings_a: Vec<Scalar> = (0..rank)
        .map(|k| derived_blinding(&blinding_key, "row-a", k as u64))
        .collect();
    let row_blindings_b: Vec<Scalar> = (0..out_dim)
        .map(|j| derived_blinding(&blinding_key, "row-b", j as u64))
        .collect();
    let row_commitments_a: Vec<[u8; 32]> = input
        .a
        .par_iter()
        .zip(row_blindings_a.par_iter())
        .map(|(row, blinding)| {
            let scalars: Vec<Scalar> = row
                .iter()
                .map(|w| scalar_from_i64(*w))
                .chain(std::iter::once(*blinding))
                .collect();
            let points: Vec<RistrettoPoint> = basis[..in_dim]
                .iter()
                .copied()
                .chain(std::iter::once(gens.B_blinding))
                .collect();
            compress(&RistrettoPoint::multiscalar_mul(
                scalars.iter(),
                points.iter(),
            ))
        })
        .collect();
    let row_commitments_b: Vec<[u8; 32]> = input
        .b
        .par_iter()
        .zip(row_blindings_b.par_iter())
        .map(|(row, blinding)| {
            let scalars: Vec<Scalar> = row
                .iter()
                .map(|w| scalar_from_i64(*w))
                .chain(std::iter::once(*blinding))
                .collect();
            let points: Vec<RistrettoPoint> = basis[..rank]
                .iter()
                .copied()
                .chain(std::iter::once(gens.B_blinding))
                .collect();
            compress(&RistrettoPoint::multiscalar_mul(
                scalars.iter(),
                points.iter(),
            ))
        })
        .collect();

    let weights: Vec<i64> = input
        .a
        .iter()
        .flatten()
        .chain(input.b.iter().flatten())
        .copied()
        .collect();
    let weight_blindings: Vec<Scalar> = (0..weights.len())
        .map(|i| derived_blinding(&blinding_key, "weight", i as u64))
        .collect();
    let weight_commitments: Vec<[u8; 32]> = weights
        .par_iter()
        .zip(weight_blindings.par_iter())
        .map(|(w, blinding)| {
            let shifted = scalar_from_i64(*w) + shift;
            compress(&(&shifted * &RISTRETTO_BASEPOINT_TABLE + blinding_table() * blinding))
        })
        .collect();

    let core = AdapterCore {
        scheme: SIGMA_SCHEME_ID.to_string(),
        schema_version: SIGMA_SCHEMA_VERSION,
        in_dim,
        rank,
        out_dim,
        fixed_point: input.fixed_point.clone(),
        scaling_num: input.scaling_num,
        scaling_den: input.scaling_den,
        row_commitments_a,
        row_commitments_b,
        weight_commitments,
    };
    Ok(AdapterSecrets {
        commitment: adapter_commitment_string(&core),
        core,
        row_blindings_a,
        row_blindings_b,
        weight_blindings,
    })
}

pub fn adapter_commitment_string(core: &AdapterCore) -> String {
    let json = serde_json::to_string(core).expect("core serializes");
    let mut hasher = Sha256::new();
    hasher.update(json.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn adapter_transcript(core: &AdapterCore) -> Transcript {
    let mut transcript = Transcript::new(TRANSCRIPT_LABEL);
    transcript.append_message(b"phase", b"adapter-setup");
    transcript.append_message(
        b"core",
        serde_json::to_string(core)
            .expect("core serializes")
            .as_bytes(),
    );
    transcript
}

/// Schnorr proof that row commitments (basis G, blinding B~) and shifted
/// per-value commitments (base B, blinding B~) open to the same weights,
/// folded over Fiat-Shamir challenges lambda (rows) and mu (columns).
fn prove_link(
    transcript: &mut Transcript,
    label: &'static [u8],
    rows: &[Vec<i64>],
    row_blindings: &[Scalar],
    value_blindings: &[Scalar],
    cols: usize,
) -> LinkProof {
    let gens = pc_gens();
    let basis = g_basis(cols);
    let row_count = rows.len();
    let lambda = transcript_scalars(transcript, label, row_count);
    let mu = transcript_scalars(transcript, label, cols);

    // z_j = sum_k lambda_k rows[k][j]; folded blindings for both forms.
    let mut z = vec![Scalar::zero(); cols];
    let mut s_fold = Scalar::zero();
    let mut t_fold = Scalar::zero();
    for (k, row) in rows.iter().enumerate() {
        for (j, w) in row.iter().enumerate() {
            z[j] += lambda[k] * scalar_from_i64(*w);
            t_fold += lambda[k] * mu[j] * value_blindings[k * cols + j];
        }
        s_fold += lambda[k] * row_blindings[k];
    }

    let rho: Vec<Scalar> = (0..cols).map(|_| random_scalar()).collect();
    let rho_s = random_scalar();
    let rho_t = random_scalar();
    let r1 = RistrettoPoint::multiscalar_mul(
        rho.iter().chain(std::iter::once(&rho_s)),
        basis[..cols]
            .iter()
            .chain(std::iter::once(&gens.B_blinding)),
    );
    let mu_rho: Scalar = mu.iter().zip(rho.iter()).map(|(m, r)| m * r).sum();
    let r2 = &mu_rho * &RISTRETTO_BASEPOINT_TABLE + gens.B_blinding * rho_t;
    transcript.append_message(b"link-r1", &compress(&r1));
    transcript.append_message(b"link-r2", &compress(&r2));
    let c = transcript_scalar(transcript, b"link-challenge");

    LinkProof {
        r1: compress(&r1),
        r2: compress(&r2),
        sigma_z: rho.iter().zip(z.iter()).map(|(r, zj)| r + c * zj).collect(),
        sigma_s: rho_s + c * s_fold,
        sigma_t: rho_t + c * t_fold,
    }
}

fn verify_link(
    transcript: &mut Transcript,
    label: &'static [u8],
    row_commitments: &[[u8; 32]],
    value_commitments: &[[u8; 32]],
    cols: usize,
    shift: &Scalar,
    proof: &LinkProof,
) -> Result<(), NativeError> {
    let gens = pc_gens();
    let basis = g_basis(cols);
    let row_count = row_commitments.len();
    if proof.sigma_z.len() != cols || value_commitments.len() != row_count * cols {
        return Err(NativeError::InvalidDimensions("link proof shape".into()));
    }
    let lambda = transcript_scalars(transcript, label, row_count);
    let mu = transcript_scalars(transcript, label, cols);
    transcript.append_message(b"link-r1", &proof.r1);
    transcript.append_message(b"link-r2", &proof.r2);
    let c = transcript_scalar(transcript, b"link-challenge");

    // P1 = sum_k lambda_k C_k
    let p1 = RistrettoPoint::vartime_multiscalar_mul(
        lambda.iter(),
        row_commitments
            .iter()
            .map(decompress)
            .collect::<Result<Vec<_>, _>>()?
            .iter(),
    );
    // P2' = sum_{k,j} lambda_k mu_j W_kj - shift * Lambda * B, where the W
    // commit shifted weights and Lambda = (sum lambda)(sum mu).
    let mut weights_scalars = Vec::with_capacity(row_count * cols);
    for lk in lambda.iter() {
        for mj in mu.iter() {
            weights_scalars.push(lk * mj);
        }
    }
    let lambda_sum: Scalar = lambda.iter().sum();
    let mu_sum: Scalar = mu.iter().sum();
    let p2 = RistrettoPoint::vartime_multiscalar_mul(
        weights_scalars.iter(),
        value_commitments
            .iter()
            .map(decompress)
            .collect::<Result<Vec<_>, _>>()?
            .iter(),
    ) - &(shift * lambda_sum * mu_sum) * &RISTRETTO_BASEPOINT_TABLE;

    let lhs1 = RistrettoPoint::vartime_multiscalar_mul(
        proof.sigma_z.iter().chain(std::iter::once(&proof.sigma_s)),
        basis[..cols]
            .iter()
            .chain(std::iter::once(&gens.B_blinding)),
    );
    if lhs1 != decompress(&proof.r1)? + p1 * c {
        return Err(NativeError::Halo2(
            "adapter link proof (rows) failed".into(),
        ));
    }
    let mu_sigma: Scalar = mu
        .iter()
        .zip(proof.sigma_z.iter())
        .map(|(m, s)| m * s)
        .sum();
    let lhs2 = &mu_sigma * &RISTRETTO_BASEPOINT_TABLE + gens.B_blinding * proof.sigma_t;
    if lhs2 != decompress(&proof.r2)? + p2 * c {
        return Err(NativeError::Halo2(
            "adapter link proof (values) failed".into(),
        ));
    }
    Ok(())
}

pub(crate) fn adapter_setup(
    input: &AdapterCommitmentInput,
    salt: &[u8; 32],
) -> Result<AdapterSetupPub, NativeError> {
    let secrets = adapter_secrets(input, salt)?;
    let mut transcript = adapter_transcript(&secrets.core);
    let in_dim = secrets.core.in_dim;
    let rank = secrets.core.rank;
    let a_values = rank * in_dim;

    let link_a = prove_link(
        &mut transcript,
        b"link-a",
        &input.a,
        &secrets.row_blindings_a,
        &secrets.weight_blindings[..a_values],
        in_dim,
    );
    let link_b = prove_link(
        &mut transcript,
        b"link-b",
        &input.b,
        &secrets.row_blindings_b,
        &secrets.weight_blindings[a_values..],
        rank,
    );

    // One-time exact range proof: every shifted weight in [0, 2*value_bound].
    let value_bound = value_bound_int(&input.fixed_point);
    let weights: Vec<BigInt> = input
        .a
        .iter()
        .flatten()
        .chain(input.b.iter().flatten())
        .map(|w| BigInt::from(*w))
        .collect();
    let mut class = BoundedClass::from_commitments(
        secrets
            .core
            .weight_commitments
            .iter()
            .map(|c| vec![*c])
            .collect(),
        weights.len(),
        &-&value_bound,
        &(&value_bound * 2),
    )?;
    class.openings = weights
        .iter()
        .zip(secrets.weight_blindings.iter())
        .map(|(w, b)| {
            let shifted: u64 = (w + &value_bound).try_into().expect("shifted weight fits");
            vec![(shifted, *b)]
        })
        .collect();
    let mut entries = Vec::new();
    class.push_entries(&mut entries)?;
    let ranges = prove_ranges(&transcript, entries)?;

    Ok(AdapterSetupPub {
        core: secrets.core,
        link_a,
        link_b,
        ranges,
    })
}

/// Verify the one-time adapter setup proofs. Results are cached by the
/// commitment string (bounded FIFO, oldest-out eviction), so a batch
/// verification pays this once per adapter and a long-running verifier
/// never drops every cached result at once.
pub fn verify_adapter_setup(setup: &AdapterSetupPub) -> Result<(), NativeError> {
    let commitment = adapter_commitment_string(&setup.core);
    static VERIFIED: OnceLock<Mutex<BoundedCache<String, ()>>> = OnceLock::new();
    let verified = VERIFIED.get_or_init(|| Mutex::new(BoundedCache::new(1024)));
    if verified
        .lock()
        .expect("verified cache poisoned")
        .peek(&commitment)
        .is_some()
    {
        return Ok(());
    }

    let core = &setup.core;
    if core.scheme != SIGMA_SCHEME_ID || core.schema_version != SIGMA_SCHEMA_VERSION {
        return Err(NativeError::InvalidDimensions(
            "unsupported adapter setup scheme".into(),
        ));
    }
    if core.row_commitments_a.len() != core.rank
        || core.row_commitments_b.len() != core.out_dim
        || core.weight_commitments.len() != core.rank * core.in_dim + core.out_dim * core.rank
        || core.in_dim == 0
        || core.rank == 0
        || core.out_dim == 0
    {
        return Err(NativeError::InvalidDimensions(
            "adapter setup shape mismatch".into(),
        ));
    }
    if core.scaling_den <= 0
        || core.fixed_point.value_bits == 0
        || core.fixed_point.value_bits > 63
        || core.fixed_point.scale_bits >= core.fixed_point.value_bits
        || core.fixed_point.intermediate_bits == 0
    {
        return Err(NativeError::InvalidDimensions(
            "adapter setup config invalid".into(),
        ));
    }
    validate_field_safety_v4(
        core.in_dim,
        core.rank,
        core.out_dim,
        &core.fixed_point,
        core.scaling_num,
        core.scaling_den,
    )?;

    let mut transcript = adapter_transcript(core);
    let value_bound = value_bound_int(&core.fixed_point);
    let shift = scalar_from_bigint(&value_bound)?;
    let a_values = core.rank * core.in_dim;
    verify_link(
        &mut transcript,
        b"link-a",
        &core.row_commitments_a,
        &core.weight_commitments[..a_values],
        core.in_dim,
        &shift,
        &setup.link_a,
    )?;
    verify_link(
        &mut transcript,
        b"link-b",
        &core.row_commitments_b,
        &core.weight_commitments[a_values..],
        core.rank,
        &shift,
        &setup.link_b,
    )?;

    let class = BoundedClass::from_commitments(
        core.weight_commitments.iter().map(|c| vec![*c]).collect(),
        core.weight_commitments.len(),
        &-&value_bound,
        &(&value_bound * 2),
    )?;
    let mut entries = Vec::new();
    class.push_entries(&mut entries)?;
    verify_ranges(&transcript, entries, &setup.ranges)?;

    verified
        .lock()
        .expect("verified cache poisoned")
        .get_or_create(&commitment, || Ok::<_, NativeError>(()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Invocation proof
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
struct LinearProof {
    r_pa: [u8; 32],
    r_su: [u8; 32],
    r_sra: [u8; 32],
    r_sw: Option<[u8; 32]>,
    r_srf: Option<[u8; 32]>,
    r_eq: Scalar,
    sigma_a: Vec<Scalar>,
    sigma_ba: Scalar,
    sigma_su: Scalar,
    sigma_bsu: Scalar,
    sigma_sra: Scalar,
    sigma_bsra: Scalar,
    sigma_sw: Option<Scalar>,
    sigma_bsw: Option<Scalar>,
    sigma_srf: Option<Scalar>,
    sigma_bsrf: Option<Scalar>,
}

#[derive(Serialize, Deserialize)]
struct QuadProof {
    r_b: [u8; 32],
    r_u: Vec<[u8; 32]>,
    c_t1: [u8; 32],
    c_t0: [u8; 32],
    sigma_b: Vec<Scalar>,
    sigma_bb: Scalar,
    sigma_u: Vec<Scalar>,
    sigma_bu: Vec<Scalar>,
    omega: Scalar,
}

#[derive(Serialize, Deserialize)]
struct InvocationProof {
    c_rema: Vec<Vec<[u8; 32]>>,
    c_u: Vec<Vec<[u8; 32]>>,
    c_w: Option<Vec<Vec<[u8; 32]>>>,
    c_remb: Vec<Vec<[u8; 32]>>,
    c_remf: Option<Vec<Vec<[u8; 32]>>>,
    linear: LinearProof,
    quad: QuadProof,
    ranges: RangeBundle,
}

struct StatementContext {
    statement: NativeStatement,
    in_dim: usize,
    rank: usize,
    out_dim: usize,
    scale: BigInt,
    trivial_scaling: bool,
    has_remf: bool,
}

fn statement_context(statement_json: &str) -> Result<StatementContext, NativeError> {
    let statement: NativeStatement =
        serde_json::from_str(statement_json).map_err(|e| NativeError::Json(e.to_string()))?;
    let in_dim = statement.x.len();
    let out_dim = statement.delta.len();
    let rank = statement.rank;
    if in_dim == 0 || out_dim == 0 || rank == 0 {
        return Err(NativeError::InvalidDimensions(
            "statement dimensions must be positive".into(),
        ));
    }
    let config = &statement.fixed_point;
    if config.value_bits == 0
        || config.value_bits > 63
        || config.scale_bits >= config.value_bits
        || config.intermediate_bits == 0
        || statement.scaling_den <= 0
    {
        return Err(NativeError::InvalidDimensions(
            "invalid statement fixed-point/scaling config".into(),
        ));
    }
    validate_field_safety_v4(
        in_dim,
        rank,
        out_dim,
        config,
        statement.scaling_num,
        statement.scaling_den,
    )?;
    let value_bound = value_bound_int(config);
    for value in statement.x.iter().chain(statement.delta.iter()) {
        let value = BigInt::from(*value);
        if value < -&value_bound || value > value_bound {
            return Err(NativeError::InvalidDimensions(
                "public statement value exceeds bound".into(),
            ));
        }
    }
    let digest = statement
        .statement_digest
        .strip_prefix("0x")
        .unwrap_or(&statement.statement_digest);
    if digest.len() != 64 || !digest.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(NativeError::InvalidDimensions(
            "statement_digest must be 32 bytes of hex".into(),
        ));
    }
    let commitment = &statement.adapter_commitment;
    if commitment.len() != 64 || !commitment.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(NativeError::InvalidDimensions(
            "adapter_commitment must be a sha256 hex string".into(),
        ));
    }
    let trivial_scaling = statement.scaling_num == 1 && statement.scaling_den == 1;
    let has_remf = statement.scaling_den != 1;
    Ok(StatementContext {
        scale: BigInt::from(1) << statement.fixed_point.scale_bits,
        statement,
        in_dim,
        rank,
        out_dim,
        trivial_scaling,
        has_remf,
    })
}

fn invocation_transcript(statement_json: &str, ctx: &StatementContext) -> Transcript {
    let mut transcript = Transcript::new(TRANSCRIPT_LABEL);
    transcript.append_message(b"phase", b"invocation");
    // The canonical statement JSON binds x, delta, dims, config, scaling,
    // the adapter commitment and the upstream statement digest.
    transcript.append_message(
        b"statement",
        blake3::hash(statement_json.as_bytes()).as_bytes(),
    );
    transcript.append_message(b"adapter", ctx.statement.adapter_commitment.as_bytes());
    transcript
}

struct ProjectedCommitments {
    p_a: RistrettoPoint,
    p_b: RistrettoPoint,
    d_u: Vec<RistrettoPoint>,
    d_su: RistrettoPoint,
    d_sra: RistrettoPoint,
    d_sw: Option<RistrettoPoint>,
    d_srb: RistrettoPoint,
    d_srf: Option<RistrettoPoint>,
    /// Public S_w = sum_j beta_j delta_j when scaling is trivial.
    public_sw: Scalar,
    public_d_delta: Scalar,
}

#[allow(clippy::too_many_arguments)]
fn projected_commitments(
    ctx: &StatementContext,
    row_commitments_a: &[[u8; 32]],
    row_commitments_b: &[[u8; 32]],
    rema: &BoundedClass,
    u: &BoundedClass,
    w: Option<&BoundedClass>,
    remb: &BoundedClass,
    remf: Option<&BoundedClass>,
    gamma: &[Scalar],
    beta: &[Scalar],
) -> Result<ProjectedCommitments, NativeError> {
    let combine =
        |scalars: &[Scalar], commitments: &[[u8; 32]]| -> Result<RistrettoPoint, NativeError> {
            Ok(RistrettoPoint::vartime_multiscalar_mul(
                scalars.iter(),
                commitments
                    .iter()
                    .map(decompress)
                    .collect::<Result<Vec<_>, _>>()?
                    .iter(),
            ))
        };
    // One flat MSM per class: sum_j s_j * (sum_i 2^(64 i) C_{j,i} + lower*B)
    // = sum_{j,i} (s_j 2^(64 i)) C_{j,i} + lower*(sum_j s_j)*B.
    let combine_class =
        |scalars: &[Scalar], class: &BoundedClass| -> Result<RistrettoPoint, NativeError> {
            let limbs = class.plan.limb_maxes.len();
            let shift_64 = Scalar::from(u64::MAX) + Scalar::one();
            let mut msm_scalars = Vec::with_capacity(scalars.len() * limbs + 1);
            let mut msm_points = Vec::with_capacity(scalars.len() * limbs + 1);
            for (scalar, limb_row) in scalars.iter().zip(class.commitments.iter()) {
                let mut base = *scalar;
                for limb in limb_row {
                    msm_scalars.push(base);
                    msm_points.push(decompress(limb)?);
                    base *= shift_64;
                }
            }
            msm_scalars.push(scalar_from_bigint(&class.lower)? * scalars.iter().sum::<Scalar>());
            msm_points.push(pc_gens().B);
            Ok(RistrettoPoint::vartime_multiscalar_mul(
                msm_scalars.iter(),
                msm_points.iter(),
            ))
        };
    let d_u: Vec<RistrettoPoint> = (0..ctx.rank)
        .map(|k| u.value_commitment(k))
        .collect::<Result<_, _>>()?;
    let mut d_su = RistrettoPoint::identity();
    for (scalar, point) in gamma.iter().zip(d_u.iter()) {
        d_su += point * scalar;
    }
    let public_sw: Scalar = beta
        .iter()
        .zip(ctx.statement.delta.iter())
        .map(|(b, d)| b * scalar_from_i64(*d))
        .sum();
    Ok(ProjectedCommitments {
        p_a: combine(gamma, row_commitments_a)?,
        p_b: combine(beta, row_commitments_b)?,
        d_su,
        d_u,
        d_sra: combine_class(gamma, rema)?,
        d_sw: w.map(|class| combine_class(beta, class)).transpose()?,
        d_srb: combine_class(beta, remb)?,
        d_srf: remf.map(|class| combine_class(beta, class)).transpose()?,
        public_d_delta: public_sw,
        public_sw,
    })
}

pub fn prove_invocation(
    statement_json: &str,
    witness_json: &str,
    salt: &[u8; 32],
) -> Result<Vec<u8>, NativeError> {
    let timing = std::env::var("ZKLORA_V4_TIMING").is_ok();
    let mut mark = std::time::Instant::now();
    let mut lap = |label: &str| {
        if timing {
            eprintln!("  prove {label}: {:?}", mark.elapsed());
        }
        mark = std::time::Instant::now();
    };
    let ctx = statement_context(statement_json)?;
    let witness: NativeWitness =
        serde_json::from_str(witness_json).map_err(|e| NativeError::Json(e.to_string()))?;
    if witness.a.len() != ctx.rank
        || witness.a.iter().any(|row| row.len() != ctx.in_dim)
        || witness.b.len() != ctx.out_dim
        || witness.b.iter().any(|row| row.len() != ctx.rank)
    {
        return Err(NativeError::InvalidDimensions(
            "witness shape does not match statement".into(),
        ));
    }
    let input = AdapterCommitmentInput {
        schema_version: SIGMA_SCHEMA_VERSION,
        in_dim: ctx.in_dim,
        rank: ctx.rank,
        out_dim: ctx.out_dim,
        fixed_point: ctx.statement.fixed_point.clone(),
        scaling_num: ctx.statement.scaling_num,
        scaling_den: ctx.statement.scaling_den,
        a: witness.a.clone(),
        b: witness.b.clone(),
    };
    let secrets = cached_adapter_secrets(&input, salt)?;
    if secrets.commitment != ctx.statement.adapter_commitment {
        return Err(NativeError::InvalidDimensions(
            "witness does not match statement adapter commitment".into(),
        ));
    }
    lap("setup");

    // --- exact reference pipeline (arbitrary precision) -------------------
    let scale = &ctx.scale;
    let mut u_values = Vec::with_capacity(ctx.rank);
    let mut rema_values = Vec::with_capacity(ctx.rank);
    for row in &witness.a {
        let raw: BigInt = row
            .iter()
            .zip(ctx.statement.x.iter())
            .map(|(w, xi)| BigInt::from(*w) * BigInt::from(*xi))
            .sum();
        let q = div_round_canonical_int(&raw, scale)?;
        rema_values.push(&raw - &q * scale);
        u_values.push(q);
    }
    let mut w_values = Vec::with_capacity(ctx.out_dim);
    let mut remb_values = Vec::with_capacity(ctx.out_dim);
    let mut remf_values = Vec::with_capacity(ctx.out_dim);
    let num = BigInt::from(ctx.statement.scaling_num);
    let den = BigInt::from(ctx.statement.scaling_den);
    for (row, delta) in witness.b.iter().zip(ctx.statement.delta.iter()) {
        let raw: BigInt = row
            .iter()
            .zip(u_values.iter())
            .map(|(weight, u)| BigInt::from(*weight) * u)
            .sum();
        let w = div_round_canonical_int(&raw, scale)?;
        remb_values.push(&raw - &w * scale);
        let scaled = &w * &num;
        let final_delta = div_round_canonical_int(&scaled, &den)?;
        if final_delta != BigInt::from(*delta) {
            return Err(NativeError::InvalidDimensions(
                "witness does not satisfy statement delta".into(),
            ));
        }
        remf_values.push(&scaled - &final_delta * &den);
        w_values.push(w);
    }

    // --- commitments -------------------------------------------------------
    let intermediate_bound = intermediate_bound_int(&ctx.statement.fixed_point);
    let (rema_lower, rema_upper) = canonical_remainder_interval_int(scale);
    let rema_width = &rema_upper - &rema_lower;
    let rema = BoundedClass::commit(&rema_values, &rema_lower, &rema_width)?;
    let u = BoundedClass::commit(&u_values, &-&intermediate_bound, &(&intermediate_bound * 2))?;
    let remb = BoundedClass::commit(&remb_values, &rema_lower, &rema_width)?;
    let w = if ctx.trivial_scaling {
        None
    } else {
        Some(BoundedClass::commit(
            &w_values,
            &-&intermediate_bound,
            &(&intermediate_bound * 2),
        )?)
    };
    let remf = if ctx.has_remf {
        let (lower, upper) = canonical_remainder_interval_int(&den);
        Some(BoundedClass::commit(
            &remf_values,
            &lower,
            &(&upper - &lower),
        )?)
    } else {
        None
    };

    lap("pipeline+commit");
    let mut transcript = invocation_transcript(statement_json, &ctx);
    rema.absorb(&mut transcript, b"c-rema");
    u.absorb(&mut transcript, b"c-u");
    if let Some(class) = &w {
        class.absorb(&mut transcript, b"c-w");
    }
    remb.absorb(&mut transcript, b"c-remb");
    if let Some(class) = &remf {
        class.absorb(&mut transcript, b"c-remf");
    }
    let gamma = transcript_scalars(&mut transcript, b"gamma", ctx.rank);
    let beta = transcript_scalars(&mut transcript, b"beta", ctx.out_dim);
    let zeta = transcript_scalar(&mut transcript, b"zeta");

    let projected = projected_commitments(
        &ctx,
        &secrets.core.row_commitments_a,
        &secrets.core.row_commitments_b,
        &rema,
        &u,
        w.as_ref(),
        &remb,
        remf.as_ref(),
        &gamma,
        &beta,
    )?;

    // Secret openings of the projected quantities.
    let gens = pc_gens();
    let basis = g_basis(ctx.in_dim.max(ctx.rank));
    let scale_scalar = scalar_from_bigint(scale)?;
    let a_proj: Vec<Scalar> = (0..ctx.in_dim)
        .map(|j| {
            (0..ctx.rank)
                .map(|k| gamma[k] * scalar_from_i64(witness.a[k][j]))
                .sum()
        })
        .collect();
    let b_proj: Vec<Scalar> = (0..ctx.rank)
        .map(|k| {
            (0..ctx.out_dim)
                .map(|j| beta[j] * scalar_from_i64(witness.b[j][k]))
                .sum()
        })
        .collect();
    let blinding_a: Scalar = gamma
        .iter()
        .zip(secrets.row_blindings_a.iter())
        .map(|(g, b)| g * b)
        .sum();
    let blinding_b: Scalar = beta
        .iter()
        .zip(secrets.row_blindings_b.iter())
        .map(|(bj, blind)| bj * blind)
        .sum();
    let fold = |scalars: &[Scalar], values: &[BigInt], blindings: &[Scalar]| -> (Scalar, Scalar) {
        let mut value_sum = Scalar::zero();
        let mut blinding_sum = Scalar::zero();
        for ((scalar, value), blinding) in scalars.iter().zip(values).zip(blindings) {
            value_sum += scalar * scalar_from_bigint(value).expect("bounded value");
            blinding_sum += scalar * blinding;
        }
        (value_sum, blinding_sum)
    };
    let (s_u, b_su) = fold(&gamma, &u_values, &u.value_blindings);
    let (s_ra, b_sra) = fold(&gamma, &rema_values, &rema.value_blindings);
    let (_s_rb, b_srb) = fold(&beta, &remb_values, &remb.value_blindings);
    let (s_w, b_sw) = match &w {
        Some(class) => fold(&beta, &w_values, &class.value_blindings),
        None => (projected.public_sw, Scalar::zero()),
    };
    let (_s_rf, b_srf) = match &remf {
        Some(class) => fold(&beta, &remf_values, &class.value_blindings),
        None => (Scalar::zero(), Scalar::zero()),
    };
    let num_scalar = scalar_from_i64(ctx.statement.scaling_num);

    // --- linear Schnorr: E1 + zeta*E3 over (a_proj, S_u, S_ra[, S_w, S_rf])
    let x_scalars: Vec<Scalar> = ctx
        .statement
        .x
        .iter()
        .map(|v| scalar_from_i64(*v))
        .collect();
    let rho_a: Vec<Scalar> = (0..ctx.in_dim).map(|_| random_scalar()).collect();
    let rho_ba = random_scalar();
    let rho_su = random_scalar();
    let rho_bsu = random_scalar();
    let rho_sra = random_scalar();
    let rho_bsra = random_scalar();
    let (rho_sw, rho_bsw, rho_srf, rho_bsrf) = if ctx.trivial_scaling {
        (
            Scalar::zero(),
            Scalar::zero(),
            Scalar::zero(),
            Scalar::zero(),
        )
    } else {
        (
            random_scalar(),
            random_scalar(),
            if ctx.has_remf {
                random_scalar()
            } else {
                Scalar::zero()
            },
            if ctx.has_remf {
                random_scalar()
            } else {
                Scalar::zero()
            },
        )
    };
    let r_pa = RistrettoPoint::multiscalar_mul(
        rho_a.iter().chain(std::iter::once(&rho_ba)),
        basis[..ctx.in_dim]
            .iter()
            .chain(std::iter::once(&gens.B_blinding)),
    );
    let commit_pair = |value: &Scalar, blinding: &Scalar| -> RistrettoPoint {
        value * &RISTRETTO_BASEPOINT_TABLE + blinding_table() * blinding
    };
    let r_su = commit_pair(&rho_su, &rho_bsu);
    let r_sra = commit_pair(&rho_sra, &rho_bsra);
    let r_sw = (!ctx.trivial_scaling).then(|| commit_pair(&rho_sw, &rho_bsw));
    let r_srf = ctx.has_remf.then(|| commit_pair(&rho_srf, &rho_bsrf));
    let x_rho: Scalar = x_scalars.iter().zip(rho_a.iter()).map(|(x, r)| x * r).sum();
    // E1: <x, a> - s*S_u - S_ra = 0 ; E3: num*S_w - den*D_delta - S_rf = 0.
    let mut r_eq = x_rho - scale_scalar * rho_su - rho_sra;
    if !ctx.trivial_scaling {
        r_eq += zeta * (num_scalar * rho_sw - rho_srf);
    }
    transcript.append_message(b"lin-r-pa", &compress(&r_pa));
    transcript.append_message(b"lin-r-su", &compress(&r_su));
    transcript.append_message(b"lin-r-sra", &compress(&r_sra));
    if let Some(point) = &r_sw {
        transcript.append_message(b"lin-r-sw", &compress(point));
    }
    if let Some(point) = &r_srf {
        transcript.append_message(b"lin-r-srf", &compress(point));
    }
    transcript.append_message(b"lin-r-eq", r_eq.as_bytes());
    let c1 = transcript_scalar(&mut transcript, b"lin-challenge");

    let linear = LinearProof {
        r_pa: compress(&r_pa),
        r_su: compress(&r_su),
        r_sra: compress(&r_sra),
        r_sw: r_sw.as_ref().map(compress),
        r_srf: r_srf.as_ref().map(compress),
        r_eq,
        sigma_a: rho_a
            .iter()
            .zip(a_proj.iter())
            .map(|(rho, a)| rho + c1 * a)
            .collect(),
        sigma_ba: rho_ba + c1 * blinding_a,
        sigma_su: rho_su + c1 * s_u,
        sigma_bsu: rho_bsu + c1 * b_su,
        sigma_sra: rho_sra + c1 * s_ra,
        sigma_bsra: rho_bsra + c1 * b_sra,
        sigma_sw: (!ctx.trivial_scaling).then(|| rho_sw + c1 * s_w),
        sigma_bsw: (!ctx.trivial_scaling).then(|| rho_bsw + c1 * b_sw),
        sigma_srf: ctx.has_remf.then(|| rho_srf + c1 * _s_rf),
        sigma_bsrf: ctx.has_remf.then(|| rho_bsrf + c1 * b_srf),
    };

    // --- quadratic sigma: <b_proj, u> = s*S_w + S_rb -----------------------
    let u_scalars: Vec<Scalar> = u_values
        .iter()
        .map(|v| scalar_from_bigint(v).expect("bounded"))
        .collect();
    let rho_b: Vec<Scalar> = (0..ctx.rank).map(|_| random_scalar()).collect();
    let rho_bb = random_scalar();
    let rho_u: Vec<Scalar> = (0..ctx.rank).map(|_| random_scalar()).collect();
    let rho_bu: Vec<Scalar> = (0..ctx.rank).map(|_| random_scalar()).collect();
    let r_b = RistrettoPoint::multiscalar_mul(
        rho_b.iter().chain(std::iter::once(&rho_bb)),
        basis[..ctx.rank]
            .iter()
            .chain(std::iter::once(&gens.B_blinding)),
    );
    let r_u: Vec<RistrettoPoint> = rho_u
        .iter()
        .zip(rho_bu.iter())
        .map(|(value, blinding)| commit_pair(value, blinding))
        .collect();
    let t1: Scalar = rho_b
        .iter()
        .zip(u_scalars.iter())
        .map(|(r, u)| r * u)
        .sum::<Scalar>()
        + b_proj
            .iter()
            .zip(rho_u.iter())
            .map(|(b, r)| b * r)
            .sum::<Scalar>();
    let t0: Scalar = rho_b.iter().zip(rho_u.iter()).map(|(rb, ru)| rb * ru).sum();
    let bt1 = random_scalar();
    let bt0 = random_scalar();
    let c_t1 = commit_pair(&t1, &bt1);
    let c_t0 = commit_pair(&t0, &bt0);
    transcript.append_message(b"quad-r-b", &compress(&r_b));
    absorb_points(
        &mut transcript,
        b"quad-r-u",
        &r_u.iter().map(compress).collect::<Vec<_>>(),
    );
    transcript.append_message(b"quad-c-t1", &compress(&c_t1));
    transcript.append_message(b"quad-c-t0", &compress(&c_t0));
    let c2 = transcript_scalar(&mut transcript, b"quad-challenge");

    // omega opens C_T0 + c2 C_T1 + c2^2 (s*D_Sw + D_Srb) on the blinding base.
    let omega = bt0 + c2 * bt1 + c2 * c2 * (scale_scalar * b_sw + b_srb);
    let quad = QuadProof {
        r_b: compress(&r_b),
        r_u: r_u.iter().map(compress).collect(),
        c_t1: compress(&c_t1),
        c_t0: compress(&c_t0),
        sigma_b: rho_b
            .iter()
            .zip(b_proj.iter())
            .map(|(rho, b)| rho + c2 * b)
            .collect(),
        sigma_bb: rho_bb + c2 * blinding_b,
        sigma_u: rho_u
            .iter()
            .zip(u_scalars.iter())
            .map(|(rho, u)| rho + c2 * u)
            .collect(),
        sigma_bu: rho_bu
            .iter()
            .zip(u.value_blindings.iter())
            .map(|(rho, b)| rho + c2 * b)
            .collect(),
        omega,
    };
    lap("sigma-protocols");
    // Absorb responses so the range-proof transcripts bind the sigma layer.
    for scalar in linear
        .sigma_a
        .iter()
        .chain(quad.sigma_b.iter())
        .chain(quad.sigma_u.iter())
    {
        transcript.append_message(b"sigma-response", scalar.as_bytes());
    }
    transcript.append_message(b"sigma-omega", quad.omega.as_bytes());

    // --- range proofs -------------------------------------------------------
    let mut entries = Vec::new();
    rema.push_entries(&mut entries)?;
    u.push_entries(&mut entries)?;
    if let Some(class) = &w {
        class.push_entries(&mut entries)?;
    }
    remb.push_entries(&mut entries)?;
    if let Some(class) = &remf {
        class.push_entries(&mut entries)?;
    }
    lap("push-entries");
    let ranges = prove_ranges_with_engine(&transcript, entries, invocation_range_engine())?;
    lap("ranges");

    let proof = InvocationProof {
        c_rema: rema.commitments,
        c_u: u.commitments,
        c_w: w.map(|class| class.commitments),
        c_remb: remb.commitments,
        c_remf: remf.map(|class| class.commitments),
        linear,
        quad,
        ranges,
    };
    bincode::serialize(&proof).map_err(|e| NativeError::Json(e.to_string()))
}

pub fn verify_invocation(
    statement_json: &str,
    proof_bytes: &[u8],
    setup: &AdapterSetupPub,
) -> Result<bool, NativeError> {
    let commitment = adapter_commitment_string(&setup.core);
    verify_invocation_cached(statement_json, proof_bytes, setup, &commitment)
}

fn verify_invocation_cached(
    statement_json: &str,
    proof_bytes: &[u8],
    setup: &AdapterSetupPub,
    commitment: &str,
) -> Result<bool, NativeError> {
    match verify_invocation_inner(statement_json, proof_bytes, setup, commitment) {
        Ok(()) => Ok(true),
        Err(NativeError::Halo2(_)) => Ok(false),
        Err(NativeError::InvalidDimensions(_)) => Ok(false),
        Err(other) => Err(other),
    }
}

fn verify_invocation_inner(
    statement_json: &str,
    proof_bytes: &[u8],
    setup: &AdapterSetupPub,
    commitment: &str,
) -> Result<(), NativeError> {
    let ctx = statement_context(statement_json)?;
    let core = &setup.core;
    if commitment != ctx.statement.adapter_commitment {
        return Err(NativeError::InvalidDimensions(
            "statement adapter commitment does not match setup".into(),
        ));
    }
    if core.in_dim != ctx.in_dim
        || core.rank != ctx.rank
        || core.out_dim != ctx.out_dim
        || core.scaling_num != ctx.statement.scaling_num
        || core.scaling_den != ctx.statement.scaling_den
        || serde_json::to_string(&core.fixed_point).map_err(|e| NativeError::Json(e.to_string()))?
            != serde_json::to_string(&ctx.statement.fixed_point)
                .map_err(|e| NativeError::Json(e.to_string()))?
    {
        return Err(NativeError::InvalidDimensions(
            "statement does not match adapter setup".into(),
        ));
    }
    verify_adapter_setup(setup)?;

    // Bound deserialization by the input size: bincode pre-allocates
    // collections from length prefixes, so an attacker-supplied blob could
    // otherwise demand far more memory than its own length. The options
    // mirror `bincode::serialize`'s wire format (fixint, trailing allowed).
    // A decode failure is a proof-level rejection (attacker-controlled
    // bytes, verify returns false), not a caller error like malformed
    // statement JSON -- hence Halo2, which the caller maps to Ok(false).
    let proof: InvocationProof = {
        use bincode::Options;
        bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .allow_trailing_bytes()
            .with_limit(proof_bytes.len().saturating_mul(2) as u64)
            .deserialize(proof_bytes)
            .map_err(|e| NativeError::Halo2(format!("invocation proof decode: {e}")))?
    };
    if (proof.c_w.is_some() == ctx.trivial_scaling) || (proof.c_remf.is_some() != ctx.has_remf) {
        return Err(NativeError::InvalidDimensions(
            "proof scaling structure mismatch".into(),
        ));
    }
    if proof.linear.sigma_a.len() != ctx.in_dim
        || proof.quad.sigma_b.len() != ctx.rank
        || proof.quad.sigma_u.len() != ctx.rank
        || proof.quad.sigma_bu.len() != ctx.rank
        || proof.quad.r_u.len() != ctx.rank
        || proof.linear.r_sw.is_some() == ctx.trivial_scaling
        || proof.linear.sigma_sw.is_some() == ctx.trivial_scaling
        || proof.linear.sigma_bsw.is_some() == ctx.trivial_scaling
        || proof.linear.r_srf.is_some() != ctx.has_remf
        || proof.linear.sigma_srf.is_some() != ctx.has_remf
        || proof.linear.sigma_bsrf.is_some() != ctx.has_remf
    {
        return Err(NativeError::InvalidDimensions(
            "proof shape mismatch".into(),
        ));
    }

    let scale = &ctx.scale;
    let intermediate_bound = intermediate_bound_int(&ctx.statement.fixed_point);
    let (rema_lower, rema_upper) = canonical_remainder_interval_int(scale);
    let rema_width = &rema_upper - &rema_lower;
    let rema =
        BoundedClass::from_commitments(proof.c_rema.clone(), ctx.rank, &rema_lower, &rema_width)?;
    let u = BoundedClass::from_commitments(
        proof.c_u.clone(),
        ctx.rank,
        &-&intermediate_bound,
        &(&intermediate_bound * 2),
    )?;
    let remb = BoundedClass::from_commitments(
        proof.c_remb.clone(),
        ctx.out_dim,
        &rema_lower,
        &rema_width,
    )?;
    let w = proof
        .c_w
        .clone()
        .map(|commitments| {
            BoundedClass::from_commitments(
                commitments,
                ctx.out_dim,
                &-&intermediate_bound,
                &(&intermediate_bound * 2),
            )
        })
        .transpose()?;
    let den = BigInt::from(ctx.statement.scaling_den);
    let remf = proof
        .c_remf
        .clone()
        .map(|commitments| {
            let (lower, upper) = canonical_remainder_interval_int(&den);
            BoundedClass::from_commitments(commitments, ctx.out_dim, &lower, &(&upper - &lower))
        })
        .transpose()?;

    let mut transcript = invocation_transcript(statement_json, &ctx);
    rema.absorb(&mut transcript, b"c-rema");
    u.absorb(&mut transcript, b"c-u");
    if let Some(class) = &w {
        class.absorb(&mut transcript, b"c-w");
    }
    remb.absorb(&mut transcript, b"c-remb");
    if let Some(class) = &remf {
        class.absorb(&mut transcript, b"c-remf");
    }
    let gamma = transcript_scalars(&mut transcript, b"gamma", ctx.rank);
    let beta = transcript_scalars(&mut transcript, b"beta", ctx.out_dim);
    let zeta = transcript_scalar(&mut transcript, b"zeta");

    let projected = projected_commitments(
        &ctx,
        &core.row_commitments_a,
        &core.row_commitments_b,
        &rema,
        &u,
        w.as_ref(),
        &remb,
        remf.as_ref(),
        &gamma,
        &beta,
    )?;

    let gens = pc_gens();
    let basis = g_basis(ctx.in_dim.max(ctx.rank));
    let scale_scalar = scalar_from_bigint(scale)?;
    let num_scalar = scalar_from_i64(ctx.statement.scaling_num);
    let den_scalar = scalar_from_i64(ctx.statement.scaling_den);

    // --- linear Schnorr -----------------------------------------------------
    let linear = &proof.linear;
    transcript.append_message(b"lin-r-pa", &linear.r_pa);
    transcript.append_message(b"lin-r-su", &linear.r_su);
    transcript.append_message(b"lin-r-sra", &linear.r_sra);
    if let Some(point) = &linear.r_sw {
        transcript.append_message(b"lin-r-sw", point);
    }
    if let Some(point) = &linear.r_srf {
        transcript.append_message(b"lin-r-srf", point);
    }
    transcript.append_message(b"lin-r-eq", linear.r_eq.as_bytes());
    let c1 = transcript_scalar(&mut transcript, b"lin-challenge");

    let lhs_pa = RistrettoPoint::vartime_multiscalar_mul(
        linear
            .sigma_a
            .iter()
            .chain(std::iter::once(&linear.sigma_ba)),
        basis[..ctx.in_dim]
            .iter()
            .chain(std::iter::once(&gens.B_blinding)),
    );
    if lhs_pa != decompress(&linear.r_pa)? + projected.p_a * c1 {
        return Err(NativeError::Halo2("linear proof: P_A check failed".into()));
    }
    let check_scalar_commitment = |sigma: &Scalar,
                                   sigma_blind: &Scalar,
                                   announcement: &[u8; 32],
                                   derived: &RistrettoPoint|
     -> Result<(), NativeError> {
        let lhs = sigma * &RISTRETTO_BASEPOINT_TABLE + gens.B_blinding * sigma_blind;
        if lhs != decompress(announcement)? + derived * c1 {
            return Err(NativeError::Halo2(
                "linear proof: scalar check failed".into(),
            ));
        }
        Ok(())
    };
    check_scalar_commitment(
        &linear.sigma_su,
        &linear.sigma_bsu,
        &linear.r_su,
        &projected.d_su,
    )?;
    check_scalar_commitment(
        &linear.sigma_sra,
        &linear.sigma_bsra,
        &linear.r_sra,
        &projected.d_sra,
    )?;
    if let (Some(sigma), Some(blind), Some(announcement), Some(derived)) = (
        &linear.sigma_sw,
        &linear.sigma_bsw,
        &linear.r_sw,
        &projected.d_sw,
    ) {
        check_scalar_commitment(sigma, blind, announcement, derived)?;
    }
    if let (Some(sigma), Some(blind), Some(announcement), Some(derived)) = (
        &linear.sigma_srf,
        &linear.sigma_bsrf,
        &linear.r_srf,
        &projected.d_srf,
    ) {
        check_scalar_commitment(sigma, blind, announcement, derived)?;
    }
    let x_scalars: Vec<Scalar> = ctx
        .statement
        .x
        .iter()
        .map(|v| scalar_from_i64(*v))
        .collect();
    let x_sigma: Scalar = x_scalars
        .iter()
        .zip(linear.sigma_a.iter())
        .map(|(x, s)| x * s)
        .sum();
    let mut lhs_eq = x_sigma - scale_scalar * linear.sigma_su - linear.sigma_sra;
    let mut rhs_eq = linear.r_eq;
    if !ctx.trivial_scaling {
        let sigma_sw = linear.sigma_sw.expect("checked above");
        let sigma_srf = linear.sigma_srf.unwrap_or_else(Scalar::zero);
        lhs_eq += zeta * (num_scalar * sigma_sw - sigma_srf);
        rhs_eq += c1 * zeta * den_scalar * projected.public_d_delta;
    }
    if lhs_eq != rhs_eq {
        return Err(NativeError::Halo2(
            "linear proof: projected equation failed".into(),
        ));
    }

    // --- quadratic sigma ----------------------------------------------------
    let quad = &proof.quad;
    transcript.append_message(b"quad-r-b", &quad.r_b);
    absorb_points(&mut transcript, b"quad-r-u", &quad.r_u);
    transcript.append_message(b"quad-c-t1", &quad.c_t1);
    transcript.append_message(b"quad-c-t0", &quad.c_t0);
    let c2 = transcript_scalar(&mut transcript, b"quad-challenge");

    let lhs_b = RistrettoPoint::vartime_multiscalar_mul(
        quad.sigma_b.iter().chain(std::iter::once(&quad.sigma_bb)),
        basis[..ctx.rank]
            .iter()
            .chain(std::iter::once(&gens.B_blinding)),
    );
    if lhs_b != decompress(&quad.r_b)? + projected.p_b * c2 {
        return Err(NativeError::Halo2("quad proof: P_B check failed".into()));
    }
    for k in 0..ctx.rank {
        let lhs =
            &quad.sigma_u[k] * &RISTRETTO_BASEPOINT_TABLE + gens.B_blinding * quad.sigma_bu[k];
        if lhs != decompress(&quad.r_u[k])? + projected.d_u[k] * c2 {
            return Err(NativeError::Halo2("quad proof: u_k check failed".into()));
        }
    }
    let inner: Scalar = quad
        .sigma_b
        .iter()
        .zip(quad.sigma_u.iter())
        .map(|(b, u)| b * u)
        .sum();
    // <sigma_b, sigma_u> B + omega B~ == C_T0 + c2 C_T1 + c2^2 (s D_Sw + D_Srb)
    // with the public-S_w variant adding c2^2 s S_w B on the right.
    let lhs = &inner * &RISTRETTO_BASEPOINT_TABLE + gens.B_blinding * quad.omega;
    let mut rhs =
        decompress(&quad.c_t0)? + decompress(&quad.c_t1)? * c2 + projected.d_srb * (c2 * c2);
    match &projected.d_sw {
        Some(d_sw) => rhs += d_sw * (c2 * c2 * scale_scalar),
        None => rhs += &(c2 * c2 * scale_scalar * projected.public_sw) * &RISTRETTO_BASEPOINT_TABLE,
    }
    if lhs != rhs {
        return Err(NativeError::Halo2(
            "quad proof: projected equation failed".into(),
        ));
    }

    for scalar in linear
        .sigma_a
        .iter()
        .chain(quad.sigma_b.iter())
        .chain(quad.sigma_u.iter())
    {
        transcript.append_message(b"sigma-response", scalar.as_bytes());
    }
    transcript.append_message(b"sigma-omega", quad.omega.as_bytes());

    // --- range proofs -------------------------------------------------------
    let mut entries = Vec::new();
    rema.push_entries(&mut entries)?;
    u.push_entries(&mut entries)?;
    if let Some(class) = &w {
        class.push_entries(&mut entries)?;
    }
    remb.push_entries(&mut entries)?;
    if let Some(class) = &remf {
        class.push_entries(&mut entries)?;
    }
    verify_range_bundle(&transcript, entries, &proof.ranges)
}

// ---------------------------------------------------------------------------
// Adapter secrets cache (per prove call the setup core is re-derived; cache
// it by salt + witness content so a batch of invocations pays one MSM pass)
// ---------------------------------------------------------------------------

fn cached_adapter_secrets(
    input: &AdapterCommitmentInput,
    salt: &[u8; 32],
) -> Result<std::sync::Arc<AdapterSecrets>, NativeError> {
    static CACHE: OnceLock<Mutex<BoundedCache<[u8; 32], AdapterSecrets>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(BoundedCache::new(8)));
    let mut hasher = blake3::Hasher::new_keyed(salt);
    hasher.update(
        serde_json::to_string(input)
            .map_err(|e| NativeError::Json(e.to_string()))?
            .as_bytes(),
    );
    let key = *hasher.finalize().as_bytes();
    // Never hold the cache lock across the (parallel) derivation: rayon
    // workers blocked on this mutex while the holder waits for stolen
    // subtasks can deadlock the pool. Racing derivations are deterministic,
    // so a duplicate insert is harmless.
    {
        let mut guard = cache.lock().expect("adapter secrets cache poisoned");
        if let Some(hit) = guard.peek(&key) {
            return Ok(hit);
        }
    }
    let secrets = adapter_secrets(input, salt)?;
    let mut guard = cache.lock().expect("adapter secrets cache poisoned");
    guard.get_or_create(&key, || Ok::<_, NativeError>(secrets))
}

// ---------------------------------------------------------------------------
// Public JSON entry points (used by the pyo3 layer and benchmarks)
// ---------------------------------------------------------------------------

pub fn adapter_setup_json(adapter_json: &str, salt_hex: &str) -> Result<String, NativeError> {
    let input: AdapterCommitmentInput =
        serde_json::from_str(adapter_json).map_err(|e| NativeError::Json(e.to_string()))?;
    let salt = parse_salt(salt_hex)?;
    let setup = adapter_setup(&input, &salt)?;
    serde_json::to_string(&setup).map_err(|e| NativeError::Json(e.to_string()))
}

pub fn adapter_commitment_v4_string(
    adapter_json: &str,
    salt_hex: &str,
) -> Result<String, NativeError> {
    let input: AdapterCommitmentInput =
        serde_json::from_str(adapter_json).map_err(|e| NativeError::Json(e.to_string()))?;
    let salt = parse_salt(salt_hex)?;
    let secrets = cached_adapter_secrets(&input, &salt)?;
    Ok(secrets.commitment.clone())
}

pub fn verify_adapter_setup_json(setup_json: &str) -> Result<bool, NativeError> {
    let setup: AdapterSetupPub =
        serde_json::from_str(setup_json).map_err(|e| NativeError::Json(e.to_string()))?;
    match verify_adapter_setup(&setup) {
        Ok(()) => Ok(true),
        Err(NativeError::Halo2(_)) | Err(NativeError::InvalidDimensions(_)) => Ok(false),
        Err(other) => Err(other),
    }
}

pub fn prove_v4_bytes(
    statement_json: &str,
    witness_json: &str,
    salt_hex: &str,
) -> Result<Vec<u8>, NativeError> {
    let salt = parse_salt(salt_hex)?;
    prove_invocation(statement_json, witness_json, &salt)
}

pub fn verify_v4_bytes(
    statement_json: &str,
    proof: &[u8],
    setup_json: &str,
) -> Result<bool, NativeError> {
    // Batch verification passes the same multi-MB setup JSON for every
    // artifact of a module; cache the parsed setup (and its commitment
    // string) by content hash. The hash covers the full JSON, so a
    // different setup can never alias a cache entry.
    type ParsedSetup = (AdapterSetupPub, String);
    static CACHE: OnceLock<Mutex<BoundedCache<[u8; 32], ParsedSetup>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(BoundedCache::new(16)));
    let key = *blake3::hash(setup_json.as_bytes()).as_bytes();
    let cached = {
        let mut guard = cache.lock().expect("setup cache poisoned");
        guard.peek(&key)
    };
    let entry = match cached {
        Some(entry) => entry,
        None => {
            let setup: AdapterSetupPub =
                serde_json::from_str(setup_json).map_err(|e| NativeError::Json(e.to_string()))?;
            let commitment = adapter_commitment_string(&setup.core);
            let mut guard = cache.lock().expect("setup cache poisoned");
            guard.get_or_create(&key, || Ok::<_, NativeError>((setup, commitment)))?
        }
    };
    let (setup, commitment) = entry.as_ref();
    verify_invocation_cached(statement_json, proof, setup, commitment)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_integer::Integer;

    /// Deterministic test case builder: exact reference pipeline in BigInt.
    #[allow(clippy::too_many_arguments)]
    fn make_case(
        in_dim: usize,
        rank: usize,
        out_dim: usize,
        scaling_num: i64,
        scaling_den: i64,
        scale_bits: u32,
        value_bits: u32,
        intermediate_bits: u32,
        seed: u64,
    ) -> (String, String, String, String) {
        let fixed_point = FixedPointConfig {
            scale_bits,
            value_bits,
            intermediate_bits,
        };
        let mut state = seed;
        let magnitude = 1i64 << scale_bits.min(20);
        let mut next = |bound: i64| -> i64 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 16) as i64) % (2 * bound + 1) - bound
        };
        let a: Vec<Vec<i64>> = (0..rank)
            .map(|_| (0..in_dim).map(|_| next(magnitude)).collect())
            .collect();
        let b: Vec<Vec<i64>> = (0..out_dim)
            .map(|_| (0..rank).map(|_| next(magnitude)).collect())
            .collect();
        let x: Vec<i64> = (0..in_dim).map(|_| next(magnitude)).collect();

        let scale = BigInt::from(1) << scale_bits;
        let div_round =
            |n: &BigInt, d: &BigInt| -> BigInt { (n + d / BigInt::from(2u8)).div_floor(d) };
        let u: Vec<BigInt> = a
            .iter()
            .map(|row| {
                let raw: BigInt = row
                    .iter()
                    .zip(x.iter())
                    .map(|(w, xi)| BigInt::from(*w) * BigInt::from(*xi))
                    .sum();
                div_round(&raw, &scale)
            })
            .collect();
        let delta: Vec<i64> = b
            .iter()
            .map(|row| {
                let raw: BigInt = row
                    .iter()
                    .zip(u.iter())
                    .map(|(w, v)| BigInt::from(*w) * v)
                    .sum();
                let w = div_round(&raw, &scale);
                let scaled = w * BigInt::from(scaling_num);
                i64::try_from(div_round(&scaled, &BigInt::from(scaling_den))).expect("delta fits")
            })
            .collect();

        let adapter_input = AdapterCommitmentInput {
            schema_version: SIGMA_SCHEMA_VERSION,
            in_dim,
            rank,
            out_dim,
            fixed_point: fixed_point.clone(),
            scaling_num,
            scaling_den,
            a: a.clone(),
            b: b.clone(),
        };
        let adapter_json = serde_json::to_string(&adapter_input).expect("adapter json");
        let salt_hex = "ab".repeat(32);
        let setup_json = adapter_setup_json(&adapter_json, &salt_hex).expect("setup");
        let commitment =
            adapter_commitment_v4_string(&adapter_json, &salt_hex).expect("commitment");

        let statement = NativeStatement {
            x,
            delta,
            fixed_point,
            rank,
            scaling_num,
            scaling_den,
            adapter_commitment: commitment,
            statement_digest: "ab".repeat(32),
        };
        let witness = NativeWitness { a, b };
        (
            serde_json::to_string(&statement).expect("statement json"),
            serde_json::to_string(&witness).expect("witness json"),
            setup_json,
            salt_hex,
        )
    }

    fn roundtrip(case: (String, String, String, String)) {
        let (statement, witness, setup, salt) = case;
        assert!(verify_adapter_setup_json(&setup).expect("setup verify call"));
        let proof = prove_v4_bytes(&statement, &witness, &salt).expect("prove");
        assert!(verify_v4_bytes(&statement, &proof, &setup).expect("verify call"));
    }

    #[test]
    fn roundtrip_trivial_scaling() {
        roundtrip(make_case(4, 2, 3, 1, 1, 20, 63, 127, 7));
    }

    #[test]
    fn roundtrip_nontrivial_scaling() {
        roundtrip(make_case(5, 2, 4, 3, 2, 8, 24, 60, 11));
    }

    #[test]
    fn roundtrip_negative_scaling_num() {
        roundtrip(make_case(3, 1, 2, -7, 4, 6, 20, 48, 13));
    }

    #[test]
    fn roundtrip_single_limb_intermediate() {
        roundtrip(make_case(4, 2, 2, 1, 1, 4, 16, 40, 17));
    }

    #[test]
    fn roundtrip_rank_one_min_dims() {
        roundtrip(make_case(1, 1, 1, 1, 1, 2, 8, 24, 19));
    }

    #[test]
    fn tampered_delta_rejected() {
        let (statement, witness, setup, salt) = make_case(4, 2, 3, 1, 1, 20, 63, 127, 23);
        let proof = prove_v4_bytes(&statement, &witness, &salt).expect("prove");
        let mut tampered: NativeStatement = serde_json::from_str(&statement).unwrap();
        tampered.delta[0] += 1;
        let tampered_json = serde_json::to_string(&tampered).unwrap();
        assert!(!verify_v4_bytes(&tampered_json, &proof, &setup).expect("verify call"));
    }

    #[test]
    fn tampered_x_rejected() {
        let (statement, witness, setup, salt) = make_case(4, 2, 3, 3, 2, 10, 32, 80, 29);
        let proof = prove_v4_bytes(&statement, &witness, &salt).expect("prove");
        let mut tampered: NativeStatement = serde_json::from_str(&statement).unwrap();
        tampered.x[1] += 1;
        let tampered_json = serde_json::to_string(&tampered).unwrap();
        assert!(!verify_v4_bytes(&tampered_json, &proof, &setup).expect("verify call"));
    }

    #[test]
    fn tampered_proof_bytes_rejected() {
        let (statement, witness, setup, salt) = make_case(4, 2, 3, 1, 1, 20, 63, 127, 31);
        let mut proof = prove_v4_bytes(&statement, &witness, &salt).expect("prove");
        let index = proof.len() / 2;
        proof[index] ^= 0x40;
        // Tampering must yield a clean reject (Ok(false)), never an Err:
        // decode failures and failed checks share the same caller contract.
        assert!(!verify_v4_bytes(&statement, &proof, &setup).expect("verify call"));
    }

    #[test]
    fn garbage_proof_bytes_rejected_cleanly() {
        let (statement, _witness, setup, _salt) = make_case(3, 1, 2, 1, 1, 8, 24, 56, 67);
        assert!(!verify_v4_bytes(&statement, b"not a proof", &setup).expect("verify call"));
        assert!(!verify_v4_bytes(&statement, &[], &setup).expect("verify call"));
    }

    #[test]
    fn wrong_adapter_rejected_at_prove_and_verify() {
        let (statement, witness, _setup, salt) = make_case(4, 2, 3, 1, 1, 20, 63, 127, 37);
        // A different adapter (different seed) has a different commitment.
        let (other_statement, other_witness, other_setup, other_salt) =
            make_case(4, 2, 3, 1, 1, 20, 63, 127, 41);
        // Proving with a witness that does not match the statement commitment
        // must fail outright.
        assert!(prove_v4_bytes(&statement, &other_witness, &other_salt).is_err());
        // A valid proof for one statement must not verify against another
        // adapter's setup.
        let proof = prove_v4_bytes(&other_statement, &other_witness, &other_salt).unwrap();
        assert!(verify_v4_bytes(&other_statement, &proof, &other_setup).unwrap());
        let (_, _, setup, _) = make_case(4, 2, 3, 1, 1, 20, 63, 127, 37);
        assert!(!verify_v4_bytes(&other_statement, &proof, &setup).unwrap());
        let _ = (witness, salt);
    }

    #[test]
    fn wrong_salt_changes_commitment() {
        let (statement, witness, _setup, _salt) = make_case(4, 2, 3, 1, 1, 20, 63, 127, 43);
        let wrong_salt = "cd".repeat(32);
        assert!(prove_v4_bytes(&statement, &witness, &wrong_salt).is_err());
    }

    #[test]
    fn tampered_adapter_setup_rejected() {
        let (_, _, setup, _) = make_case(3, 1, 2, 1, 1, 8, 24, 56, 47);
        let mut parsed: AdapterSetupPub = serde_json::from_str(&setup).unwrap();
        // Flip one weight commitment: links/ranges must fail.
        parsed.core.weight_commitments[0][5] ^= 0x11;
        let tampered = serde_json::to_string(&parsed).unwrap();
        assert!(!verify_adapter_setup_json(&tampered).unwrap_or(false));
    }

    #[test]
    fn statement_digest_must_be_well_formed() {
        let (statement, witness, _setup, salt) = make_case(3, 1, 2, 1, 1, 8, 24, 56, 53);
        let mut parsed: NativeStatement = serde_json::from_str(&statement).unwrap();
        parsed.statement_digest = "zz".repeat(32);
        let bad = serde_json::to_string(&parsed).unwrap();
        assert!(prove_v4_bytes(&bad, &witness, &salt).is_err());
    }

    #[test]
    fn proof_is_bound_to_statement_digest() {
        let (statement, witness, setup, salt) = make_case(4, 2, 3, 1, 1, 20, 63, 127, 59);
        let proof = prove_v4_bytes(&statement, &witness, &salt).expect("prove");
        let mut parsed: NativeStatement = serde_json::from_str(&statement).unwrap();
        parsed.statement_digest = "cd".repeat(32);
        let other = serde_json::to_string(&parsed).unwrap();
        assert!(!verify_v4_bytes(&other, &proof, &setup).expect("verify call"));
    }

    #[test]
    fn out_of_range_public_values_rejected() {
        let (statement, witness, _setup, salt) = make_case(3, 1, 2, 1, 1, 8, 24, 56, 61);
        let mut parsed: NativeStatement = serde_json::from_str(&statement).unwrap();
        parsed.x[0] = 1 << 40; // exceeds 24-bit value bound
        let bad = serde_json::to_string(&parsed).unwrap();
        assert!(prove_v4_bytes(&bad, &witness, &salt).is_err());
    }
}

#[cfg(test)]
mod security_tests {
    use super::*;

    fn adapter(in_dim: usize, rank: usize, out_dim: usize, tweak: i64) -> AdapterCommitmentInput {
        AdapterCommitmentInput {
            schema_version: SIGMA_SCHEMA_VERSION,
            in_dim,
            rank,
            out_dim,
            fixed_point: FixedPointConfig {
                scale_bits: 4,
                value_bits: 16,
                intermediate_bits: 40,
            },
            scaling_num: 1,
            scaling_den: 1,
            a: (0..rank)
                .map(|k| {
                    (0..in_dim)
                        .map(|j| (k * in_dim + j) as i64 + tweak)
                        .collect()
                })
                .collect(),
            b: (0..out_dim)
                .map(|j| (0..rank).map(|k| (j * rank + k) as i64).collect())
                .collect(),
        }
    }

    #[test]
    fn same_shape_adapters_share_no_blindings() {
        // Two same-shaped adapters under one salt differ only in the A
        // matrix. If blindings depended only on (salt, index), every
        // commitment at an unchanged index would collide, and the
        // difference at a changed index would open to (w1 - w2) * B with
        // zero blinding, leaking exact weight differences from the public
        // manifest.
        let salt = [9u8; 32];
        let first = adapter(3, 2, 3, 0);
        let second = adapter(3, 2, 3, 1); // shifts every A entry by 1
        let s1 = adapter_secrets(&first, &salt).unwrap();
        let s2 = adapter_secrets(&second, &salt).unwrap();

        // B matrices are identical across the two adapters; their
        // commitments must still differ at every index (independent
        // blindings), and so must every row commitment.
        let a_values = 2 * 3;
        for (c1, c2) in s1.core.weight_commitments[a_values..]
            .iter()
            .zip(s2.core.weight_commitments[a_values..].iter())
        {
            assert_ne!(
                c1, c2,
                "identical weights must not produce equal commitments"
            );
        }
        for (c1, c2) in s1
            .core
            .row_commitments_b
            .iter()
            .zip(s2.core.row_commitments_b.iter())
        {
            assert_ne!(c1, c2);
        }
        // And the difference at a changed A index must not open to
        // delta_w * B (which a manifest holder could brute-force).
        let delta_w = Scalar::one(); // w2 - w1 = 1 at every A index
        let c1 = decompress(&s1.core.weight_commitments[0]).unwrap();
        let c2 = decompress(&s2.core.weight_commitments[0]).unwrap();
        assert_ne!(c2 - c1, &delta_w * &RISTRETTO_BASEPOINT_TABLE);
    }

    #[test]
    fn oversized_dimensions_rejected() {
        let statement = NativeStatement {
            x: vec![0],
            delta: vec![0],
            fixed_point: FixedPointConfig {
                scale_bits: 4,
                value_bits: 16,
                intermediate_bits: 40,
            },
            rank: MAX_RANK + 1,
            scaling_num: 1,
            scaling_den: 1,
            adapter_commitment: "0".repeat(64),
            statement_digest: "ab".repeat(32),
        };
        let json = serde_json::to_string(&statement).unwrap();
        assert!(statement_context(&json).is_err());
    }

    #[test]
    fn limb_plan_enforces_slack_invariant() {
        // Single-limb widths: always exact, any value admitted.
        assert!(limb_plan(&BigInt::from(12345)).is_ok());
        // Quotient-cap form 2^k - 2: slack exactly +1, admitted.
        assert!(limb_plan(&((BigInt::from(1) << 127) - 2)).is_ok());
        // Sloppy multi-limb width (2^70): low limbs would admit values up
        // to ~2^70 + 2^64 above the cap; must be rejected.
        assert!(limb_plan(&(BigInt::from(1) << 70)).is_err());
    }
}
