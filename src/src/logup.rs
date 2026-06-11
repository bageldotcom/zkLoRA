//! Zero-knowledge LogUp range engine.
//!
//! Proves that a batch of Pedersen-committed values v_e lie in [0, 2^{n_e})
//! by decomposing each into 8-bit digits and proving every digit lies in the
//! table T = [0, 256) with a LogUp lookup argument:
//!
//! `sum_i 1/(alpha - d_i) == sum_t m_t/(alpha - t)`
//!
//! for a Fiat-Shamir challenge alpha drawn after the digit and multiplicity
//! commitments. The rational identity is enforced through committed helper
//! vectors h_i = 1/(alpha - d_i) and g_t = m_t/(alpha - t) whose defining
//! products are checked by two sumchecks over multilinear extensions.
//!
//! Zero knowledge is structural rather than analytic: every sumcheck round
//! polynomial coefficient is sent as a Pedersen commitment, never in the
//! clear, and all sumcheck verifier checks (round consistency, claim
//! chaining, final evaluation, MLE evaluations of the committed vectors,
//! digit recomposition) are linear relations over committed scalars, so they
//! are discharged by one generalized Schnorr proof. The single bilinear
//! relation (the final evaluation product E_h * E_d) uses a standard
//! product sigma protocol. Everything the verifier sees is a perfectly
//! hiding commitment, a uniformly distributed Schnorr response, or public
//! data.
//!
//! Soundness chain: Pedersen binding fixes all committed scalars; the
//! Schnorr extracts openings satisfying every sumcheck verifier relation
//! with honestly Fiat-Shamir-derived round challenges, so standard sumcheck
//! soundness applies; the two sumchecks force h and g to satisfy their
//! defining products on the whole cube (Schwartz-Zippel over tau); the
//! LogUp lemma then forces every digit into the table; and the random
//! projection over rho forces each committed value to equal its digit
//! recomposition mod l, hence to be an integer in [0, 2^{n_e}). All under
//! the same discrete-log + Fiat-Shamir assumptions as the rest of the
//! backend.

use curve25519_dalek_ng::constants::RISTRETTO_BASEPOINT_TABLE;
use curve25519_dalek_ng::ristretto::RistrettoPoint;
use curve25519_dalek_ng::scalar::Scalar;
use curve25519_dalek_ng::traits::VartimeMultiscalarMul;
use merlin::Transcript;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::sigma::{
    absorb_points, blinding_table, compress, decompress, g_basis, par_msm, pc_gens, random_scalar,
    transcript_scalar, transcript_scalars, RangeEntry,
};
use crate::NativeError;

const TABLE_BITS: usize = 8;
const TABLE_SIZE: usize = 1 << TABLE_BITS;

// ---------------------------------------------------------------------------
// Small utilities
// ---------------------------------------------------------------------------

fn batch_invert(values: &[Scalar]) -> Result<Vec<Scalar>, NativeError> {
    // Montgomery batch inversion; rejects zero inputs (negligible-probability
    // challenge collision -- the prover simply errors and the statement can
    // be re-proven, soundness is unaffected).
    let mut prefix = Vec::with_capacity(values.len());
    let mut acc = Scalar::one();
    for value in values {
        if *value == Scalar::zero() {
            return Err(NativeError::Halo2(
                "logup challenge collided with a table value".into(),
            ));
        }
        prefix.push(acc);
        acc *= value;
    }
    let mut inv_acc = acc.invert();
    let mut out = vec![Scalar::zero(); values.len()];
    for index in (0..values.len()).rev() {
        out[index] = inv_acc * prefix[index];
        inv_acc *= values[index];
    }
    Ok(out)
}

/// eq-weight table for challenge vector rs; index bits are consumed most
/// significant first, matching the sumcheck fold order below.
fn eq_table(rs: &[Scalar]) -> Vec<Scalar> {
    let mut table = vec![Scalar::one()];
    for r in rs {
        let mut next = Vec::with_capacity(table.len() * 2);
        for w in &table {
            next.push(w * (Scalar::one() - r));
            next.push(w * r);
        }
        table = next;
    }
    table
}

fn eq_point(a: &[Scalar], b: &[Scalar]) -> Scalar {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y + (Scalar::one() - x) * (Scalar::one() - y))
        .product()
}

fn inner(a: &[Scalar], b: &[Scalar]) -> Scalar {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn poly_eval4(coeffs: &[Scalar; 4], x: &Scalar) -> Scalar {
    coeffs[0] + x * (coeffs[1] + x * (coeffs[2] + x * coeffs[3]))
}

fn vector_commit(values: &[Scalar], blinding: &Scalar) -> RistrettoPoint {
    let basis = g_basis(values.len());
    par_msm(values, &basis[..values.len()]) + blinding_table() * blinding
}

fn scalar_commit(value: &Scalar, blinding: &Scalar) -> RistrettoPoint {
    value * &RISTRETTO_BASEPOINT_TABLE + blinding_table() * blinding
}

// ---------------------------------------------------------------------------
// zk sumcheck with committed round polynomials
// ---------------------------------------------------------------------------

/// Prover state for one sumcheck of G(X) = e(X) * (p(X) * q(X) - w(X)) over
/// {0,1}^m, where all factors are multilinear. Round coefficients are
/// returned as values + blindings; their commitments are absorbed before
/// each challenge.
struct SumcheckRun {
    coeff_values: Vec<[Scalar; 4]>,
    coeff_blindings: Vec<[Scalar; 4]>,
    coeff_commitments: Vec<[[u8; 32]; 4]>,
    rs: Vec<Scalar>,
}

fn coeffs_from_evals(g0: Scalar, g1: Scalar, g2: Scalar, g3: Scalar) -> [Scalar; 4] {
    let inv2 = Scalar::from(2u64).invert();
    let inv6 = Scalar::from(6u64).invert();
    let c3 = (g3 - g2 * Scalar::from(3u64) + g1 * Scalar::from(3u64) - g0) * inv6;
    let c2 = (g2 - g1 * Scalar::from(2u64) + g0 - c3 * Scalar::from(6u64)) * inv2;
    let c0 = g0;
    let c1 = g1 - c0 - c2 - c3;
    [c0, c1, c2, c3]
}

fn fold(values: &mut Vec<Scalar>, r: &Scalar) {
    let half = values.len() / 2;
    for index in 0..half {
        let lo = values[index];
        let hi = values[index + half];
        values[index] = lo + r * (hi - lo);
    }
    values.truncate(half);
}

fn sumcheck_prove(
    transcript: &mut Transcript,
    label: &'static [u8],
    mut e: Vec<Scalar>,
    mut p: Vec<Scalar>,
    mut q: Vec<Scalar>,
    mut w: Vec<Scalar>,
) -> SumcheckRun {
    let mut run = SumcheckRun {
        coeff_values: Vec::new(),
        coeff_blindings: Vec::new(),
        coeff_commitments: Vec::new(),
        rs: Vec::new(),
    };
    while e.len() > 1 {
        let half = e.len() / 2;
        let eval_pair = |index: usize| {
            let mut local = [Scalar::zero(); 4];
            let (e0, e1) = (e[index], e[index + half]);
            let (p0, p1) = (p[index], p[index + half]);
            let (q0, q1) = (q[index], q[index + half]);
            let (w0, w1) = (w[index], w[index + half]);
            let (de, dp, dq, dw) = (e1 - e0, p1 - p0, q1 - q0, w1 - w0);
            let (mut ev, mut pv, mut qv, mut wv) = (e0, p0, q0, w0);
            local[0] = ev * (pv * qv - wv);
            for eval in local.iter_mut().skip(1) {
                ev += de;
                pv += dp;
                qv += dq;
                wv += dw;
                *eval = ev * (pv * qv - wv);
            }
            local
        };
        let evals = if half >= 2048 {
            (0..half).into_par_iter().map(eval_pair).reduce(
                || [Scalar::zero(); 4],
                |mut acc, local| {
                    for (a, l) in acc.iter_mut().zip(local.iter()) {
                        *a += l;
                    }
                    acc
                },
            )
        } else {
            let mut acc = [Scalar::zero(); 4];
            for index in 0..half {
                let local = eval_pair(index);
                for (a, l) in acc.iter_mut().zip(local.iter()) {
                    *a += l;
                }
            }
            acc
        };
        let coeffs = coeffs_from_evals(evals[0], evals[1], evals[2], evals[3]);
        let blindings = [
            random_scalar(),
            random_scalar(),
            random_scalar(),
            random_scalar(),
        ];
        let commitments: [[u8; 32]; 4] =
            std::array::from_fn(|k| compress(&scalar_commit(&coeffs[k], &blindings[k])));
        for commitment in &commitments {
            transcript.append_message(label, commitment);
        }
        let r = transcript_scalar(transcript, label);
        fold(&mut e, &r);
        fold(&mut p, &r);
        fold(&mut q, &r);
        fold(&mut w, &r);
        run.coeff_values.push(coeffs);
        run.coeff_blindings.push(blindings);
        run.coeff_commitments.push(commitments);
        run.rs.push(r);
    }
    run
}

/// Verifier-side challenge replay: absorb the committed round polynomials
/// and re-derive the round challenges.
fn sumcheck_replay(
    transcript: &mut Transcript,
    label: &'static [u8],
    rounds: &[[[u8; 32]; 4]],
) -> Vec<Scalar> {
    rounds
        .iter()
        .map(|commitments| {
            for commitment in commitments {
                transcript.append_message(label, commitment);
            }
            transcript_scalar(transcript, label)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Proof structures
// ---------------------------------------------------------------------------

#[derive(Clone, Serialize, Deserialize)]
struct ProductProof {
    a2: [u8; 32],
    a3: [u8; 32],
    s_y: Scalar,
    s_by: Scalar,
    s_t: Scalar,
}

/// One scalar witness with its commitment blinding (for the mega-Schnorr).
#[derive(Clone, Serialize, Deserialize)]
struct ScalarOpen {
    sigma: Scalar,
    sigma_b: Scalar,
}

#[derive(Clone, Serialize, Deserialize)]
struct MegaSchnorr {
    r_d: [u8; 32],
    r_m: [u8; 32],
    r_h: [u8; 32],
    r_g: [u8; 32],
    r_v: [u8; 32],
    r_coeffs_b: Vec<[[u8; 32]; 4]>,
    r_coeffs_c: Vec<[[u8; 32]; 4]>,
    r_eh: [u8; 32],
    r_ed: [u8; 32],
    r_eg: [u8; 32],
    r_em: [u8; 32],
    r_z: [u8; 32],
    t_constraints: Vec<Scalar>,
    sigma_d: Vec<Scalar>,
    sigma_bd: Scalar,
    sigma_m: Vec<Scalar>,
    sigma_bm: Scalar,
    sigma_h: Vec<Scalar>,
    sigma_bh: Scalar,
    sigma_g: Vec<Scalar>,
    sigma_bg: Scalar,
    sigma_v: ScalarOpen,
    sigma_coeffs_b: Vec<[ScalarOpen; 4]>,
    sigma_coeffs_c: Vec<[ScalarOpen; 4]>,
    sigma_eh: ScalarOpen,
    sigma_ed: ScalarOpen,
    sigma_eg: ScalarOpen,
    sigma_em: ScalarOpen,
    sigma_z: ScalarOpen,
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct LogUpProof {
    c_d: [u8; 32],
    c_m: [u8; 32],
    c_h: [u8; 32],
    c_g: [u8; 32],
    rounds_b: Vec<[[u8; 32]; 4]>,
    rounds_c: Vec<[[u8; 32]; 4]>,
    c_eh: [u8; 32],
    c_ed: [u8; 32],
    c_eg: [u8; 32],
    c_em: [u8; 32],
    c_z: [u8; 32],
    product: ProductProof,
    schnorr: MegaSchnorr,
}

// ---------------------------------------------------------------------------
// Shared linear-constraint system
// ---------------------------------------------------------------------------

/// All witness scalars of the linear system, in one struct so prover
/// randomizers, witnesses, and verifier responses share the evaluation code.
struct LinearWitness {
    d: Vec<Scalar>,
    m: Vec<Scalar>,
    h: Vec<Scalar>,
    g: Vec<Scalar>,
    v: Scalar,
    cb: Vec<[Scalar; 4]>,
    cc: Vec<[Scalar; 4]>,
    eh: Scalar,
    ed: Scalar,
    eg: Scalar,
    em: Scalar,
    z: Scalar,
}

struct PublicData {
    /// recomposition weights over the digit domain (rho_e * 2^{8k})
    weights: Vec<Scalar>,
    alpha: Scalar,
    rs_b: Vec<Scalar>,
    rs_c: Vec<Scalar>,
    eq_r_b: Vec<Scalar>,
    eq_r_c: Vec<Scalar>,
    /// eq(tau, r) for each sumcheck
    eqv_b: Scalar,
    eqv_c: Scalar,
    /// identity MLE of the table evaluated at rs_c
    identv: Scalar,
    /// MLE of the public real-slot indicator evaluated at rs_b
    es: Scalar,
}

/// Evaluate every linear constraint L_i(W). The proof is valid iff
/// L_i(witness) = K_i with K given by `constraint_constants`.
fn constraints_eval(w: &LinearWitness, p: &PublicData) -> Vec<Scalar> {
    let mut out = Vec::new();
    // (A) sum h == sum g
    let sum_h: Scalar = w.h.iter().sum();
    let sum_g: Scalar = w.g.iter().sum();
    out.push(sum_h - sum_g);
    // (D) digit recomposition: <weights, d> == V
    out.push(inner(&p.weights, &w.d) - w.v);
    // sumcheck B round consistency: g_j(0) + g_j(1) == claim_j
    for (j, coeffs) in w.cb.iter().enumerate() {
        let round_sum = coeffs[0] + coeffs[0] + coeffs[1] + coeffs[2] + coeffs[3];
        let claim = if j == 0 {
            Scalar::zero()
        } else {
            poly_eval4(&w.cb[j - 1], &p.rs_b[j - 1])
        };
        out.push(round_sum - claim);
    }
    // sumcheck B final: g_m(r_m) == eqv_b * (alpha*E_h - z - es), where es
    // is the public real-slot-indicator MLE evaluated at r (the -es constant
    // lands in `constraint_constants`).
    let last_b = poly_eval4(
        w.cb.last().expect("at least one round"),
        p.rs_b.last().expect("at least one round"),
    );
    out.push(last_b - p.eqv_b * (p.alpha * w.eh - w.z));
    // sumcheck C round consistency
    for (j, coeffs) in w.cc.iter().enumerate() {
        let round_sum = coeffs[0] + coeffs[0] + coeffs[1] + coeffs[2] + coeffs[3];
        let claim = if j == 0 {
            Scalar::zero()
        } else {
            poly_eval4(&w.cc[j - 1], &p.rs_c[j - 1])
        };
        out.push(round_sum - claim);
    }
    // sumcheck C final: g_8(r_8) == eqv_c * (E_g*(alpha - identv) - E_m)
    let last_c = poly_eval4(
        w.cc.last().expect("table rounds"),
        p.rs_c.last().expect("table rounds"),
    );
    out.push(last_c - p.eqv_c * ((p.alpha - p.identv) * w.eg - w.em));
    // MLE evaluations of the committed vectors
    out.push(w.eh - inner(&p.eq_r_b, &w.h));
    out.push(w.ed - inner(&p.eq_r_b, &w.d));
    out.push(w.eg - inner(&p.eq_r_c, &w.g));
    out.push(w.em - inner(&p.eq_r_c, &w.m));
    out
}

/// Constants K_i such that a valid witness satisfies L_i(witness) = K_i.
fn constraint_constants(p: &PublicData) -> Vec<Scalar> {
    let mut out = vec![Scalar::zero(); 2 + p.rs_b.len()];
    // final B constraint carries the public indicator term: L = -eqv_b * es
    out.push(-(p.eqv_b * p.es));
    out.extend(vec![Scalar::zero(); p.rs_c.len() + 1]);
    out.extend(vec![Scalar::zero(); 4]);
    out
}

impl LinearWitness {
    /// Schnorr randomizers. Pad coordinates of d and h are publicly zero, so
    /// their randomizers are zero too: responses stay zero there and the
    /// announcement MSMs skip them.
    fn random_like(&self, real_len: usize) -> Self {
        let rand_vec = |len: usize| (0..len).map(|_| random_scalar()).collect::<Vec<_>>();
        let rand_vec_real = |len: usize| {
            let mut v = (0..real_len.min(len))
                .map(|_| random_scalar())
                .collect::<Vec<_>>();
            v.resize(len, Scalar::zero());
            v
        };
        let rand_coeffs = |len: usize| {
            (0..len)
                .map(|_| std::array::from_fn(|_| random_scalar()))
                .collect::<Vec<[Scalar; 4]>>()
        };
        Self {
            d: rand_vec_real(self.d.len()),
            m: rand_vec(self.m.len()),
            h: rand_vec_real(self.h.len()),
            g: rand_vec(self.g.len()),
            v: random_scalar(),
            cb: rand_coeffs(self.cb.len()),
            cc: rand_coeffs(self.cc.len()),
            eh: random_scalar(),
            ed: random_scalar(),
            eg: random_scalar(),
            em: random_scalar(),
            z: random_scalar(),
        }
    }

    fn respond(&self, rho: &Self, c: &Scalar) -> Self {
        let vec_resp = |r: &[Scalar], w: &[Scalar]| {
            r.iter()
                .zip(w.iter())
                .map(|(r, w)| r + c * w)
                .collect::<Vec<_>>()
        };
        let coeff_resp = |r: &[[Scalar; 4]], w: &[[Scalar; 4]]| {
            r.iter()
                .zip(w.iter())
                .map(|(r, w)| std::array::from_fn(|k| r[k] + c * w[k]))
                .collect::<Vec<[Scalar; 4]>>()
        };
        Self {
            d: vec_resp(&rho.d, &self.d),
            m: vec_resp(&rho.m, &self.m),
            h: vec_resp(&rho.h, &self.h),
            g: vec_resp(&rho.g, &self.g),
            v: rho.v + c * self.v,
            cb: coeff_resp(&rho.cb, &self.cb),
            cc: coeff_resp(&rho.cc, &self.cc),
            eh: rho.eh + c * self.eh,
            ed: rho.ed + c * self.ed,
            eg: rho.eg + c * self.eg,
            em: rho.em + c * self.em,
            z: rho.z + c * self.z,
        }
    }
}

// ---------------------------------------------------------------------------
// Digit layout
// ---------------------------------------------------------------------------

struct DigitLayout {
    /// (entry index, digit shift) for each digit slot; pads excluded.
    positions: Vec<(usize, usize)>,
    padded_len: usize,
}

fn digit_layout(entries: &[RangeEntry]) -> Result<DigitLayout, NativeError> {
    let mut positions = Vec::new();
    for (index, entry) in entries.iter().enumerate() {
        if entry.n % TABLE_BITS != 0 || entry.n == 0 || entry.n > 64 {
            return Err(NativeError::InvalidDimensions(
                "range entry width must be a multiple of 8".into(),
            ));
        }
        for digit in 0..(entry.n / TABLE_BITS) {
            positions.push((index, digit));
        }
    }
    if positions.is_empty() {
        return Err(NativeError::InvalidDimensions(
            "no range entries to prove".into(),
        ));
    }
    let padded_len = positions.len().next_power_of_two().max(2);
    Ok(DigitLayout {
        positions,
        padded_len,
    })
}

fn recomposition_weights(layout: &DigitLayout, rho: &[Scalar]) -> Vec<Scalar> {
    let mut weights = vec![Scalar::zero(); layout.padded_len];
    for (slot, (entry_index, digit)) in layout.positions.iter().enumerate() {
        weights[slot] = rho[*entry_index] * Scalar::from(1u64 << (TABLE_BITS * digit));
    }
    weights
}

// ---------------------------------------------------------------------------
// Prove / verify
// ---------------------------------------------------------------------------

pub(crate) fn prove(
    transcript: &mut Transcript,
    entries: &[RangeEntry],
) -> Result<LogUpProof, NativeError> {
    let timing = std::env::var("ZKLORA_V4_TIMING").is_ok();
    let mut mark = std::time::Instant::now();
    let mut lap = |label: &str| {
        if timing {
            eprintln!("    logup {label}: {:?}", mark.elapsed());
        }
        mark = std::time::Instant::now();
    };
    let gens = pc_gens();
    let layout = digit_layout(entries)?;
    let n_pad = layout.padded_len;

    // Digits, zero-padded to a power of two. Pad coordinates are public
    // zeros: they carry zero recomposition weight, the lookup claim is
    // gated by the public indicator vector s (1 on real slots, 0 on pads),
    // and announcement randomizers vanish there, so pads cost nothing in
    // any MSM and a prover stuffing junk into pad coordinates constrains
    // nothing (h is forced to 0 at pads and d pads are never used).
    let real_len = layout.positions.len();
    let mut digits = vec![Scalar::zero(); n_pad];
    let mut digit_ints = vec![0u64; n_pad];
    for (slot, (entry_index, digit)) in layout.positions.iter().enumerate() {
        let value = (entries[*entry_index].value >> (TABLE_BITS * digit)) & 0xff;
        digits[slot] = Scalar::from(value);
        digit_ints[slot] = value;
    }
    let mut multiplicities = vec![0u64; TABLE_SIZE];
    for value in &digit_ints[..real_len] {
        multiplicities[*value as usize] += 1;
    }
    let m_scalars: Vec<Scalar> = multiplicities.iter().map(|m| Scalar::from(*m)).collect();

    transcript.append_message(b"logup", b"v1");
    transcript.append_u64(b"logup-n", n_pad as u64);
    // Bind the entries this engine is proving directly into its own
    // transcript (defense in depth: the caller already absorbed every class
    // commitment, but the recomposition challenge rho must be sound even if
    // this engine is ever reused on a transcript that did not).
    for entry in entries {
        transcript.append_u64(b"logup-entry-bits", entry.n as u64);
        transcript.append_message(b"logup-entry-commitment", &entry.commitment);
    }
    let b_d = random_scalar();
    let b_m = random_scalar();
    let (c_d, c_m) = rayon::join(
        || vector_commit(&digits, &b_d),
        || vector_commit(&m_scalars, &b_m),
    );
    transcript.append_message(b"logup-c-d", &compress(&c_d));
    transcript.append_message(b"logup-c-m", &compress(&c_m));
    lap("digits+commit-dm");
    let alpha = transcript_scalar(transcript, b"logup-alpha");
    let rho = transcript_scalars(transcript, b"logup-rho", entries.len());
    lap("challenges-rho");

    // Helper vectors h = s/(alpha - d), g = m/(alpha - t).
    let k_vec: Vec<Scalar> = digits.iter().map(|d| alpha - d).collect();
    let mut h = batch_invert(&k_vec[..real_len])?;
    h.resize(n_pad, Scalar::zero());
    let table_k: Vec<Scalar> = (0..TABLE_SIZE)
        .map(|t| alpha - Scalar::from(t as u64))
        .collect();
    let table_k_inv = batch_invert(&table_k)?;
    let g: Vec<Scalar> = m_scalars
        .iter()
        .zip(table_k_inv.iter())
        .map(|(m, inv)| m * inv)
        .collect();
    let b_h = random_scalar();
    let b_g = random_scalar();
    let (c_h, c_g) = rayon::join(|| vector_commit(&h, &b_h), || vector_commit(&g, &b_g));
    transcript.append_message(b"logup-c-h", &compress(&c_h));
    transcript.append_message(b"logup-c-g", &compress(&c_g));

    lap("h-g-commit");
    let m_rounds = n_pad.trailing_zeros() as usize;
    let tau_b = transcript_scalars(transcript, b"logup-tau-b", m_rounds);
    let tau_c = transcript_scalars(transcript, b"logup-tau-c", TABLE_BITS);

    // Sumcheck B over the digit domain: e*(h*k - s) sums to zero, with s
    // the public real-slot indicator.
    let mut indicator = vec![Scalar::one(); real_len];
    indicator.resize(n_pad, Scalar::zero());
    let run_b = sumcheck_prove(
        transcript,
        b"logup-sc-b",
        eq_table(&tau_b),
        h.clone(),
        k_vec,
        indicator,
    );
    // Sumcheck C over the table: e*(g*k' - m) sums to zero.
    let run_c = sumcheck_prove(
        transcript,
        b"logup-sc-c",
        eq_table(&tau_c),
        g.clone(),
        table_k,
        m_scalars.clone(),
    );

    lap("sumchecks");
    // Final evaluations and the product witness z = E_h * E_d.
    let eq_r_b = eq_table(&run_b.rs);
    let eq_r_c = eq_table(&run_c.rs);
    let publics_es: Scalar = eq_r_b[..real_len].iter().sum();
    let eh = inner(&eq_r_b, &h);
    let ed = inner(&eq_r_b, &digits);
    let eg = inner(&eq_r_c, &g);
    let em = inner(&eq_r_c, &m_scalars);
    let z = eh * ed;
    let (b_eh, b_ed, b_eg, b_em, b_z) = (
        random_scalar(),
        random_scalar(),
        random_scalar(),
        random_scalar(),
        random_scalar(),
    );
    let c_eh = scalar_commit(&eh, &b_eh);
    let c_ed = scalar_commit(&ed, &b_ed);
    let c_eg = scalar_commit(&eg, &b_eg);
    let c_em = scalar_commit(&em, &b_em);
    let c_z = scalar_commit(&z, &b_z);
    for point in [&c_eh, &c_ed, &c_eg, &c_em, &c_z] {
        transcript.append_message(b"logup-evals", &compress(point));
    }

    // Product proof: z = E_h * E_d, proving the same E_d opens C_z over
    // base (C_eh, B~): C_z = E_d * C_eh + t * B~ with t = b_z - E_d*b_eh.
    let t = b_z - ed * b_eh;
    let rho_y = random_scalar();
    let rho_by = random_scalar();
    let rho_t = random_scalar();
    let a2 = scalar_commit(&rho_y, &rho_by);
    let a3 = c_eh * rho_y + gens.B_blinding * rho_t;
    transcript.append_message(b"logup-prod-a2", &compress(&a2));
    transcript.append_message(b"logup-prod-a3", &compress(&a3));
    let e_chal = transcript_scalar(transcript, b"logup-prod-challenge");
    let product = ProductProof {
        a2: compress(&a2),
        a3: compress(&a3),
        s_y: rho_y + e_chal * ed,
        s_by: rho_by + e_chal * b_ed,
        s_t: rho_t + e_chal * t,
    };

    // Mega-Schnorr over the full linear system.
    let v_value: Scalar = rho
        .iter()
        .zip(entries.iter())
        .map(|(r, entry)| r * Scalar::from(entry.value))
        .sum();
    let v_blinding: Scalar = rho
        .iter()
        .zip(entries.iter())
        .map(|(r, entry)| r * entry.blinding)
        .sum();
    let witness = LinearWitness {
        d: digits,
        m: m_scalars,
        h,
        g,
        v: v_value,
        cb: run_b.coeff_values.clone(),
        cc: run_c.coeff_values.clone(),
        eh,
        ed,
        eg,
        em,
        z,
    };
    let publics = PublicData {
        weights: recomposition_weights(&layout, &rho),
        alpha,
        rs_b: run_b.rs.clone(),
        rs_c: run_c.rs.clone(),
        eq_r_b,
        eq_r_c,
        eqv_b: eq_point(&tau_b, &run_b.rs),
        eqv_c: eq_point(&tau_c, &run_c.rs),
        identv: run_c
            .rs
            .iter()
            .enumerate()
            .map(|(j, r)| Scalar::from(1u64 << (TABLE_BITS - 1 - j)) * r)
            .sum(),
        es: publics_es,
    };
    debug_assert_eq!(
        constraints_eval(&witness, &publics),
        constraint_constants(&publics)
    );

    let rho_w = witness.random_like(real_len);
    // Blindings of the witness commitments, in the same component order.
    let wb = LinearWitnessBlindings {
        b_d,
        b_m,
        b_h,
        b_g,
        b_v: v_blinding,
        cb: run_b.coeff_blindings.clone(),
        cc: run_c.coeff_blindings.clone(),
        b_eh,
        b_ed,
        b_eg,
        b_em,
        b_z,
    };
    let rb = LinearWitnessBlindings {
        b_d: random_scalar(),
        b_m: random_scalar(),
        b_h: random_scalar(),
        b_g: random_scalar(),
        b_v: random_scalar(),
        cb: (0..run_b.coeff_blindings.len())
            .map(|_| std::array::from_fn(|_| random_scalar()))
            .collect(),
        cc: (0..run_c.coeff_blindings.len())
            .map(|_| std::array::from_fn(|_| random_scalar()))
            .collect(),
        b_eh: random_scalar(),
        b_ed: random_scalar(),
        b_eg: random_scalar(),
        b_em: random_scalar(),
        b_z: random_scalar(),
    };

    lap("evals+product+wb");
    let ((r_d, r_m), (r_h, r_g)) = rayon::join(
        || {
            rayon::join(
                || vector_commit(&rho_w.d, &rb.b_d),
                || vector_commit(&rho_w.m, &rb.b_m),
            )
        },
        || {
            rayon::join(
                || vector_commit(&rho_w.h, &rb.b_h),
                || vector_commit(&rho_w.g, &rb.b_g),
            )
        },
    );
    let r_v = scalar_commit(&rho_w.v, &rb.b_v);
    let commit_coeffs = |values: &[[Scalar; 4]], blindings: &[[Scalar; 4]]| {
        values
            .iter()
            .zip(blindings.iter())
            .map(|(v, b)| std::array::from_fn(|k| compress(&scalar_commit(&v[k], &b[k]))))
            .collect::<Vec<[[u8; 32]; 4]>>()
    };
    let r_coeffs_b = commit_coeffs(&rho_w.cb, &rb.cb);
    let r_coeffs_c = commit_coeffs(&rho_w.cc, &rb.cc);
    let r_eh = scalar_commit(&rho_w.eh, &rb.b_eh);
    let r_ed = scalar_commit(&rho_w.ed, &rb.b_ed);
    let r_eg = scalar_commit(&rho_w.eg, &rb.b_eg);
    let r_em = scalar_commit(&rho_w.em, &rb.b_em);
    let r_z = scalar_commit(&rho_w.z, &rb.b_z);
    let t_constraints = constraints_eval(&rho_w, &publics);

    lap("schnorr-announce");
    transcript.append_message(b"logup-schnorr-r-d", &compress(&r_d));
    transcript.append_message(b"logup-schnorr-r-m", &compress(&r_m));
    transcript.append_message(b"logup-schnorr-r-h", &compress(&r_h));
    transcript.append_message(b"logup-schnorr-r-g", &compress(&r_g));
    transcript.append_message(b"logup-schnorr-r-v", &compress(&r_v));
    for round in r_coeffs_b.iter().chain(r_coeffs_c.iter()) {
        absorb_points(transcript, b"logup-schnorr-coeffs", round);
    }
    for point in [&r_eh, &r_ed, &r_eg, &r_em, &r_z] {
        transcript.append_message(b"logup-schnorr-evals", &compress(point));
    }
    for value in &t_constraints {
        transcript.append_message(b"logup-schnorr-t", value.as_bytes());
    }
    let c = transcript_scalar(transcript, b"logup-schnorr-challenge");

    let response = witness.respond(&rho_w, &c);
    let coeff_open = |resp: &[[Scalar; 4]], rho_b: &[[Scalar; 4]], w_b: &[[Scalar; 4]]| {
        resp.iter()
            .zip(rho_b.iter().zip(w_b.iter()))
            .map(|(resp, (rho_b, w_b))| {
                std::array::from_fn(|k| ScalarOpen {
                    sigma: resp[k],
                    sigma_b: rho_b[k] + c * w_b[k],
                })
            })
            .collect::<Vec<[ScalarOpen; 4]>>()
    };

    let schnorr = MegaSchnorr {
        r_d: compress(&r_d),
        r_m: compress(&r_m),
        r_h: compress(&r_h),
        r_g: compress(&r_g),
        r_v: compress(&r_v),
        r_coeffs_b,
        r_coeffs_c,
        r_eh: compress(&r_eh),
        r_ed: compress(&r_ed),
        r_eg: compress(&r_eg),
        r_em: compress(&r_em),
        r_z: compress(&r_z),
        t_constraints,
        sigma_d: response.d.clone(),
        sigma_bd: rb.b_d + c * wb.b_d,
        sigma_m: response.m.clone(),
        sigma_bm: rb.b_m + c * wb.b_m,
        sigma_h: response.h.clone(),
        sigma_bh: rb.b_h + c * wb.b_h,
        sigma_g: response.g.clone(),
        sigma_bg: rb.b_g + c * wb.b_g,
        sigma_v: ScalarOpen {
            sigma: response.v,
            sigma_b: rb.b_v + c * wb.b_v,
        },
        sigma_coeffs_b: coeff_open(&response.cb, &rb.cb, &wb.cb),
        sigma_coeffs_c: coeff_open(&response.cc, &rb.cc, &wb.cc),
        sigma_eh: ScalarOpen {
            sigma: response.eh,
            sigma_b: rb.b_eh + c * wb.b_eh,
        },
        sigma_ed: ScalarOpen {
            sigma: response.ed,
            sigma_b: rb.b_ed + c * wb.b_ed,
        },
        sigma_eg: ScalarOpen {
            sigma: response.eg,
            sigma_b: rb.b_eg + c * wb.b_eg,
        },
        sigma_em: ScalarOpen {
            sigma: response.em,
            sigma_b: rb.b_em + c * wb.b_em,
        },
        sigma_z: ScalarOpen {
            sigma: response.z,
            sigma_b: rb.b_z + c * wb.b_z,
        },
    };

    lap("schnorr-respond");
    Ok(LogUpProof {
        c_d: compress(&c_d),
        c_m: compress(&c_m),
        c_h: compress(&c_h),
        c_g: compress(&c_g),
        rounds_b: run_b.coeff_commitments,
        rounds_c: run_c.coeff_commitments,
        c_eh: compress(&c_eh),
        c_ed: compress(&c_ed),
        c_eg: compress(&c_eg),
        c_em: compress(&c_em),
        c_z: compress(&c_z),
        product,
        schnorr,
    })
}

struct LinearWitnessBlindings {
    b_d: Scalar,
    b_m: Scalar,
    b_h: Scalar,
    b_g: Scalar,
    b_v: Scalar,
    cb: Vec<[Scalar; 4]>,
    cc: Vec<[Scalar; 4]>,
    b_eh: Scalar,
    b_ed: Scalar,
    b_eg: Scalar,
    b_em: Scalar,
    b_z: Scalar,
}

pub(crate) fn verify(
    transcript: &mut Transcript,
    entries: &[RangeEntry],
    proof: &LogUpProof,
) -> Result<(), NativeError> {
    let gens = pc_gens();
    let layout = digit_layout(entries)?;
    let n_pad = layout.padded_len;
    let m_rounds = n_pad.trailing_zeros() as usize;
    let fail = |what: &str| NativeError::Halo2(format!("logup verification failed: {what}"));

    if proof.rounds_b.len() != m_rounds
        || proof.rounds_c.len() != TABLE_BITS
        || proof.schnorr.sigma_d.len() != n_pad
        || proof.schnorr.sigma_h.len() != n_pad
        || proof.schnorr.sigma_m.len() != TABLE_SIZE
        || proof.schnorr.sigma_g.len() != TABLE_SIZE
        || proof.schnorr.sigma_coeffs_b.len() != m_rounds
        || proof.schnorr.sigma_coeffs_c.len() != TABLE_BITS
        || proof.schnorr.r_coeffs_b.len() != m_rounds
        || proof.schnorr.r_coeffs_c.len() != TABLE_BITS
    {
        return Err(fail("shape"));
    }

    transcript.append_message(b"logup", b"v1");
    transcript.append_u64(b"logup-n", n_pad as u64);
    for entry in entries {
        transcript.append_u64(b"logup-entry-bits", entry.n as u64);
        transcript.append_message(b"logup-entry-commitment", &entry.commitment);
    }
    transcript.append_message(b"logup-c-d", &proof.c_d);
    transcript.append_message(b"logup-c-m", &proof.c_m);
    let alpha = transcript_scalar(transcript, b"logup-alpha");
    let rho = transcript_scalars(transcript, b"logup-rho", entries.len());
    transcript.append_message(b"logup-c-h", &proof.c_h);
    transcript.append_message(b"logup-c-g", &proof.c_g);
    let tau_b = transcript_scalars(transcript, b"logup-tau-b", m_rounds);
    let tau_c = transcript_scalars(transcript, b"logup-tau-c", TABLE_BITS);
    let rs_b = sumcheck_replay(transcript, b"logup-sc-b", &proof.rounds_b);
    let rs_c = sumcheck_replay(transcript, b"logup-sc-c", &proof.rounds_c);
    for point in [proof.c_eh, proof.c_ed, proof.c_eg, proof.c_em, proof.c_z] {
        transcript.append_message(b"logup-evals", &point);
    }

    // Product proof.
    transcript.append_message(b"logup-prod-a2", &proof.product.a2);
    transcript.append_message(b"logup-prod-a3", &proof.product.a3);
    let e_chal = transcript_scalar(transcript, b"logup-prod-challenge");
    let c_eh = decompress(&proof.c_eh)?;
    let c_ed = decompress(&proof.c_ed)?;
    let c_z = decompress(&proof.c_z)?;
    let lhs = scalar_commit(&proof.product.s_y, &proof.product.s_by);
    if lhs != decompress(&proof.product.a2)? + c_ed * e_chal {
        return Err(fail("product proof (opening)"));
    }
    let lhs = c_eh * proof.product.s_y + gens.B_blinding * proof.product.s_t;
    if lhs != decompress(&proof.product.a3)? + c_z * e_chal {
        return Err(fail("product proof (relation)"));
    }

    // Mega-Schnorr replay.
    let eq_r_b = eq_table(&rs_b);
    let publics = PublicData {
        weights: recomposition_weights(&layout, &rho),
        alpha,
        eq_r_c: eq_table(&rs_c),
        eqv_b: eq_point(&tau_b, &rs_b),
        eqv_c: eq_point(&tau_c, &rs_c),
        identv: rs_c
            .iter()
            .enumerate()
            .map(|(j, r)| Scalar::from(1u64 << (TABLE_BITS - 1 - j)) * r)
            .sum(),
        es: eq_r_b[..layout.positions.len()].iter().sum(),
        eq_r_b,
        rs_b,
        rs_c,
    };
    let constants = constraint_constants(&publics);
    if proof.schnorr.t_constraints.len() != constants.len() {
        return Err(fail("constraint count"));
    }

    let schnorr = &proof.schnorr;
    transcript.append_message(b"logup-schnorr-r-d", &schnorr.r_d);
    transcript.append_message(b"logup-schnorr-r-m", &schnorr.r_m);
    transcript.append_message(b"logup-schnorr-r-h", &schnorr.r_h);
    transcript.append_message(b"logup-schnorr-r-g", &schnorr.r_g);
    transcript.append_message(b"logup-schnorr-r-v", &schnorr.r_v);
    for round in schnorr.r_coeffs_b.iter().chain(schnorr.r_coeffs_c.iter()) {
        absorb_points(transcript, b"logup-schnorr-coeffs", round);
    }
    for point in [
        &schnorr.r_eh,
        &schnorr.r_ed,
        &schnorr.r_eg,
        &schnorr.r_em,
        &schnorr.r_z,
    ] {
        transcript.append_message(b"logup-schnorr-evals", point);
    }
    for value in &schnorr.t_constraints {
        transcript.append_message(b"logup-schnorr-t", value.as_bytes());
    }
    let c = transcript_scalar(transcript, b"logup-schnorr-challenge");

    // Commitment-equation checks.
    let check_vector =
        |sigma: &[Scalar],
         sigma_b: &Scalar,
         announcement: &[u8; 32],
         commitment: &[u8; 32]|
         -> Result<bool, NativeError> {
            Ok(vector_commit(sigma, sigma_b)
                == decompress(announcement)? + decompress(commitment)? * c)
        };
    // Pad coordinates of d and h are public zeros by construction (zero
    // recomposition weight, indicator-gated lookup). Soundness does not
    // depend on it, but enforcing it here turns the convention into a
    // checked invariant.
    let real_len = layout.positions.len();
    if schnorr.sigma_d[real_len..]
        .iter()
        .chain(schnorr.sigma_h[real_len..].iter())
        .any(|s| *s != Scalar::zero())
    {
        return Err(fail("nonzero pad response"));
    }
    type VectorCheck<'a> = (
        &'a [Scalar],
        &'a Scalar,
        &'a [u8; 32],
        &'a [u8; 32],
        &'a str,
    );
    let vector_checks: [VectorCheck; 4] = [
        (
            &schnorr.sigma_d,
            &schnorr.sigma_bd,
            &schnorr.r_d,
            &proof.c_d,
            "d opening",
        ),
        (
            &schnorr.sigma_m,
            &schnorr.sigma_bm,
            &schnorr.r_m,
            &proof.c_m,
            "m opening",
        ),
        (
            &schnorr.sigma_h,
            &schnorr.sigma_bh,
            &schnorr.r_h,
            &proof.c_h,
            "h opening",
        ),
        (
            &schnorr.sigma_g,
            &schnorr.sigma_bg,
            &schnorr.r_g,
            &proof.c_g,
            "g opening",
        ),
    ];
    vector_checks.into_par_iter().try_for_each(
        |(sigma, sigma_b, announcement, commitment, what)| {
            if check_vector(sigma, sigma_b, announcement, commitment)? {
                Ok(())
            } else {
                Err(fail(what))
            }
        },
    )?;
    // V opens P_v = sum rho_e C_e.
    let p_v = RistrettoPoint::vartime_multiscalar_mul(
        rho.iter(),
        entries
            .iter()
            .map(|entry| decompress(&entry.commitment))
            .collect::<Result<Vec<_>, _>>()?
            .iter(),
    );
    if scalar_commit(&schnorr.sigma_v.sigma, &schnorr.sigma_v.sigma_b)
        != decompress(&schnorr.r_v)? + p_v * c
    {
        return Err(fail("recomposition value opening"));
    }
    let check_scalar = |open: &ScalarOpen,
                        announcement: &[u8; 32],
                        commitment: &[u8; 32]|
     -> Result<bool, NativeError> {
        Ok(scalar_commit(&open.sigma, &open.sigma_b)
            == decompress(announcement)? + decompress(commitment)? * c)
    };
    for ((opens, announcements), commitments) in schnorr
        .sigma_coeffs_b
        .iter()
        .zip(schnorr.r_coeffs_b.iter())
        .zip(proof.rounds_b.iter())
        .chain(
            schnorr
                .sigma_coeffs_c
                .iter()
                .zip(schnorr.r_coeffs_c.iter())
                .zip(proof.rounds_c.iter()),
        )
    {
        for k in 0..4 {
            if !check_scalar(&opens[k], &announcements[k], &commitments[k])? {
                return Err(fail("round coefficient opening"));
            }
        }
    }
    for (open, (announcement, commitment)) in [
        (&schnorr.sigma_eh, (&schnorr.r_eh, &proof.c_eh)),
        (&schnorr.sigma_ed, (&schnorr.r_ed, &proof.c_ed)),
        (&schnorr.sigma_eg, (&schnorr.r_eg, &proof.c_eg)),
        (&schnorr.sigma_em, (&schnorr.r_em, &proof.c_em)),
        (&schnorr.sigma_z, (&schnorr.r_z, &proof.c_z)),
    ] {
        if !check_scalar(open, announcement, commitment)? {
            return Err(fail("evaluation opening"));
        }
    }

    // Linear constraint checks: L(sigma) == T + c*K.
    let response = LinearWitness {
        d: schnorr.sigma_d.clone(),
        m: schnorr.sigma_m.clone(),
        h: schnorr.sigma_h.clone(),
        g: schnorr.sigma_g.clone(),
        v: schnorr.sigma_v.sigma,
        cb: schnorr
            .sigma_coeffs_b
            .iter()
            .map(|round| std::array::from_fn(|k| round[k].sigma))
            .collect(),
        cc: schnorr
            .sigma_coeffs_c
            .iter()
            .map(|round| std::array::from_fn(|k| round[k].sigma))
            .collect(),
        eh: schnorr.sigma_eh.sigma,
        ed: schnorr.sigma_ed.sigma,
        eg: schnorr.sigma_eg.sigma,
        em: schnorr.sigma_em.sigma,
        z: schnorr.sigma_z.sigma,
    };
    let evaluated = constraints_eval(&response, &publics);
    for ((left, t), k) in evaluated
        .iter()
        .zip(schnorr.t_constraints.iter())
        .zip(constants.iter())
    {
        if *left != t + c * k {
            return Err(fail("linear constraint"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(value: u64, n: usize) -> RangeEntry {
        let blinding = random_scalar();
        RangeEntry {
            n,
            value,
            blinding,
            commitment: compress(&scalar_commit(&Scalar::from(value), &blinding)),
        }
    }

    fn entries_mixed() -> Vec<RangeEntry> {
        vec![
            entry(0, 8),
            entry(255, 8),
            entry(65535, 16),
            entry(123456, 32),
            entry(u64::MAX, 64),
            entry(1, 64),
            entry((1 << 20) - 1, 32),
        ]
    }

    #[test]
    fn roundtrip_mixed_widths() {
        let entries = entries_mixed();
        let mut prover_transcript = Transcript::new(b"logup-test");
        let proof = prove(&mut prover_transcript, &entries).expect("prove");
        let mut verifier_transcript = Transcript::new(b"logup-test");
        verify(&mut verifier_transcript, &entries, &proof).expect("verify");
    }

    #[test]
    fn single_entry_roundtrip() {
        let entries = vec![entry(7, 8)];
        let mut prover_transcript = Transcript::new(b"logup-test");
        let proof = prove(&mut prover_transcript, &entries).expect("prove");
        let mut verifier_transcript = Transcript::new(b"logup-test");
        verify(&mut verifier_transcript, &entries, &proof).expect("verify");
    }

    #[test]
    fn commitment_to_out_of_range_value_rejected() {
        // A valid proof must not verify against a commitment whose opening
        // exceeds the claimed width: the recomposition Schnorr ties the
        // committed value to its in-range digit decomposition.
        let entries = entries_mixed();
        let mut prover_transcript = Transcript::new(b"logup-test");
        let proof = prove(&mut prover_transcript, &entries).expect("prove");
        let mut tampered = entries.clone();
        tampered[0] = RangeEntry {
            n: 8,
            value: 0,
            blinding: Scalar::zero(),
            commitment: compress(&scalar_commit(&Scalar::from(300u64), &entries[0].blinding)),
        };
        let mut verifier_transcript = Transcript::new(b"logup-test");
        assert!(verify(&mut verifier_transcript, &tampered, &proof).is_err());
    }

    #[test]
    fn tampered_round_commitment_rejected() {
        let entries = entries_mixed();
        let mut prover_transcript = Transcript::new(b"logup-test");
        let mut proof = prove(&mut prover_transcript, &entries).expect("prove");
        proof.rounds_b[0][1][7] ^= 0x20;
        let mut verifier_transcript = Transcript::new(b"logup-test");
        assert!(verify(&mut verifier_transcript, &entries, &proof).is_err());
    }

    #[test]
    fn transcript_binding_rejects_context_swap() {
        let entries = entries_mixed();
        let mut prover_transcript = Transcript::new(b"logup-test");
        let proof = prove(&mut prover_transcript, &entries).expect("prove");
        let mut other_context = Transcript::new(b"logup-other");
        assert!(verify(&mut other_context, &entries, &proof).is_err());
    }
}
