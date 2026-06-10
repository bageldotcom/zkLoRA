use ff::PrimeField;
#[cfg(any(test, feature = "python"))]
use halo2_gadgets::poseidon::primitives::Hash as NativePoseidonHash;
use halo2_gadgets::poseidon::{
    primitives::{ConstantLength, P128Pow5T3},
    Hash as PoseidonHash, Pow5Chip, Pow5Config,
};
use halo2_proofs::{
    circuit::{AssignedCell, Layouter, SimpleFloorPlanner, Value},
    pasta::{vesta, EqAffine, Fp},
    plonk::{
        create_proof, keygen_pk, keygen_vk, verify_proof, Advice, Circuit, Column,
        ConstraintSystem, Error, Instance, ProvingKey, Selector, SingleVerifier, VerifyingKey,
    },
    poly::commitment::Params,
    poly::Rotation,
    transcript::{Blake2bRead, Blake2bWrite, Challenge255},
};
use num_bigint::{BigInt, BigUint, Sign};
use num_integer::Integer;
use num_traits::{One, Signed, Zero};
use rand_core::OsRng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::convert::TryInto;
use std::sync::{Arc, Mutex, OnceLock};

const ADAPTER_COMMITMENT_DOMAIN: u64 = 0x5a4b4c4f5241; // "ZKLORA"
const ADAPTER_COMMITMENT_VERSION: u64 = 1;
// Must match proof_contract.SCHEMA_VERSION; it is hashed into adapter commitments.
const ARTIFACT_SCHEMA_VERSION: u64 = 2;
const FIELD_SAFE_BITS: usize = 250;
const POSEIDON_PAIR_ROWS: usize = 96;
// Caps for the legacy backend: artifacts beyond these shapes are rejected before
// any keygen work so a hostile statement cannot stall the verifier.
const MAX_LEGACY_K: u32 = 24;
const MAX_LEGACY_DIM: usize = 16_384;
const MAX_LEGACY_RANK: usize = 1_024;

#[derive(Debug, thiserror::Error)]
pub enum NativeError {
    #[error("invalid dimensions: {0}")]
    InvalidDimensions(String),
    #[error("halo2 error: {0}")]
    Halo2(String),
    #[error("json error: {0}")]
    Json(String),
}

#[cfg(feature = "python")]
impl From<NativeError> for pyo3::PyErr {
    fn from(value: NativeError) -> Self {
        pyo3::exceptions::PyValueError::new_err(value.to_string())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FixedPointConfig {
    pub scale_bits: u32,
    pub value_bits: u32,
    pub intermediate_bits: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NativeStatement {
    pub x: Vec<i64>,
    pub delta: Vec<i64>,
    pub fixed_point: FixedPointConfig,
    #[serde(default = "default_rank")]
    pub rank: usize,
    #[serde(default = "default_scaling_num")]
    pub scaling_num: i64,
    #[serde(default = "default_scaling_den")]
    pub scaling_den: i64,
    #[serde(default)]
    pub adapter_commitment: String,
    #[serde(default)]
    pub statement_digest: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NativeWitness {
    pub a: Vec<Vec<i64>>,
    pub b: Vec<Vec<i64>>,
}

#[cfg(any(test, feature = "python"))]
#[derive(Clone, Debug, Serialize, Deserialize)]
struct AdapterCommitmentInput {
    pub schema_version: u64,
    pub in_dim: usize,
    pub rank: usize,
    pub out_dim: usize,
    pub fixed_point: FixedPointConfig,
    pub scaling_num: i64,
    pub scaling_den: i64,
    pub a: Vec<Vec<i64>>,
    pub b: Vec<Vec<i64>>,
}

#[derive(Clone, Debug)]
struct LoraCircuit {
    a: Vec<Vec<i64>>,
    b: Vec<Vec<i64>>,
    x: Vec<i64>,
    delta: Vec<i64>,
    fixed_point: FixedPointConfig,
    scaling_num: i64,
    scaling_den: i64,
    adapter_commitment: String,
    statement_digest: String,
}

#[derive(Clone, Debug)]
struct LoraConfig {
    advice: [Column<Advice>; 4],
    instance: Column<Instance>,
    mul: Selector,
    add: Selector,
    div_round: Selector,
    boolean: Selector,
    poseidon_config: Pow5Config<Fp, 3, 2>,
}

impl Circuit<Fp> for LoraCircuit {
    type Config = LoraConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self {
            a: vec![vec![0; self.in_dim()]; self.rank()],
            b: vec![vec![0; self.rank()]; self.out_dim()],
            x: vec![0; self.in_dim()],
            delta: vec![0; self.out_dim()],
            fixed_point: self.fixed_point.clone(),
            scaling_num: self.scaling_num,
            scaling_den: self.scaling_den,
            adapter_commitment: self.adapter_commitment.clone(),
            statement_digest: self.statement_digest.clone(),
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        let advice = [
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        let instance = meta.instance_column();
        for column in advice {
            meta.enable_equality(column);
        }
        meta.enable_equality(instance);

        let mul = meta.selector();
        meta.create_gate("mul", |meta| {
            let s = meta.query_selector(mul);
            let lhs = meta.query_advice(advice[0], Rotation::cur());
            let rhs = meta.query_advice(advice[1], Rotation::cur());
            let out = meta.query_advice(advice[2], Rotation::cur());
            vec![s * (lhs * rhs - out)]
        });

        let add = meta.selector();
        meta.create_gate("add", |meta| {
            let s = meta.query_selector(add);
            let lhs = meta.query_advice(advice[0], Rotation::cur());
            let rhs = meta.query_advice(advice[1], Rotation::cur());
            let out = meta.query_advice(advice[2], Rotation::cur());
            vec![s * (lhs + rhs - out)]
        });

        let div_round = meta.selector();
        meta.create_gate("bounded deterministic division witness", |meta| {
            let s = meta.query_selector(div_round);
            let raw = meta.query_advice(advice[0], Rotation::cur());
            let quotient = meta.query_advice(advice[1], Rotation::cur());
            let denominator = meta.query_advice(advice[2], Rotation::cur());
            let remainder = meta.query_advice(advice[3], Rotation::cur());
            vec![s * (raw - quotient * denominator - remainder)]
        });

        let boolean = meta.selector();
        meta.create_gate("boolean bit", |meta| {
            let s = meta.query_selector(boolean);
            let bit = meta.query_advice(advice[0], Rotation::cur());
            vec![s * bit.clone() * (bit - halo2_proofs::plonk::Expression::Constant(Fp::from(1)))]
        });

        let poseidon_state = (0..3).map(|_| meta.advice_column()).collect::<Vec<_>>();
        for column in &poseidon_state {
            meta.enable_equality(*column);
        }
        let poseidon_partial_sbox = meta.advice_column();
        let rc_a = (0..3).map(|_| meta.fixed_column()).collect::<Vec<_>>();
        let rc_b = (0..3).map(|_| meta.fixed_column()).collect::<Vec<_>>();
        meta.enable_constant(rc_b[0]);
        let poseidon_config = Pow5Chip::configure::<P128Pow5T3>(
            meta,
            poseidon_state.try_into().expect("poseidon width"),
            poseidon_partial_sbox,
            rc_a.try_into().expect("poseidon rc_a"),
            rc_b.try_into().expect("poseidon rc_b"),
        );

        LoraConfig {
            advice,
            instance,
            mul,
            add,
            div_round,
            boolean,
            poseidon_config,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        self.validate().map_err(|_| Error::Synthesis)?;

        let (x_cells, delta_cells, binding_cells, adapter_words) = layouter.assign_region(
            || "zklora lora delta relation",
            |mut region| {
                let mut offset = 0usize;
                let scale = scale_bigint(&self.fixed_point);
                let value_bound = signed_bound(self.fixed_point.value_bits)?;
                let intermediate_bound = signed_bound(self.fixed_point.intermediate_bits)?;
                let raw_a_bound = &value_bound * &value_bound * BigInt::from(self.in_dim());
                let raw_b_bound = &value_bound * &intermediate_bound * BigInt::from(self.rank());
                let scaled_raw_bound = &intermediate_bound * BigInt::from(self.scaling_num).abs();
                let product_value_bound = &value_bound * &value_bound;
                let product_intermediate_bound = &value_bound * &intermediate_bound;

                let zero = assign_constant_cell(
                    &mut region,
                    config.advice[0],
                    offset,
                    Fp::from(0),
                    "zero",
                )?;
                offset += 1;
                let scale_cell = assign_constant_cell(
                    &mut region,
                    config.advice[0],
                    offset,
                    fp_from_bigint_checked(&scale).map_err(|_| Error::Synthesis)?,
                    "fixed point scale",
                )?;
                offset += 1;
                let scaling_num_cell = assign_constant_cell(
                    &mut region,
                    config.advice[0],
                    offset,
                    fp_from_i64(self.scaling_num),
                    "scaling numerator",
                )?;
                offset += 1;
                let scaling_den_cell = assign_constant_cell(
                    &mut region,
                    config.advice[0],
                    offset,
                    fp_from_i64(self.scaling_den),
                    "scaling denominator",
                )?;
                offset += 1;

                let mut adapter_words = Vec::new();
                push_adapter_constant(
                    &mut region,
                    &config,
                    &mut adapter_words,
                    &mut offset,
                    BigInt::from(ADAPTER_COMMITMENT_DOMAIN),
                    "adapter domain",
                )?;
                push_adapter_constant(
                    &mut region,
                    &config,
                    &mut adapter_words,
                    &mut offset,
                    BigInt::from(ADAPTER_COMMITMENT_VERSION),
                    "adapter commitment version",
                )?;
                push_adapter_constant(
                    &mut region,
                    &config,
                    &mut adapter_words,
                    &mut offset,
                    BigInt::from(ARTIFACT_SCHEMA_VERSION),
                    "adapter schema version",
                )?;
                for (label, value) in [
                    ("adapter in_dim", BigInt::from(self.in_dim())),
                    ("adapter rank", BigInt::from(self.rank())),
                    ("adapter out_dim", BigInt::from(self.out_dim())),
                    (
                        "adapter scale_bits",
                        BigInt::from(self.fixed_point.scale_bits),
                    ),
                    (
                        "adapter value_bits",
                        BigInt::from(self.fixed_point.value_bits),
                    ),
                    (
                        "adapter intermediate_bits",
                        BigInt::from(self.fixed_point.intermediate_bits),
                    ),
                    ("adapter scaling_num", BigInt::from(self.scaling_num)),
                    ("adapter scaling_den", BigInt::from(self.scaling_den)),
                ] {
                    push_adapter_constant(
                        &mut region,
                        &config,
                        &mut adapter_words,
                        &mut offset,
                        value,
                        label,
                    )?;
                }

                let mut x_cells = Vec::with_capacity(self.x.len());
                for (i, value) in self.x.iter().enumerate() {
                    let value_big = BigInt::from(*value);
                    let cell = region.assign_advice(
                        || format!("public x {i}"),
                        config.advice[0],
                        offset,
                        || Value::known(fp_from_i64(*value)),
                    )?;
                    offset += 1;
                    offset = range_check_signed_interval(
                        &mut region,
                        &config,
                        &cell,
                        &value_big,
                        &(-&value_bound),
                        &value_bound,
                        offset,
                    )?;
                    x_cells.push(cell);
                }

                let mut intermediate = Vec::with_capacity(self.rank());
                for rank_index in 0..self.rank() {
                    let mut acc = zero.clone();
                    let mut raw_value = BigInt::zero();
                    for input_index in 0..self.in_dim() {
                        let weight_value = self.a[rank_index][input_index];
                        let weight_big = BigInt::from(weight_value);
                        let weight = region.assign_advice(
                            || format!("A[{rank_index}][{input_index}]"),
                            config.advice[1],
                            offset,
                            || Value::known(fp_from_i64(weight_value)),
                        )?;
                        offset += 1;
                        offset = range_check_signed_interval(
                            &mut region,
                            &config,
                            &weight,
                            &weight_big,
                            &(-&value_bound),
                            &value_bound,
                            offset,
                        )?;
                        adapter_words.push(weight.clone());

                        let product = assign_mul(
                            &mut region,
                            &config,
                            &x_cells[input_index],
                            &weight,
                            offset,
                        )?;
                        offset += 1;
                        let product_value = &weight_big * BigInt::from(self.x[input_index]);
                        offset = range_check_signed_interval(
                            &mut region,
                            &config,
                            &product,
                            &product_value,
                            &(-&product_value_bound),
                            &product_value_bound,
                            offset,
                        )?;
                        let next_acc = assign_add(&mut region, &config, &acc, &product, offset)?;
                        offset += 1;
                        raw_value += product_value;
                        acc = next_acc;
                    }
                    offset = range_check_signed_interval(
                        &mut region,
                        &config,
                        &acc,
                        &raw_value,
                        &(-&raw_a_bound),
                        &raw_a_bound,
                        offset,
                    )?;
                    let q = div_round_to_canonical_interval(&raw_value, &scale)?;
                    let (intermediate_cell, next_offset) = assign_div_round(
                        &mut region,
                        &config,
                        &acc,
                        &raw_value,
                        &q,
                        &scale,
                        &scale_cell,
                        &raw_a_bound,
                        &intermediate_bound,
                        offset,
                    )?;
                    offset = next_offset;
                    intermediate.push((intermediate_cell, q));
                }

                let mut delta_cells = Vec::with_capacity(self.out_dim());
                for out_index in 0..self.out_dim() {
                    let mut acc = zero.clone();
                    let mut raw_value = BigInt::zero();
                    for rank_index in 0..self.rank() {
                        let weight_value = self.b[out_index][rank_index];
                        let weight_big = BigInt::from(weight_value);
                        let weight = region.assign_advice(
                            || format!("B[{out_index}][{rank_index}]"),
                            config.advice[1],
                            offset,
                            || Value::known(fp_from_i64(weight_value)),
                        )?;
                        offset += 1;
                        offset = range_check_signed_interval(
                            &mut region,
                            &config,
                            &weight,
                            &weight_big,
                            &(-&value_bound),
                            &value_bound,
                            offset,
                        )?;
                        adapter_words.push(weight.clone());

                        let product = assign_mul(
                            &mut region,
                            &config,
                            &intermediate[rank_index].0,
                            &weight,
                            offset,
                        )?;
                        offset += 1;
                        let product_value = &weight_big * &intermediate[rank_index].1;
                        offset = range_check_signed_interval(
                            &mut region,
                            &config,
                            &product,
                            &product_value,
                            &(-&product_intermediate_bound),
                            &product_intermediate_bound,
                            offset,
                        )?;
                        let next_acc = assign_add(&mut region, &config, &acc, &product, offset)?;
                        offset += 1;
                        raw_value += product_value;
                        acc = next_acc;
                    }
                    offset = range_check_signed_interval(
                        &mut region,
                        &config,
                        &acc,
                        &raw_value,
                        &(-&raw_b_bound),
                        &raw_b_bound,
                        offset,
                    )?;
                    let rescaled = div_round_to_canonical_interval(&raw_value, &scale)?;
                    let (rescaled_cell, next_offset) = assign_div_round(
                        &mut region,
                        &config,
                        &acc,
                        &raw_value,
                        &rescaled,
                        &scale,
                        &scale_cell,
                        &raw_b_bound,
                        &intermediate_bound,
                        offset,
                    )?;
                    offset = next_offset;

                    let scaled_raw = &rescaled * BigInt::from(self.scaling_num);
                    let scaled_raw_cell = assign_mul(
                        &mut region,
                        &config,
                        &rescaled_cell,
                        &scaling_num_cell,
                        offset,
                    )?;
                    offset += 1;
                    offset = range_check_signed_interval(
                        &mut region,
                        &config,
                        &scaled_raw_cell,
                        &scaled_raw,
                        &(-&scaled_raw_bound),
                        &scaled_raw_bound,
                        offset,
                    )?;

                    let scaling_den_big = BigInt::from(self.scaling_den);
                    let final_delta =
                        div_round_to_canonical_interval(&scaled_raw, &scaling_den_big)?;
                    let (final_cell, next_offset) = assign_div_round(
                        &mut region,
                        &config,
                        &scaled_raw_cell,
                        &scaled_raw,
                        &final_delta,
                        &scaling_den_big,
                        &scaling_den_cell,
                        &scaled_raw_bound,
                        &value_bound,
                        offset,
                    )?;
                    offset = next_offset;
                    delta_cells.push(final_cell);
                }

                let statement_digest_cells = assign_statement_digest_cells(
                    &mut region,
                    &config,
                    &self.statement_digest,
                    &mut offset,
                )?;

                Ok((
                    x_cells,
                    delta_cells,
                    vec![
                        scale_cell,
                        scaling_num_cell,
                        scaling_den_cell,
                        statement_digest_cells[0].clone(),
                        statement_digest_cells[1].clone(),
                    ],
                    adapter_words,
                ))
            },
        )?;

        let adapter_commitment_cell = hash_adapter_words(&config, &mut layouter, adapter_words)?;

        for (i, cell) in x_cells.iter().enumerate() {
            layouter.constrain_instance(cell.cell(), config.instance, i)?;
        }
        for (i, cell) in delta_cells.iter().enumerate() {
            layouter.constrain_instance(cell.cell(), config.instance, self.in_dim() + i)?;
        }
        let binding_offset = self.in_dim() + self.out_dim();
        for (i, cell) in binding_cells.iter().enumerate() {
            layouter.constrain_instance(cell.cell(), config.instance, binding_offset + i)?;
        }
        layouter.constrain_instance(
            adapter_commitment_cell.cell(),
            config.instance,
            binding_offset + binding_cells.len(),
        )?;
        Ok(())
    }
}

impl LoraCircuit {
    fn in_dim(&self) -> usize {
        self.x.len()
    }

    fn rank(&self) -> usize {
        self.a.len()
    }

    fn out_dim(&self) -> usize {
        self.delta.len()
    }

    fn validate(&self) -> Result<(), NativeError> {
        if self.x.is_empty() {
            return Err(NativeError::InvalidDimensions(
                "x cannot be empty".to_string(),
            ));
        }
        if self.scaling_den <= 0 {
            return Err(NativeError::InvalidDimensions(
                "scaling denominator must be positive".to_string(),
            ));
        }
        if self.fixed_point.scale_bits >= self.fixed_point.value_bits {
            return Err(NativeError::InvalidDimensions(
                "scale_bits must be less than value_bits".to_string(),
            ));
        }
        if self.fixed_point.value_bits == 0
            || self.fixed_point.value_bits > 63
            || self.fixed_point.intermediate_bits == 0
        {
            return Err(NativeError::InvalidDimensions(
                "invalid fixed-point bit widths".to_string(),
            ));
        }
        if self.a.is_empty() || self.b.is_empty() {
            return Err(NativeError::InvalidDimensions(
                "A and B cannot be empty".to_string(),
            ));
        }
        let in_dim = self.in_dim();
        if in_dim > MAX_LEGACY_DIM || self.out_dim() > MAX_LEGACY_DIM {
            return Err(NativeError::InvalidDimensions(format!(
                "legacy artifact exceeds verification caps: dims {}x{} beyond {}",
                in_dim,
                self.out_dim(),
                MAX_LEGACY_DIM
            )));
        }
        if self.rank() > MAX_LEGACY_RANK {
            return Err(NativeError::InvalidDimensions(format!(
                "legacy artifact exceeds verification caps: rank {} beyond {}",
                self.rank(),
                MAX_LEGACY_RANK
            )));
        }
        for row in &self.a {
            if row.len() != in_dim {
                return Err(NativeError::InvalidDimensions(
                    "A row width must match x length".to_string(),
                ));
            }
        }
        let rank = self.rank();
        for row in &self.b {
            if row.len() != rank {
                return Err(NativeError::InvalidDimensions(
                    "B row width must match A rank".to_string(),
                ));
            }
        }
        if self.b.len() != self.delta.len() {
            return Err(NativeError::InvalidDimensions(
                "B row count must match delta length".to_string(),
            ));
        }
        for (label, values) in [
            ("x", self.x.iter().copied().collect::<Vec<_>>()),
            ("delta", self.delta.iter().copied().collect::<Vec<_>>()),
            ("A", self.a.iter().flatten().copied().collect::<Vec<_>>()),
            ("B", self.b.iter().flatten().copied().collect::<Vec<_>>()),
        ] {
            for value in values {
                check_signed_bound(
                    &BigInt::from(value),
                    &signed_bound(self.fixed_point.value_bits)
                        .map_err(|_| NativeError::InvalidDimensions("bad value_bits".into()))?,
                    label,
                )?;
            }
        }
        validate_field_safety(self)?;
        parse_adapter_commitment(&self.adapter_commitment)?;
        statement_digest_limbs(&self.statement_digest)?;
        Ok(())
    }
}

fn assign_mul(
    region: &mut halo2_proofs::circuit::Region<'_, Fp>,
    config: &LoraConfig,
    lhs: &AssignedCell<Fp, Fp>,
    rhs: &AssignedCell<Fp, Fp>,
    offset: usize,
) -> Result<AssignedCell<Fp, Fp>, Error> {
    config.mul.enable(region, offset)?;
    lhs.copy_advice(|| "mul lhs", region, config.advice[0], offset)?;
    rhs.copy_advice(|| "mul rhs", region, config.advice[1], offset)?;
    region.assign_advice(
        || "mul out",
        config.advice[2],
        offset,
        || lhs.value().copied() * rhs.value().copied(),
    )
}

fn assign_add(
    region: &mut halo2_proofs::circuit::Region<'_, Fp>,
    config: &LoraConfig,
    lhs: &AssignedCell<Fp, Fp>,
    rhs: &AssignedCell<Fp, Fp>,
    offset: usize,
) -> Result<AssignedCell<Fp, Fp>, Error> {
    config.add.enable(region, offset)?;
    lhs.copy_advice(|| "add lhs", region, config.advice[0], offset)?;
    rhs.copy_advice(|| "add rhs", region, config.advice[1], offset)?;
    region.assign_advice(
        || "add out",
        config.advice[2],
        offset,
        || lhs.value().copied() + rhs.value().copied(),
    )
}

fn assign_constant_cell(
    region: &mut halo2_proofs::circuit::Region<'_, Fp>,
    column: Column<Advice>,
    offset: usize,
    value: Fp,
    label: &'static str,
) -> Result<AssignedCell<Fp, Fp>, Error> {
    region.assign_advice_from_constant(|| label, column, offset, value)
}

fn push_adapter_constant(
    region: &mut halo2_proofs::circuit::Region<'_, Fp>,
    config: &LoraConfig,
    adapter_words: &mut Vec<AssignedCell<Fp, Fp>>,
    offset: &mut usize,
    value: BigInt,
    label: &'static str,
) -> Result<(), Error> {
    let cell = assign_constant_cell(
        region,
        config.advice[0],
        *offset,
        fp_from_bigint_checked(&value).map_err(|_| Error::Synthesis)?,
        label,
    )?;
    *offset += 1;
    adapter_words.push(cell);
    Ok(())
}

fn assign_statement_digest_cells(
    region: &mut halo2_proofs::circuit::Region<'_, Fp>,
    config: &LoraConfig,
    statement_digest: &str,
    offset: &mut usize,
) -> Result<[AssignedCell<Fp, Fp>; 2], Error> {
    let limbs = statement_digest_limbs(statement_digest).map_err(|_| Error::Synthesis)?;
    let first = region.assign_advice(
        || "statement digest high limb",
        config.advice[0],
        *offset,
        || Value::known(fp_from_biguint_checked(&limbs[0]).expect("digest limb fits")),
    )?;
    *offset += 1;
    let second = region.assign_advice(
        || "statement digest low limb",
        config.advice[0],
        *offset,
        || Value::known(fp_from_biguint_checked(&limbs[1]).expect("digest limb fits")),
    )?;
    *offset += 1;
    Ok([first, second])
}

fn assign_div_round(
    region: &mut halo2_proofs::circuit::Region<'_, Fp>,
    config: &LoraConfig,
    raw: &AssignedCell<Fp, Fp>,
    raw_value: &BigInt,
    quotient: &BigInt,
    denominator: &BigInt,
    denominator_cell: &AssignedCell<Fp, Fp>,
    raw_bound: &BigInt,
    quotient_bound: &BigInt,
    mut offset: usize,
) -> Result<(AssignedCell<Fp, Fp>, usize), Error> {
    if denominator <= &BigInt::zero() {
        return Err(Error::Synthesis);
    }
    offset = range_check_signed_interval(
        region,
        config,
        raw,
        raw_value,
        &(-raw_bound),
        raw_bound,
        offset,
    )?;
    let remainder = raw_value - quotient * denominator;
    config.div_round.enable(region, offset)?;
    raw.copy_advice(|| "division raw", region, config.advice[0], offset)?;
    let quotient_cell = region.assign_advice(
        || "division quotient",
        config.advice[1],
        offset,
        || Value::known(fp_from_bigint_checked(quotient).expect("quotient fits")),
    )?;
    denominator_cell.copy_advice(|| "division denominator", region, config.advice[2], offset)?;
    let remainder_cell = region.assign_advice(
        || "division remainder",
        config.advice[3],
        offset,
        || Value::known(fp_from_bigint_checked(&remainder).expect("remainder fits")),
    )?;
    offset += 1;
    offset = range_check_signed_interval(
        region,
        config,
        &quotient_cell,
        quotient,
        &(-quotient_bound),
        quotient_bound,
        offset,
    )?;
    let (lower, upper) = canonical_remainder_interval(denominator);
    offset = range_check_signed_interval(
        region,
        config,
        &remainder_cell,
        &remainder,
        &lower,
        &upper,
        offset,
    )?;
    Ok((quotient_cell, offset))
}

fn range_check_signed_interval(
    region: &mut halo2_proofs::circuit::Region<'_, Fp>,
    config: &LoraConfig,
    value_cell: &AssignedCell<Fp, Fp>,
    value: &BigInt,
    lower: &BigInt,
    upper: &BigInt,
    mut offset: usize,
) -> Result<usize, Error> {
    if lower > upper || value < lower || value > upper {
        return Err(Error::Synthesis);
    }
    let shift = -lower;
    let shifted = value + &shift;
    let max = upper - lower;
    let shifted_unsigned = shifted.to_biguint().ok_or(Error::Synthesis)?;
    let max_unsigned = max.to_biguint().ok_or(Error::Synthesis)?;
    let diff = &max_unsigned - &shifted_unsigned;
    let bits = bit_length_biguint(&max_unsigned).max(1);
    if bits > FIELD_SAFE_BITS {
        return Err(Error::Synthesis);
    }

    let shift_cell = assign_constant_cell(
        region,
        config.advice[1],
        offset,
        fp_from_bigint_checked(&shift).map_err(|_| Error::Synthesis)?,
        "range shift",
    )?;
    offset += 1;
    let shifted_cell = assign_add(region, config, value_cell, &shift_cell, offset)?;
    offset += 1;
    offset = range_check_unsigned(
        region,
        config,
        &shifted_cell,
        &shifted_unsigned,
        bits,
        offset,
    )?;

    let max_cell = assign_constant_cell(
        region,
        config.advice[1],
        offset,
        fp_from_biguint_checked(&max_unsigned).map_err(|_| Error::Synthesis)?,
        "range max",
    )?;
    offset += 1;
    let diff_cell = region.assign_advice(
        || "range diff",
        config.advice[1],
        offset,
        || Value::known(fp_from_biguint_checked(&diff).expect("diff fits")),
    )?;
    offset += 1;
    let sum = assign_add(region, config, &shifted_cell, &diff_cell, offset)?;
    offset += 1;
    region.constrain_equal(sum.cell(), max_cell.cell())?;
    range_check_unsigned(region, config, &diff_cell, &diff, bits, offset)
}

fn range_check_unsigned(
    region: &mut halo2_proofs::circuit::Region<'_, Fp>,
    config: &LoraConfig,
    value_cell: &AssignedCell<Fp, Fp>,
    value: &BigUint,
    bits: usize,
    mut offset: usize,
) -> Result<usize, Error> {
    if bits == 0 || bits > FIELD_SAFE_BITS {
        return Err(Error::Synthesis);
    }
    let mut acc = assign_constant_cell(
        region,
        config.advice[0],
        offset,
        Fp::from(0),
        "range accumulator zero",
    )?;
    offset += 1;
    for bit_index in (0..bits).rev() {
        let bit_value = if ((value >> bit_index) & BigUint::one()).is_zero() {
            0
        } else {
            1
        };
        config.boolean.enable(region, offset)?;
        let bit = region.assign_advice(
            || "range bit",
            config.advice[0],
            offset,
            || Value::known(Fp::from(bit_value)),
        )?;
        offset += 1;
        let two = assign_constant_cell(region, config.advice[1], offset, Fp::from(2), "two")?;
        offset += 1;
        let doubled = assign_mul(region, config, &acc, &two, offset)?;
        offset += 1;
        acc = assign_add(region, config, &doubled, &bit, offset)?;
        offset += 1;
    }
    region.constrain_equal(acc.cell(), value_cell.cell())?;
    Ok(offset)
}

fn hash_adapter_words(
    config: &LoraConfig,
    layouter: &mut impl Layouter<Fp>,
    words: Vec<AssignedCell<Fp, Fp>>,
) -> Result<AssignedCell<Fp, Fp>, Error> {
    let mut acc = layouter.assign_region(
        || "adapter commitment init",
        |mut region| assign_constant_cell(&mut region, config.advice[0], 0, Fp::from(0), "zero"),
    )?;
    for (index, word) in words.into_iter().enumerate() {
        let chip = Pow5Chip::construct(config.poseidon_config.clone());
        let hasher = PoseidonHash::<_, _, P128Pow5T3, ConstantLength<2>, 3, 2>::init(
            chip,
            layouter.namespace(|| format!("adapter hash {index} init")),
        )?;
        acc = hasher.hash(
            layouter.namespace(|| format!("adapter hash {index}")),
            [acc, word],
        )?;
    }
    Ok(acc)
}

fn scale_bigint(config: &FixedPointConfig) -> BigInt {
    BigInt::one() << config.scale_bits
}

fn signed_bound(bits: u32) -> Result<BigInt, Error> {
    if bits == 0 || bits as usize > FIELD_SAFE_BITS {
        return Err(Error::Synthesis);
    }
    Ok((BigInt::one() << (bits - 1)) - BigInt::one())
}

fn check_signed_bound(value: &BigInt, bound: &BigInt, label: &str) -> Result<(), NativeError> {
    if value < &(-bound) || value > bound {
        return Err(NativeError::InvalidDimensions(format!(
            "{label} value {value} exceeds signed bound +/-{bound}"
        )));
    }
    Ok(())
}

fn validate_field_safety(circuit: &LoraCircuit) -> Result<(), NativeError> {
    let value_bits = circuit.fixed_point.value_bits as usize;
    let intermediate_bits = circuit.fixed_point.intermediate_bits as usize;
    let in_bits = ceil_log2(circuit.in_dim().max(1));
    let rank_bits = ceil_log2(circuit.rank().max(1));
    let scaling_bits = bit_length_i64(circuit.scaling_num).max(1);
    let candidates = [
        value_bits,
        intermediate_bits,
        value_bits.saturating_mul(2).saturating_add(in_bits),
        value_bits
            .saturating_add(intermediate_bits)
            .saturating_add(rank_bits),
        intermediate_bits.saturating_add(scaling_bits),
    ];
    if candidates.iter().any(|bits| *bits > FIELD_SAFE_BITS) {
        return Err(NativeError::InvalidDimensions(
            "fixed-point config and dimensions exceed Pasta field-safe integer bounds".to_string(),
        ));
    }
    Ok(())
}

fn canonical_remainder_interval(denominator: &BigInt) -> (BigInt, BigInt) {
    let two = BigInt::from(2u8);
    let floor_half = denominator / &two;
    (-&floor_half, (denominator - BigInt::one()) / &two)
}

fn div_round_to_canonical_interval(
    numerator: &BigInt,
    denominator: &BigInt,
) -> Result<BigInt, Error> {
    if denominator <= &BigInt::zero() {
        return Err(Error::Synthesis);
    }
    let two = BigInt::from(2u8);
    let floor_half = denominator / &two;
    Ok((numerator + &floor_half).div_floor(denominator))
}

fn fp_from_i64(value: i64) -> Fp {
    fp_from_bigint_checked(&BigInt::from(value)).expect("i64 fits in field-safe bound")
}

fn fp_from_bigint_checked(value: &BigInt) -> Result<Fp, NativeError> {
    if bit_length_bigint_abs(value) > FIELD_SAFE_BITS {
        return Err(NativeError::InvalidDimensions(
            "integer exceeds field-safe bit bound".to_string(),
        ));
    }
    if value.sign() == Sign::Minus {
        Ok(-fp_from_biguint_checked(
            &value.abs().to_biguint().expect("absolute value"),
        )?)
    } else {
        Ok(fp_from_biguint_checked(
            &value.to_biguint().expect("non-negative value"),
        )?)
    }
}

fn fp_from_biguint_checked(value: &BigUint) -> Result<Fp, NativeError> {
    if bit_length_biguint(value) > FIELD_SAFE_BITS {
        return Err(NativeError::InvalidDimensions(
            "integer exceeds field-safe bit bound".to_string(),
        ));
    }
    Ok(fp_from_biguint(value))
}

fn fp_from_biguint(value: &BigUint) -> Fp {
    let mut out = Fp::from(0);
    let mut base = Fp::from(1);
    let limb_base = Fp::from_raw([0, 1, 0, 0]);
    for limb in value.to_u64_digits() {
        out += base * Fp::from(limb);
        base *= limb_base;
    }
    out
}

fn fp_to_decimal(value: Fp) -> String {
    let repr = value.to_repr();
    let bigint = BigUint::from_bytes_le(repr.as_ref());
    bigint.to_str_radix(10)
}

fn parse_adapter_commitment(value: &str) -> Result<BigUint, NativeError> {
    let parsed = BigUint::parse_bytes(value.as_bytes(), 10).ok_or_else(|| {
        NativeError::InvalidDimensions("adapter_commitment must be a decimal field string".into())
    })?;
    let canonical = fp_to_decimal(fp_from_biguint(&parsed));
    let normalized = value.trim_start_matches('0');
    let normalized = if normalized.is_empty() {
        "0"
    } else {
        normalized
    };
    if canonical != normalized {
        return Err(NativeError::InvalidDimensions(
            "adapter_commitment must be a canonical field string".to_string(),
        ));
    }
    Ok(parsed)
}

fn statement_digest_limbs(statement_digest: &str) -> Result<[BigUint; 2], NativeError> {
    let digest = statement_digest
        .strip_prefix("0x")
        .unwrap_or(statement_digest);
    if digest.len() != 64 || !digest.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(NativeError::InvalidDimensions(
            "statement_digest must be 32 bytes of hex".to_string(),
        ));
    }
    let high = BigUint::parse_bytes(digest[..32].as_bytes(), 16).ok_or_else(|| {
        NativeError::InvalidDimensions("invalid statement_digest high limb".to_string())
    })?;
    let low = BigUint::parse_bytes(digest[32..].as_bytes(), 16).ok_or_else(|| {
        NativeError::InvalidDimensions("invalid statement_digest low limb".to_string())
    })?;
    Ok([high, low])
}

fn bit_length_biguint(value: &BigUint) -> usize {
    value.bits() as usize
}

fn bit_length_bigint_abs(value: &BigInt) -> usize {
    value.abs().to_biguint().map_or(0, |v| v.bits() as usize)
}

fn bit_length_i64(value: i64) -> usize {
    if value == 0 {
        0
    } else {
        64 - value.unsigned_abs().leading_zeros() as usize
    }
}

fn ceil_log2(value: usize) -> usize {
    if value <= 1 {
        0
    } else {
        usize::BITS as usize - (value - 1).leading_zeros() as usize
    }
}

#[cfg(any(test, feature = "python"))]
fn adapter_commitment_words_from_input(
    input: &AdapterCommitmentInput,
) -> Result<Vec<Fp>, NativeError> {
    let mut words = Vec::new();
    for value in [
        BigInt::from(ADAPTER_COMMITMENT_DOMAIN),
        BigInt::from(ADAPTER_COMMITMENT_VERSION),
        BigInt::from(input.schema_version),
        BigInt::from(input.in_dim),
        BigInt::from(input.rank),
        BigInt::from(input.out_dim),
        BigInt::from(input.fixed_point.scale_bits),
        BigInt::from(input.fixed_point.value_bits),
        BigInt::from(input.fixed_point.intermediate_bits),
        BigInt::from(input.scaling_num),
        BigInt::from(input.scaling_den),
    ] {
        words.push(fp_from_bigint_checked(&value)?);
    }
    for value in input.a.iter().flatten().chain(input.b.iter().flatten()) {
        words.push(fp_from_i64(*value));
    }
    Ok(words)
}

#[cfg(any(test, feature = "python"))]
fn adapter_commitment_for_input(input: &AdapterCommitmentInput) -> Result<String, NativeError> {
    let mut acc = Fp::from(0);
    for word in adapter_commitment_words_from_input(input)? {
        acc =
            NativePoseidonHash::<_, P128Pow5T3, ConstantLength<2>, 3, 2>::init().hash([acc, word]);
    }
    Ok(fp_to_decimal(acc))
}

fn public_inputs(circuit: &LoraCircuit) -> Result<Vec<Fp>, NativeError> {
    let mut inputs: Vec<Fp> = circuit
        .x
        .iter()
        .chain(circuit.delta.iter())
        .map(|value| fp_from_i64(*value))
        .collect();
    inputs.push(fp_from_bigint_checked(&scale_bigint(&circuit.fixed_point))?);
    inputs.push(fp_from_i64(circuit.scaling_num));
    inputs.push(fp_from_i64(circuit.scaling_den));
    for limb in statement_digest_limbs(&circuit.statement_digest)? {
        inputs.push(fp_from_biguint_checked(&limb)?);
    }
    inputs.push(fp_from_biguint(&parse_adapter_commitment(
        &circuit.adapter_commitment,
    )?));
    Ok(inputs)
}

fn rows_needed(circuit: &LoraCircuit) -> usize {
    let value_bits = circuit.fixed_point.value_bits as usize;
    let intermediate_bits = circuit.fixed_point.intermediate_bits as usize;
    let raw_a_bits = value_bits
        .saturating_mul(2)
        .saturating_add(ceil_log2(circuit.in_dim().max(1)));
    let raw_b_bits = value_bits
        .saturating_add(intermediate_bits)
        .saturating_add(ceil_log2(circuit.rank().max(1)));
    let scaling_bits = bit_length_i64(circuit.scaling_num).max(1);
    let scaled_bits = intermediate_bits.saturating_add(scaling_bits);
    let product_bits = value_bits
        .saturating_mul(2)
        .max(value_bits.saturating_add(intermediate_bits))
        .max(scaled_bits);
    let accumulator_bits = raw_a_bits.max(raw_b_bits);
    let range_rows = |bits: usize| 8 * bits.max(1) + 16;
    let matrix_values = circuit.rank() * circuit.in_dim() + circuit.out_dim() * circuit.rank();
    let products = matrix_values + circuit.out_dim();
    let divs = circuit.rank() + 2 * circuit.out_dim();
    let adapter_words = 11 + matrix_values;
    32 + circuit.in_dim()
        + 4 * matrix_values
        + 4 * products
        + 4 * circuit.out_dim()
        + circuit.in_dim() * range_rows(value_bits)
        + matrix_values * range_rows(value_bits)
        + products * range_rows(product_bits)
        + (circuit.rank() + circuit.out_dim()) * range_rows(accumulator_bits)
        + divs
            * (range_rows(raw_a_bits.max(raw_b_bits).max(scaled_bits))
                + range_rows(intermediate_bits))
        + adapter_words * POSEIDON_PAIR_ROWS
}

fn k_for(circuit: &LoraCircuit) -> u32 {
    let rows = rows_needed(circuit).next_power_of_two();
    rows.trailing_zeros().max(8)
}

/// Cache key covering everything the circuit structure (and therefore the
/// params/keys) depends on: the in-circuit constants are derived solely from
/// dims, fixed-point widths, and scaling; witness values never enter keygen.
type LegacyShapeKey = (u32, usize, usize, usize, u32, u32, u32, i64, i64);

/// Both caches are bounded: the cache keys span every attacker-influenceable
/// statement field, so an unbounded map fed varying shapes is a slow memory
/// DoS. Entries near k = MAX_LEGACY_K are GB-scale, hence the small caps.
const MAX_LEGACY_KEY_CACHE_ENTRIES: usize = 4;
const MAX_LEGACY_PARAMS_CACHE_ENTRIES: usize = 4;

/// Minimal LRU map: a HashMap with a monotonically increasing use stamp per
/// entry; inserting beyond capacity evicts the least recently used entry.
struct BoundedLru<K, V> {
    map: HashMap<K, (u64, V)>,
    counter: u64,
    capacity: usize,
}

impl<K: std::hash::Hash + Eq + Clone, V: Clone> BoundedLru<K, V> {
    fn new(capacity: usize) -> Self {
        BoundedLru {
            map: HashMap::new(),
            counter: 0,
            capacity: capacity.max(1),
        }
    }

    fn get(&mut self, key: &K) -> Option<V> {
        self.counter += 1;
        let stamp = self.counter;
        self.map.get_mut(key).map(|slot| {
            slot.0 = stamp;
            slot.1.clone()
        })
    }

    fn insert(&mut self, key: K, value: V) {
        self.counter += 1;
        if !self.map.contains_key(&key) && self.map.len() >= self.capacity {
            if let Some(oldest) = self
                .map
                .iter()
                .min_by_key(|(_, (stamp, _))| *stamp)
                .map(|(k, _)| k.clone())
            {
                self.map.remove(&oldest);
            }
        }
        self.map.insert(key, (self.counter, value));
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.map.len()
    }

    #[cfg(test)]
    fn contains_key(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }
}

/// Verification only ever needs the verifying key; the proving key is built
/// (and cached) lazily the first time a shape is actually proven, so a
/// verifier never pays keygen_pk for hostile or one-off shapes.
enum LegacyKeys {
    VerifyOnly {
        params: Arc<Params<EqAffine>>,
        vk: VerifyingKey<EqAffine>,
    },
    Prover {
        params: Arc<Params<EqAffine>>,
        pk: ProvingKey<EqAffine>,
    },
}

impl LegacyKeys {
    fn params(&self) -> &Params<EqAffine> {
        match self {
            LegacyKeys::VerifyOnly { params, .. } => params,
            LegacyKeys::Prover { params, .. } => params,
        }
    }

    fn vk(&self) -> &VerifyingKey<EqAffine> {
        match self {
            LegacyKeys::VerifyOnly { vk, .. } => vk,
            LegacyKeys::Prover { pk, .. } => pk.get_vk(),
        }
    }

    fn pk(&self) -> Option<&ProvingKey<EqAffine>> {
        match self {
            LegacyKeys::VerifyOnly { .. } => None,
            LegacyKeys::Prover { pk, .. } => Some(pk),
        }
    }
}

static LEGACY_KEY_CACHE: OnceLock<Mutex<BoundedLru<LegacyShapeKey, Arc<LegacyKeys>>>> =
    OnceLock::new();
static LEGACY_PARAMS_CACHE: OnceLock<Mutex<BoundedLru<u32, Arc<Params<EqAffine>>>>> =
    OnceLock::new();

/// Cached values are immutable once inserted (Arc'd keys/params plus LRU
/// bookkeeping), so a panic in another thread cannot leave them torn;
/// recover from poisoning instead of propagating panics through PyO3.
fn lock_recovering<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn legacy_cache() -> &'static Mutex<BoundedLru<LegacyShapeKey, Arc<LegacyKeys>>> {
    LEGACY_KEY_CACHE.get_or_init(|| Mutex::new(BoundedLru::new(MAX_LEGACY_KEY_CACHE_ENTRIES)))
}

fn legacy_params_for(k: u32) -> Arc<Params<EqAffine>> {
    let cache = LEGACY_PARAMS_CACHE
        .get_or_init(|| Mutex::new(BoundedLru::new(MAX_LEGACY_PARAMS_CACHE_ENTRIES)));
    if let Some(found) = lock_recovering(cache).get(&k) {
        return found;
    }
    // Built outside the lock: Params::new at large k takes seconds and two
    // racing builders are deterministic, so last-write-wins is harmless.
    let params = Arc::new(Params::<EqAffine>::new(k));
    lock_recovering(cache).insert(k, params.clone());
    params
}

fn legacy_shape_key(circuit: &LoraCircuit, k: u32) -> LegacyShapeKey {
    (
        k,
        circuit.in_dim(),
        circuit.rank(),
        circuit.out_dim(),
        circuit.fixed_point.scale_bits,
        circuit.fixed_point.value_bits,
        circuit.fixed_point.intermediate_bits,
        circuit.scaling_num,
        circuit.scaling_den,
    )
}

fn legacy_keys_for(circuit: &LoraCircuit, need_pk: bool) -> Result<Arc<LegacyKeys>, NativeError> {
    let k = k_for(circuit);
    if k > MAX_LEGACY_K {
        return Err(NativeError::InvalidDimensions(format!(
            "legacy artifact exceeds verification caps: k {k} beyond {MAX_LEGACY_K}"
        )));
    }
    let key = legacy_shape_key(circuit, k);
    if let Some(found) = lock_recovering(legacy_cache()).get(&key) {
        if !need_pk || found.pk().is_some() {
            return Ok(found);
        }
    }
    // Keygen runs outside the lock so concurrent callers on other shapes are
    // not serialized behind it; duplicated keygen on the same shape is
    // deterministic and last-write-wins.
    let params = legacy_params_for(k);
    let empty = circuit.without_witnesses();
    let vk = keygen_vk(&params, &empty).map_err(|e| NativeError::Halo2(e.to_string()))?;
    let entry = if need_pk {
        let pk = keygen_pk(&params, vk, &empty).map_err(|e| NativeError::Halo2(e.to_string()))?;
        Arc::new(LegacyKeys::Prover { params, pk })
    } else {
        Arc::new(LegacyKeys::VerifyOnly { params, vk })
    };
    lock_recovering(legacy_cache()).insert(key, entry.clone());
    Ok(entry)
}

fn circuit_from_json(statement_json: &str, witness_json: &str) -> Result<LoraCircuit, NativeError> {
    let statement: NativeStatement =
        serde_json::from_str(statement_json).map_err(|e| NativeError::Json(e.to_string()))?;
    let witness: NativeWitness =
        serde_json::from_str(witness_json).map_err(|e| NativeError::Json(e.to_string()))?;
    let circuit = LoraCircuit {
        a: witness.a,
        b: witness.b,
        x: statement.x,
        delta: statement.delta,
        fixed_point: statement.fixed_point,
        scaling_num: statement.scaling_num,
        scaling_den: statement.scaling_den,
        adapter_commitment: statement.adapter_commitment,
        statement_digest: statement.statement_digest,
    };
    if statement.rank != 0 && statement.rank != circuit.rank() {
        return Err(NativeError::InvalidDimensions(
            "statement rank does not match witness".to_string(),
        ));
    }
    circuit.validate()?;
    Ok(circuit)
}

fn default_rank() -> usize {
    1
}

fn default_scaling_num() -> i64 {
    1
}

fn default_scaling_den() -> i64 {
    1
}

pub fn prove_bytes(statement_json: &str, witness_json: &str) -> Result<Vec<u8>, NativeError> {
    let circuit = circuit_from_json(statement_json, witness_json)?;
    let keys = legacy_keys_for(&circuit, true)?;
    let pk = keys.pk().expect("prover cache entry carries a proving key");
    let instances = public_inputs(&circuit)?;
    let instance_refs: Vec<&[Fp]> = vec![instances.as_slice()];
    let mut transcript = Blake2bWrite::<_, vesta::Affine, Challenge255<_>>::init(vec![]);
    create_proof(
        keys.params(),
        pk,
        &[circuit],
        &[instance_refs.as_slice()],
        &mut OsRng,
        &mut transcript,
    )
    .map_err(|e| NativeError::Halo2(e.to_string()))?;
    Ok(transcript.finalize())
}

pub fn verify_bytes(statement_json: &str, proof: &[u8]) -> Result<bool, NativeError> {
    let statement: NativeStatement =
        serde_json::from_str(statement_json).map_err(|e| NativeError::Json(e.to_string()))?;
    let witness_shape = NativeWitness {
        a: vec![vec![0; statement.x.len()]; statement.rank.max(1)],
        b: vec![vec![0; statement.rank.max(1)]; statement.delta.len()],
    };
    let circuit = circuit_from_json(
        statement_json,
        &serde_json::to_string(&witness_shape).map_err(|e| NativeError::Json(e.to_string()))?,
    )?;
    let keys = legacy_keys_for(&circuit, false)?;
    let instances = public_inputs(&circuit)?;
    let instance_refs: Vec<&[Fp]> = vec![instances.as_slice()];
    let mut transcript = Blake2bRead::<_, vesta::Affine, Challenge255<_>>::init(proof);
    let result = verify_proof(
        keys.params(),
        keys.vk(),
        SingleVerifier::new(keys.params()),
        &[instance_refs.as_slice()],
        &mut transcript,
    );
    Ok(result.is_ok())
}

pub fn statement_digest_hex(statement_json: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(statement_json.as_bytes());
    format!("{:x}", hasher.finalize())
}

#[cfg(feature = "python")]
#[pyo3::pyfunction]
fn prove(statement_json: &str, witness_json: &str) -> pyo3::PyResult<Vec<u8>> {
    Ok(prove_bytes(statement_json, witness_json)?)
}

#[cfg(feature = "python")]
#[pyo3::pyfunction]
fn verify(statement_json: &str, proof: &[u8]) -> pyo3::PyResult<bool> {
    Ok(verify_bytes(statement_json, proof)?)
}

#[cfg(feature = "python")]
#[pyo3::pyfunction]
fn statement_digest(statement_json: &str) -> pyo3::PyResult<String> {
    Ok(statement_digest_hex(statement_json))
}

#[cfg(feature = "python")]
#[pyo3::pyfunction]
fn adapter_commitment(adapter_json: &str) -> pyo3::PyResult<String> {
    let input: AdapterCommitmentInput =
        serde_json::from_str(adapter_json).map_err(|e| NativeError::Json(e.to_string()))?;
    Ok(adapter_commitment_for_input(&input)?)
}

#[cfg(feature = "python")]
#[pyo3::pymodule]
fn _native_prover(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::types::PyModuleMethods;
    use pyo3::wrap_pyfunction;

    m.add_function(wrap_pyfunction!(prove, m)?)?;
    m.add_function(wrap_pyfunction!(verify, m)?)?;
    m.add_function(wrap_pyfunction!(statement_digest, m)?)?;
    m.add_function(wrap_pyfunction!(adapter_commitment, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::dev::MockProver;

    fn adapter_input() -> AdapterCommitmentInput {
        AdapterCommitmentInput {
            schema_version: 2,
            in_dim: 2,
            rank: 1,
            out_dim: 2,
            fixed_point: FixedPointConfig {
                scale_bits: 0,
                value_bits: 8,
                intermediate_bits: 16,
            },
            scaling_num: 1,
            scaling_den: 1,
            a: vec![vec![2, -1]],
            b: vec![vec![3], vec![-2]],
        }
    }

    fn valid_circuit() -> LoraCircuit {
        let input = adapter_input();
        LoraCircuit {
            a: input.a.clone(),
            b: input.b.clone(),
            x: vec![4, 5],
            delta: vec![9, -6],
            fixed_point: input.fixed_point.clone(),
            scaling_num: input.scaling_num,
            scaling_den: input.scaling_den,
            adapter_commitment: adapter_commitment_for_input(&input).unwrap(),
            statement_digest: "11".repeat(32),
        }
    }

    fn minimal_circuit() -> LoraCircuit {
        let input = AdapterCommitmentInput {
            schema_version: ARTIFACT_SCHEMA_VERSION,
            in_dim: 1,
            rank: 1,
            out_dim: 1,
            fixed_point: FixedPointConfig {
                scale_bits: 1,
                value_bits: 8,
                intermediate_bits: 16,
            },
            scaling_num: 1,
            scaling_den: 1,
            a: vec![vec![1]],
            b: vec![vec![-2]],
        };
        LoraCircuit {
            a: input.a.clone(),
            b: input.b.clone(),
            x: vec![2],
            delta: vec![-1],
            fixed_point: input.fixed_point.clone(),
            scaling_num: input.scaling_num,
            scaling_den: input.scaling_den,
            adapter_commitment: adapter_commitment_for_input(&input).unwrap(),
            statement_digest: "22".repeat(32),
        }
    }

    #[test]
    fn poseidon_adapter_commitment_is_deterministic() {
        let input = adapter_input();
        let first = adapter_commitment_for_input(&input).unwrap();
        let second = adapter_commitment_for_input(&input).unwrap();
        assert_eq!(first, second);
        let mut changed = input;
        changed.b[0][0] += 1;
        assert_ne!(first, adapter_commitment_for_input(&changed).unwrap());
    }

    #[test]
    fn legacy_key_cache_reuses_and_upgrades_entries() {
        let circuit = minimal_circuit();
        let verify_only = legacy_keys_for(&circuit, false).unwrap();
        assert!(verify_only.pk().is_none());
        let cached = legacy_keys_for(&circuit, false).unwrap();
        assert!(Arc::ptr_eq(&verify_only, &cached));

        // A prover call on the same shape upgrades the entry in place...
        let prover = legacy_keys_for(&circuit, true).unwrap();
        assert!(prover.pk().is_some());
        // ...and both later verifiers and provers share the upgraded entry.
        let reused_verify = legacy_keys_for(&circuit, false).unwrap();
        assert!(Arc::ptr_eq(&prover, &reused_verify));
        let reused_prover = legacy_keys_for(&circuit, true).unwrap();
        assert!(Arc::ptr_eq(&prover, &reused_prover));

        let key = legacy_shape_key(&circuit, k_for(&circuit));
        assert!(lock_recovering(legacy_cache()).contains_key(&key));
    }

    #[test]
    fn bounded_lru_evicts_least_recently_used() {
        let mut lru: BoundedLru<u32, u32> = BoundedLru::new(2);
        lru.insert(1, 10);
        lru.insert(2, 20);
        assert_eq!(lru.get(&1), Some(10)); // touch 1 so 2 becomes the oldest
        lru.insert(3, 30);
        assert_eq!(lru.len(), 2);
        assert!(lru.contains_key(&1));
        assert!(!lru.contains_key(&2));
        assert!(lru.contains_key(&3));

        // Re-inserting an existing key must not evict anything.
        lru.insert(1, 11);
        assert_eq!(lru.len(), 2);
        assert_eq!(lru.get(&1), Some(11));
        assert!(lru.contains_key(&3));
    }

    #[test]
    fn legacy_caps_reject_oversized_dimensions() {
        let fixed_point = FixedPointConfig {
            scale_bits: 1,
            value_bits: 8,
            intermediate_bits: 16,
        };
        let wide = LoraCircuit {
            a: vec![vec![0; MAX_LEGACY_DIM + 1]],
            b: vec![vec![0]],
            x: vec![0; MAX_LEGACY_DIM + 1],
            delta: vec![0],
            fixed_point: fixed_point.clone(),
            scaling_num: 1,
            scaling_den: 1,
            adapter_commitment: "0".to_string(),
            statement_digest: "22".repeat(32),
        };
        let err = wide.validate().unwrap_err();
        assert!(err.to_string().contains("exceeds verification caps"));

        let deep = LoraCircuit {
            a: vec![vec![0]; MAX_LEGACY_RANK + 1],
            b: vec![vec![0; MAX_LEGACY_RANK + 1]],
            x: vec![0],
            delta: vec![0],
            fixed_point,
            scaling_num: 1,
            scaling_den: 1,
            adapter_commitment: "0".to_string(),
            statement_digest: "22".repeat(32),
        };
        let err = deep.validate().unwrap_err();
        assert!(err.to_string().contains("exceeds verification caps"));
    }

    #[test]
    fn mock_prover_accepts_valid_lora_relation() {
        let circuit = valid_circuit();
        let instances = public_inputs(&circuit).unwrap();
        let prover = MockProver::run(k_for(&circuit), &circuit, vec![instances]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }

    #[test]
    fn mock_prover_rejects_tampered_delta_and_commitment() {
        let circuit = valid_circuit();
        let mut instances = public_inputs(&circuit).unwrap();
        instances[circuit.in_dim()] += Fp::from(1);
        let prover = MockProver::run(k_for(&circuit), &circuit, vec![instances]).unwrap();
        assert!(prover.verify().is_err());

        let mut commitment_instances = public_inputs(&circuit).unwrap();
        let last = commitment_instances.len() - 1;
        commitment_instances[last] += Fp::from(1);
        let prover =
            MockProver::run(k_for(&circuit), &circuit, vec![commitment_instances]).unwrap();
        assert!(prover.verify().is_err());
    }

    #[test]
    #[ignore = "IPA proof generation for the Poseidon/range-check circuit is intentionally slow"]
    fn real_proof_verifies_for_tiny_relation() {
        let circuit = minimal_circuit();
        let statement = NativeStatement {
            x: circuit.x.clone(),
            delta: circuit.delta.clone(),
            fixed_point: circuit.fixed_point.clone(),
            rank: circuit.rank(),
            scaling_num: circuit.scaling_num,
            scaling_den: circuit.scaling_den,
            adapter_commitment: circuit.adapter_commitment.clone(),
            statement_digest: circuit.statement_digest.clone(),
        };
        let witness = NativeWitness {
            a: circuit.a.clone(),
            b: circuit.b.clone(),
        };
        let statement_json = serde_json::to_string(&statement).unwrap();
        let witness_json = serde_json::to_string(&witness).unwrap();
        let proof = prove_bytes(&statement_json, &witness_json).unwrap();
        assert!(verify_bytes(&statement_json, &proof).unwrap());

        let mut tampered = statement;
        tampered.adapter_commitment = adapter_commitment_for_input(&AdapterCommitmentInput {
            b: vec![vec![4], vec![-2]],
            ..adapter_input()
        })
        .unwrap();
        let tampered_json = serde_json::to_string(&tampered).unwrap();
        assert!(!verify_bytes(&tampered_json, &proof).unwrap());
    }

    #[test]
    fn deterministic_half_point_rounding_is_unique() {
        let statement = NativeStatement {
            x: vec![1],
            delta: vec![1],
            fixed_point: FixedPointConfig {
                scale_bits: 1,
                value_bits: 8,
                intermediate_bits: 16,
            },
            rank: 1,
            scaling_num: 1,
            scaling_den: 1,
            adapter_commitment: adapter_commitment_for_input(&AdapterCommitmentInput {
                schema_version: 2,
                in_dim: 1,
                rank: 1,
                out_dim: 1,
                fixed_point: FixedPointConfig {
                    scale_bits: 1,
                    value_bits: 8,
                    intermediate_bits: 16,
                },
                scaling_num: 1,
                scaling_den: 1,
                a: vec![vec![1]],
                b: vec![vec![2]],
            })
            .unwrap(),
            statement_digest: "22".repeat(32),
        };
        let witness = NativeWitness {
            a: vec![vec![1]],
            b: vec![vec![2]],
        };
        let statement_json = serde_json::to_string(&statement).unwrap();
        let witness_json = serde_json::to_string(&witness).unwrap();
        let circuit = circuit_from_json(&statement_json, &witness_json).unwrap();
        let instances = public_inputs(&circuit).unwrap();
        let prover = MockProver::run(k_for(&circuit), &circuit, vec![instances]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }
}
