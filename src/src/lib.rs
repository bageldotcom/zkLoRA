use ff::PrimeField;
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
        ConstraintSystem, Error, Fixed, Instance, ProvingKey, Selector, SingleVerifier,
        TableColumn, VerifyingKey,
    },
    poly::commitment::Params,
    poly::Rotation,
    transcript::{Blake2bRead, Blake2bWrite, Challenge255},
};
use num_bigint::{BigInt, BigUint, Sign};
use num_integer::Integer;
use num_traits::{One, Signed, Zero};
use rand_core::OsRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};
use std::convert::TryInto;
use std::sync::{Arc, Mutex, OnceLock};

const ADAPTER_COMMITMENT_DOMAIN: u64 = 0x5a4b4c4f5241; // "ZKLORA"
const ADAPTER_COMMITMENT_VERSION: u64 = 1;
// Must match proof_contract.SCHEMA_VERSION; it is hashed into adapter commitments.
const ARTIFACT_SCHEMA_VERSION: u64 = 2;
const FIELD_SAFE_BITS: usize = 250;
const POSEIDON_PAIR_ROWS: usize = 96;
const RANGE_WINDOW_MIN: u32 = 4;
const RANGE_WINDOW_MAX: u32 = 16;
const ROW_MARGIN: usize = 128;

/// Cache key capturing everything the circuit layout (and therefore the
/// params/proving key/verifying key) depends on. Witness values, the adapter
/// commitment, and the statement digest are advice/instance data and do not
/// influence key generation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CircuitKey {
    in_dim: usize,
    rank: usize,
    out_dim: usize,
    scale_bits: u32,
    value_bits: u32,
    intermediate_bits: u32,
    scaling_num: i64,
    scaling_den: i64,
}

impl CircuitKey {
    fn from_circuit(circuit: &LoraCircuit) -> Self {
        Self {
            in_dim: circuit.in_dim(),
            rank: circuit.rank(),
            out_dim: circuit.out_dim(),
            scale_bits: circuit.fixed_point.scale_bits,
            value_bits: circuit.fixed_point.value_bits,
            intermediate_bits: circuit.fixed_point.intermediate_bits,
            scaling_num: circuit.scaling_num,
            scaling_den: circuit.scaling_den,
        }
    }
}

/// Simple bounded FIFO cache. Proving keys and SRS params at large `k` are
/// hundreds of MB, so we cap how many distinct shapes stay resident.
struct BoundedCache<K, V> {
    map: HashMap<K, Arc<V>>,
    order: VecDeque<K>,
    cap: usize,
}

impl<K: Clone + std::hash::Hash + Eq, V> BoundedCache<K, V> {
    fn new(cap: usize) -> Self {
        Self {
            map: HashMap::new(),
            order: VecDeque::new(),
            cap: cap.max(1),
        }
    }

    fn get_or_create<E>(
        &mut self,
        key: &K,
        create: impl FnOnce() -> Result<V, E>,
    ) -> Result<Arc<V>, E> {
        if let Some(value) = self.map.get(key) {
            return Ok(value.clone());
        }
        let value = Arc::new(create()?);
        while self.order.len() >= self.cap {
            if let Some(evicted) = self.order.pop_front() {
                self.map.remove(&evicted);
            }
        }
        self.order.push_back(key.clone());
        self.map.insert(key.clone(), value.clone());
        Ok(value)
    }
}

fn cache_cap(env_var: &str, default: usize) -> usize {
    std::env::var(env_var)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
        .max(1)
}

fn params_for(k: u32) -> Arc<Params<EqAffine>> {
    static CACHE: OnceLock<Mutex<BoundedCache<u32, Params<EqAffine>>>> = OnceLock::new();
    let cache = CACHE
        .get_or_init(|| Mutex::new(BoundedCache::new(cache_cap("ZKLORA_PARAMS_CACHE_CAP", 4))));
    let mut guard = cache.lock().expect("params cache poisoned");
    guard
        .get_or_create::<std::convert::Infallible>(&k, || Ok(Params::new(k)))
        .expect("params creation is infallible")
}

fn proving_key_for(
    key: &CircuitKey,
    params: &Params<EqAffine>,
    empty_circuit: &LoraCircuit,
) -> Result<Arc<ProvingKey<EqAffine>>, NativeError> {
    static CACHE: OnceLock<Mutex<BoundedCache<CircuitKey, ProvingKey<EqAffine>>>> = OnceLock::new();
    let cache =
        CACHE.get_or_init(|| Mutex::new(BoundedCache::new(cache_cap("ZKLORA_PK_CACHE_CAP", 2))));
    let mut guard = cache.lock().expect("pk cache poisoned");
    guard.get_or_create(key, || {
        let vk = keygen_vk(params, empty_circuit).map_err(|e| NativeError::Halo2(e.to_string()))?;
        keygen_pk(params, vk, empty_circuit).map_err(|e| NativeError::Halo2(e.to_string()))
    })
}

fn verifying_key_for(
    key: &CircuitKey,
    params: &Params<EqAffine>,
    empty_circuit: &LoraCircuit,
) -> Result<Arc<VerifyingKey<EqAffine>>, NativeError> {
    static CACHE: OnceLock<Mutex<BoundedCache<CircuitKey, VerifyingKey<EqAffine>>>> =
        OnceLock::new();
    let cache =
        CACHE.get_or_init(|| Mutex::new(BoundedCache::new(cache_cap("ZKLORA_VK_CACHE_CAP", 8))));
    let mut guard = cache.lock().expect("vk cache poisoned");
    guard.get_or_create(key, || {
        keygen_vk(params, empty_circuit).map_err(|e| NativeError::Halo2(e.to_string()))
    })
}

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
    /// Complex selector activating one step of the range-check running sum.
    q_range: Selector,
    /// Per-row decomposition radix (2^window on active rows).
    rc_factor: Column<Fixed>,
    /// Per-row limb scale: 1 on full limbs, 2^(window - top_bits) on the top
    /// limb so the most-significant limb is constrained to exactly its width.
    rc_scale: Column<Fixed>,
    /// Lookup table holding 0..2^window.
    rc_table: TableColumn,
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

        // Lookup-based range decomposition: a running sum z walks down the
        // value, peeling one limb per active row. On row i the constraint
        //   q * scale_i * (z_i - z_{i+1} * factor_i)  ∈  rc_table
        // forces limb_i = z_i - z_{i+1} * 2^window into [0, 2^window), and the
        // top limb is multiplied by 2^(window - top_bits) so it is bounded to
        // exactly its residual width. With z_n constrained to zero the chain
        // telescopes to z_0 = Σ limb_i · 2^(window·i) < 2^bits with no padding
        // slack, preserving the exact interval semantics of the previous
        // bit-decomposition while using ~window× fewer rows.
        let q_range = meta.complex_selector();
        let rc_factor = meta.fixed_column();
        let rc_scale = meta.fixed_column();
        let rc_table = meta.lookup_table_column();
        meta.lookup(|meta| {
            let q = meta.query_selector(q_range);
            let z_cur = meta.query_advice(advice[3], Rotation::cur());
            let z_next = meta.query_advice(advice[3], Rotation::next());
            let factor = meta.query_fixed(rc_factor);
            let scale = meta.query_fixed(rc_scale);
            vec![(q * scale * (z_cur - z_next * factor), rc_table)]
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
            q_range,
            rc_factor,
            rc_scale,
            rc_table,
            poseidon_config,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        self.validate().map_err(|_| Error::Synthesis)?;
        let window = self.window_and_k().0;

        layouter.assign_table(
            || "range check table",
            |mut table| {
                for value in 0..(1u64 << window) {
                    table.assign_cell(
                        || "range table value",
                        config.rc_table,
                        value as usize,
                        || Value::known(Fp::from(value)),
                    )?;
                }
                Ok(())
            },
        )?;

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
                        window,
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
                            window,
                            &weight,
                            &weight_big,
                            &(-&value_bound),
                            &value_bound,
                            offset,
                        )?;
                        adapter_words.push(weight.clone());

                        // The product of two individually range-checked values
                        // cannot wrap the field (|w·x| <= value_bound^2 and the
                        // accumulated sum stays below the field-safe bound per
                        // validate_field_safety), so the accumulator is bounded
                        // once inside assign_div_round instead of per product.
                        let product = assign_mul(
                            &mut region,
                            &config,
                            &x_cells[input_index],
                            &weight,
                            offset,
                        )?;
                        offset += 1;
                        let product_value = &weight_big * BigInt::from(self.x[input_index]);
                        let next_acc = assign_add(&mut region, &config, &acc, &product, offset)?;
                        offset += 1;
                        raw_value += product_value;
                        acc = next_acc;
                    }
                    let q = div_round_to_canonical_interval(&raw_value, &scale)?;
                    let (intermediate_cell, next_offset) = assign_div_round(
                        &mut region,
                        &config,
                        window,
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
                            window,
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
                        let next_acc = assign_add(&mut region, &config, &acc, &product, offset)?;
                        offset += 1;
                        raw_value += product_value;
                        acc = next_acc;
                    }
                    let rescaled = div_round_to_canonical_interval(&raw_value, &scale)?;
                    let (rescaled_cell, next_offset) = assign_div_round(
                        &mut region,
                        &config,
                        window,
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

                    let scaling_den_big = BigInt::from(self.scaling_den);
                    let final_delta =
                        div_round_to_canonical_interval(&scaled_raw, &scaling_den_big)?;
                    let (final_cell, next_offset) = assign_div_round(
                        &mut region,
                        &config,
                        window,
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

    fn window_and_k(&self) -> (u32, u32) {
        window_and_k_for(self)
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
    window: u32,
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
        window,
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
        window,
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
        window,
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
    window: u32,
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
        window,
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
    range_check_unsigned(region, config, window, &diff_cell, &diff, bits, offset)
}

/// Constrain `value_cell` to `[0, 2^bits)` using a lookup-backed running sum.
///
/// Row `offset + i` holds `z_i` in `advice[3]`; the lookup argument enforces
/// `scale_i * (z_i - z_{i+1} * 2^window) ∈ [0, 2^window)` per active row, with
/// `scale_i = 2^(window - top_bits)` on the final limb so the decomposition
/// covers exactly `bits` bits. `z_n` is constrained to zero, so the chain
/// telescopes to `z_0 = value < 2^bits` with every limb non-negative.
fn range_check_unsigned(
    region: &mut halo2_proofs::circuit::Region<'_, Fp>,
    config: &LoraConfig,
    window: u32,
    value_cell: &AssignedCell<Fp, Fp>,
    value: &BigUint,
    bits: usize,
    mut offset: usize,
) -> Result<usize, Error> {
    if bits == 0 || bits > FIELD_SAFE_BITS || window == 0 {
        return Err(Error::Synthesis);
    }
    let limb_count = bits.div_ceil(window as usize);
    let top_bits = bits - (limb_count - 1) * window as usize;
    let factor = Fp::from(1u64 << window);
    let mut z_value = value.clone();

    for limb_index in 0..limb_count {
        config.q_range.enable(region, offset)?;
        region.assign_fixed(
            || "range factor",
            config.rc_factor,
            offset,
            || Value::known(factor),
        )?;
        let scale = if limb_index + 1 == limb_count {
            Fp::from(1u64 << (window as usize - top_bits))
        } else {
            Fp::from(1)
        };
        region.assign_fixed(
            || "range scale",
            config.rc_scale,
            offset,
            || Value::known(scale),
        )?;
        let z_cell = region.assign_advice(
            || "range z",
            config.advice[3],
            offset,
            || Value::known(fp_from_biguint_checked(&z_value).expect("z fits")),
        )?;
        if limb_index == 0 {
            region.constrain_equal(z_cell.cell(), value_cell.cell())?;
        }
        z_value >>= window;
        offset += 1;
    }
    region.assign_advice_from_constant(
        || "range z terminal",
        config.advice[3],
        offset,
        Fp::from(0),
    )?;
    offset += 1;
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

/// Rows used by one signed-interval range check at the given window:
/// shift constant + add + two running-sum chains (limbs + terminal zero each)
/// + max constant + diff + sum row.
fn signed_check_rows(bits: usize, window: u32) -> usize {
    let limbs = bits.max(1).div_ceil(window as usize);
    2 * (limbs + 1) + 5
}

fn rows_needed(circuit: &LoraCircuit, window: u32) -> usize {
    let value_bits = circuit.fixed_point.value_bits as usize;
    let intermediate_bits = circuit.fixed_point.intermediate_bits as usize;
    let scale_bits = circuit.fixed_point.scale_bits as usize;
    let raw_a_bits = value_bits
        .saturating_mul(2)
        .saturating_add(ceil_log2(circuit.in_dim().max(1)))
        .saturating_add(1);
    let raw_b_bits = value_bits
        .saturating_add(intermediate_bits)
        .saturating_add(ceil_log2(circuit.rank().max(1)))
        .saturating_add(1);
    let scaling_bits = bit_length_i64(circuit.scaling_num).max(1);
    let scaled_bits = intermediate_bits.saturating_add(scaling_bits).max(1);
    let den_bits = bit_length_i64(circuit.scaling_den).max(1);
    let rc = |bits: usize| signed_check_rows(bits, window);
    let matrix_values = circuit.rank() * circuit.in_dim() + circuit.out_dim() * circuit.rank();
    let adapter_words = 11 + matrix_values;
    // div block: raw range check + division row + quotient check + remainder check
    let div_a = rc(raw_a_bits) + 1 + rc(intermediate_bits) + rc(scale_bits.max(1));
    let div_b = rc(raw_b_bits) + 1 + rc(intermediate_bits) + rc(scale_bits.max(1));
    let div_final = rc(scaled_bits) + 1 + rc(value_bits) + rc(den_bits);
    32 + circuit.in_dim() * (1 + rc(value_bits))
        + matrix_values * (3 + rc(value_bits))
        + circuit.rank() * div_a
        + circuit.out_dim() * (div_b + 1 + div_final)
        + adapter_words * POSEIDON_PAIR_ROWS
        + ROW_MARGIN
}

/// Deterministically choose the lookup window and circuit size from the
/// statement shape alone, so prover and verifier always derive the same
/// circuit. Smaller windows shrink the lookup table for tiny circuits while
/// larger windows minimise rows for big ones; we pick the (k, window) pair
/// with the smallest k, preferring larger windows on ties.
fn window_and_k_for(circuit: &LoraCircuit) -> (u32, u32) {
    let mut best: Option<(u32, u32)> = None;
    for window in RANGE_WINDOW_MIN..=RANGE_WINDOW_MAX {
        let rows = rows_needed(circuit, window);
        let table_rows = (1usize << window) + ROW_MARGIN;
        let needed = rows.max(table_rows).next_power_of_two();
        let k = needed.trailing_zeros().max(8);
        let candidate = (k, window);
        best = Some(match best {
            None => candidate,
            Some((best_k, best_w)) => {
                if k < best_k || (k == best_k && window > best_w) {
                    candidate
                } else {
                    (best_k, best_w)
                }
            }
        });
    }
    let (k, window) = best.expect("window range is non-empty");
    (window, k)
}

fn k_for(circuit: &LoraCircuit) -> u32 {
    circuit.window_and_k().1
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
    let k = k_for(&circuit);
    let params = params_for(k);
    let key = CircuitKey::from_circuit(&circuit);
    let pk = proving_key_for(&key, &params, &circuit.without_witnesses())?;
    let instances = public_inputs(&circuit)?;
    let instance_refs: Vec<&[Fp]> = vec![instances.as_slice()];
    let mut transcript = Blake2bWrite::<_, vesta::Affine, Challenge255<_>>::init(vec![]);
    create_proof(
        &params,
        &pk,
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
    let k = k_for(&circuit);
    let params = params_for(k);
    let key = CircuitKey::from_circuit(&circuit);
    let vk = verifying_key_for(&key, &params, &circuit)?;
    let instances = public_inputs(&circuit)?;
    let instance_refs: Vec<&[Fp]> = vec![instances.as_slice()];
    let mut transcript = Blake2bRead::<_, vesta::Affine, Challenge255<_>>::init(proof);
    let result = verify_proof(
        &params,
        &vk,
        SingleVerifier::new(&params),
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

/// Shared validation/bounds context for the exact integer delta fast path.
struct DeltaContext {
    in_dim: usize,
    value_bound: i128,
    intermediate_bound: i128,
    scale: i128,
    scaling_num: i64,
    scaling_den: i64,
}

fn delta_context(
    a: &[Vec<i64>],
    b: &[Vec<i64>],
    scaling_num: i64,
    scaling_den: i64,
    scale_bits: u32,
    value_bits: u32,
    intermediate_bits: u32,
) -> Result<DeltaContext, NativeError> {
    if scaling_den <= 0 {
        return Err(NativeError::InvalidDimensions(
            "scaling_den must be positive".into(),
        ));
    }
    if value_bits == 0 || value_bits > 63 || intermediate_bits == 0 || intermediate_bits > 127 {
        return Err(NativeError::InvalidDimensions(
            "bit widths outside native fast-path range".into(),
        ));
    }
    if scale_bits >= value_bits {
        return Err(NativeError::InvalidDimensions(
            "scale_bits must be less than value_bits".into(),
        ));
    }
    let rank = a.len();
    if rank == 0 {
        return Err(NativeError::InvalidDimensions(
            "rank must be positive".into(),
        ));
    }
    let in_dim = a[0].len();
    for row in a {
        if row.len() != in_dim {
            return Err(NativeError::InvalidDimensions(
                "A row width must match x length".into(),
            ));
        }
    }
    for row in b {
        if row.len() != rank {
            return Err(NativeError::InvalidDimensions(
                "B row width must match A rank".into(),
            ));
        }
    }
    let value_bound = (1i128 << (value_bits - 1)) - 1;
    for value in a.iter().flatten() {
        check_native_bound(*value, value_bound, "A")?;
    }
    for value in b.iter().flatten() {
        check_native_bound(*value, value_bound, "B")?;
    }
    Ok(DeltaContext {
        in_dim,
        value_bound,
        intermediate_bound: (1i128 << (intermediate_bits - 1)) - 1,
        scale: 1i128 << scale_bits,
        scaling_num,
        scaling_den,
    })
}

fn check_native_bound(value: i64, bound: i128, label: &str) -> Result<(), NativeError> {
    let value = value as i128;
    if value < -bound || value > bound {
        return Err(NativeError::InvalidDimensions(format!(
            "{label} value {value} exceeds signed bound +/-{bound}"
        )));
    }
    Ok(())
}

fn delta_for_row(
    ctx: &DeltaContext,
    a: &[Vec<i64>],
    b: &[Vec<i64>],
    x: &[i64],
) -> Result<Vec<i64>, NativeError> {
    if x.len() != ctx.in_dim {
        return Err(NativeError::InvalidDimensions(format!(
            "x expected length {}, got {}",
            ctx.in_dim,
            x.len()
        )));
    }
    for value in x {
        check_native_bound(*value, ctx.value_bound, "x")?;
    }

    let overflow =
        || NativeError::InvalidDimensions("intermediate value exceeds native range".into());
    let div_round = |numerator: i128, denominator: i128| -> i128 {
        // denominator > 0 here; floor((n + d/2) / d), matching div_floor.
        let n = numerator + denominator / 2;
        n.div_euclid(denominator)
    };

    let mut intermediate = Vec::with_capacity(a.len());
    for row in a {
        let mut raw: i128 = 0;
        for (weight, x_i) in row.iter().zip(x.iter()) {
            let product = (*weight as i128)
                .checked_mul(*x_i as i128)
                .ok_or_else(overflow)?;
            raw = raw.checked_add(product).ok_or_else(overflow)?;
        }
        if raw < -ctx.intermediate_bound || raw > ctx.intermediate_bound {
            return Err(NativeError::InvalidDimensions(format!(
                "intermediate value {raw} exceeds signed bound +/-{}",
                ctx.intermediate_bound
            )));
        }
        intermediate.push(div_round(raw, ctx.scale));
    }

    let mut delta = Vec::with_capacity(b.len());
    for row in b {
        let mut raw: i128 = 0;
        for (weight, value) in row.iter().zip(intermediate.iter()) {
            let product = (*weight as i128).checked_mul(*value).ok_or_else(overflow)?;
            raw = raw.checked_add(product).ok_or_else(overflow)?;
        }
        if raw < -ctx.intermediate_bound || raw > ctx.intermediate_bound {
            return Err(NativeError::InvalidDimensions(format!(
                "intermediate value {raw} exceeds signed bound +/-{}",
                ctx.intermediate_bound
            )));
        }
        let rescaled = div_round(raw, ctx.scale);
        let scaled = rescaled
            .checked_mul(ctx.scaling_num as i128)
            .ok_or_else(overflow)?;
        let out = div_round(scaled, ctx.scaling_den as i128);
        if out < -ctx.value_bound || out > ctx.value_bound {
            return Err(NativeError::InvalidDimensions(format!(
                "delta value {out} exceeds signed bound +/-{}",
                ctx.value_bound
            )));
        }
        delta.push(out as i64);
    }
    Ok(delta)
}

/// Exact integer LoRA delta computation mirroring
/// `proof_contract.compute_delta_quantized`. All arithmetic uses checked i128
/// operations; any overflow or bound violation is reported as an error so the
/// Python caller can fall back to its arbitrary-precision implementation. The
/// rounding rule is the same canonical half-up floor division used in-circuit.
#[allow(clippy::too_many_arguments)]
pub fn compute_delta_quantized_native(
    a: &[Vec<i64>],
    b: &[Vec<i64>],
    x: &[i64],
    scaling_num: i64,
    scaling_den: i64,
    scale_bits: u32,
    value_bits: u32,
    intermediate_bits: u32,
) -> Result<Vec<i64>, NativeError> {
    let ctx = delta_context(
        a,
        b,
        scaling_num,
        scaling_den,
        scale_bits,
        value_bits,
        intermediate_bits,
    )?;
    delta_for_row(&ctx, a, b, x)
}

/// Batched variant: validates the adapter once and computes every row's delta
/// in parallel. Semantics per row are identical to the single-row function.
#[allow(clippy::too_many_arguments)]
pub fn compute_delta_rows_native(
    a: &[Vec<i64>],
    b: &[Vec<i64>],
    xs: &[Vec<i64>],
    scaling_num: i64,
    scaling_den: i64,
    scale_bits: u32,
    value_bits: u32,
    intermediate_bits: u32,
) -> Result<Vec<Vec<i64>>, NativeError> {
    let ctx = delta_context(
        a,
        b,
        scaling_num,
        scaling_den,
        scale_bits,
        value_bits,
        intermediate_bits,
    )?;
    xs.par_iter()
        .map(|x| delta_for_row(&ctx, a, b, x))
        .collect()
}

/// Exact fixed-point quantization of one decimal string, replicating
/// `proof_contract.quantize_scalar`: the caller passes Python's `str(float)`
/// output (the semantics anchor for quantization), and this routine computes
/// `round_half_up_away_from_zero(decimal_value * 2^scale_bits)` in exact
/// integer arithmetic. Rust's own float formatter is deliberately NOT used:
/// it can pick a different shortest round-trip digit string than CPython's
/// repr for the same f64, which would silently change quantized values.
pub fn quantize_decimal_str_exact(
    text: &str,
    scale_bits: u32,
    value_bits: u32,
) -> Result<i64, NativeError> {
    if value_bits == 0 || value_bits > 63 || scale_bits >= value_bits {
        return Err(NativeError::InvalidDimensions(
            "fixed-point bits outside native fast-path range".into(),
        ));
    }
    let value_bound = (1i128 << (value_bits - 1)) - 1;

    let bad = || NativeError::InvalidDimensions(format!("cannot quantize value {text:?}"));
    let trimmed = text.trim();
    let (negative, unsigned) = match trimmed.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, trimmed.strip_prefix('+').unwrap_or(trimmed)),
    };
    let (mantissa_text, exponent) = match unsigned.split_once(['e', 'E']) {
        Some((mantissa, exp)) => (mantissa, exp.parse::<i32>().map_err(|_| bad())?),
        None => (unsigned, 0),
    };
    let (int_text, frac_text) = match mantissa_text.split_once('.') {
        Some((int_part, frac_part)) => (int_part, frac_part),
        None => (mantissa_text, ""),
    };
    if int_text.is_empty() && frac_text.is_empty() {
        return Err(bad());
    }
    let digits: String = format!("{int_text}{frac_text}");
    if digits.is_empty() || digits.len() > 30 || !digits.bytes().all(|b| b.is_ascii_digit()) {
        // Non-finite values ('inf'/'nan'), or more digits than the i128 fast
        // path can hold: defer to the Python reference implementation.
        return Err(bad());
    }
    let mantissa: i128 = digits.parse().map_err(|_| bad())?;
    let pow10 = exponent
        .checked_sub(frac_text.len() as i32)
        .ok_or_else(bad)?;

    let bound_err = || {
        NativeError::InvalidDimensions(format!(
            "quantized value of {text:?} exceeds signed bound +/-{value_bound}"
        ))
    };
    let magnitude = if mantissa == 0 {
        0i128
    } else if pow10 >= 0 {
        // Integral decimal value: scale exactly, no rounding needed.
        let mut scaled = mantissa;
        for _ in 0..pow10 {
            scaled = scaled.checked_mul(10).ok_or_else(bound_err)?;
        }
        scaled
            .checked_mul(1i128 << scale_bits)
            .ok_or_else(bound_err)?
    } else {
        let k = -pow10 as u32;
        // numerator = mantissa * 2^scale_bits, denominator = 10^k. With at
        // most 30 mantissa digits the numerator can overflow i128 only via
        // checked_mul (reported as out-of-fast-path); for k beyond i128's
        // 10^38 capacity the quotient is zero when numerator < 10^k / 2.
        let numerator = mantissa
            .checked_mul(1i128 << scale_bits)
            .ok_or_else(bound_err)?;
        if k > 38 {
            // 10^39 / 2 = 5e38 exceeds i128::MAX (~1.7e38), so any in-range
            // numerator is strictly below half the denominator: rounds to 0.
            0
        } else {
            let denominator = 10i128.pow(k);
            (numerator + denominator / 2) / denominator
        }
    };
    let signed = if negative { -magnitude } else { magnitude };
    if signed < -value_bound || signed > value_bound {
        return Err(NativeError::InvalidDimensions(format!(
            "quantized value {signed} exceeds signed bound +/-{value_bound}"
        )));
    }
    Ok(signed as i64)
}

/// Quantize whole rows in parallel with exact scalar semantics.
pub fn quantize_rows_exact(
    rows: &[Vec<String>],
    scale_bits: u32,
    value_bits: u32,
) -> Result<Vec<Vec<i64>>, NativeError> {
    rows.par_iter()
        .map(|row| {
            row.iter()
                .map(|value| quantize_decimal_str_exact(value, scale_bits, value_bits))
                .collect()
        })
        .collect()
}

const MERKLE_EMPTY: [u8; 32] = [0u8; 32];

/// Hiding Merkle root over f64 leaves, byte-identical to
/// `zklora.polynomial_commit._merkle_root` (BLAKE3 leaves salted with the
/// nonce, right-padded with the EMPTY leaf so internal levels stay even).
pub fn merkle_root_f64(values: &[f64], nonce: &[u8]) -> [u8; 32] {
    if values.is_empty() {
        return MERKLE_EMPTY;
    }
    let mut level: Vec<[u8; 32]> = values
        .par_iter()
        .map(|value| {
            let mut hasher = blake3::Hasher::new();
            hasher.update(&value.to_be_bytes());
            hasher.update(nonce);
            *hasher.finalize().as_bytes()
        })
        .collect();
    if level.len() % 2 == 1 {
        level.push(MERKLE_EMPTY);
    }
    while level.len() > 1 {
        let mut next: Vec<[u8; 32]> = level
            .par_chunks_exact(2)
            .map(|pair| {
                let mut hasher = blake3::Hasher::new();
                hasher.update(&pair[0]);
                hasher.update(&pair[1]);
                *hasher.finalize().as_bytes()
            })
            .collect();
        if next.len() % 2 == 1 && next.len() != 1 {
            next.push(MERKLE_EMPTY);
        }
        level = next;
    }
    level[0]
}

#[cfg(feature = "python")]
#[pyo3::pyfunction]
fn prove(
    py: pyo3::Python<'_>,
    statement_json: &str,
    witness_json: &str,
) -> pyo3::PyResult<Vec<u8>> {
    py.detach(|| Ok(prove_bytes(statement_json, witness_json)?))
}

#[cfg(feature = "python")]
#[pyo3::pyfunction]
fn verify(py: pyo3::Python<'_>, statement_json: &str, proof: &[u8]) -> pyo3::PyResult<bool> {
    py.detach(|| Ok(verify_bytes(statement_json, proof)?))
}

#[cfg(feature = "python")]
#[pyo3::pyfunction]
fn statement_digest(statement_json: &str) -> pyo3::PyResult<String> {
    Ok(statement_digest_hex(statement_json))
}

#[cfg(feature = "python")]
#[pyo3::pyfunction]
fn adapter_commitment(py: pyo3::Python<'_>, adapter_json: &str) -> pyo3::PyResult<String> {
    py.detach(|| {
        let input: AdapterCommitmentInput =
            serde_json::from_str(adapter_json).map_err(|e| NativeError::Json(e.to_string()))?;
        Ok(adapter_commitment_for_input(&input)?)
    })
}

#[cfg(feature = "python")]
#[pyo3::pyfunction]
#[allow(clippy::too_many_arguments)]
fn compute_delta_quantized(
    py: pyo3::Python<'_>,
    a: Vec<Vec<i64>>,
    b: Vec<Vec<i64>>,
    x: Vec<i64>,
    scaling_num: i64,
    scaling_den: i64,
    scale_bits: u32,
    value_bits: u32,
    intermediate_bits: u32,
) -> pyo3::PyResult<Vec<i64>> {
    py.detach(|| {
        Ok(compute_delta_quantized_native(
            &a,
            &b,
            &x,
            scaling_num,
            scaling_den,
            scale_bits,
            value_bits,
            intermediate_bits,
        )?)
    })
}

#[cfg(feature = "python")]
#[pyo3::pyfunction]
#[allow(clippy::too_many_arguments)]
fn compute_delta_rows(
    py: pyo3::Python<'_>,
    a: Vec<Vec<i64>>,
    b: Vec<Vec<i64>>,
    xs: Vec<Vec<i64>>,
    scaling_num: i64,
    scaling_den: i64,
    scale_bits: u32,
    value_bits: u32,
    intermediate_bits: u32,
) -> pyo3::PyResult<Vec<Vec<i64>>> {
    py.detach(|| {
        Ok(compute_delta_rows_native(
            &a,
            &b,
            &xs,
            scaling_num,
            scaling_den,
            scale_bits,
            value_bits,
            intermediate_bits,
        )?)
    })
}

#[cfg(feature = "python")]
#[pyo3::pyfunction]
fn quantize_rows(
    py: pyo3::Python<'_>,
    rows: Vec<Vec<String>>,
    scale_bits: u32,
    value_bits: u32,
) -> pyo3::PyResult<Vec<Vec<i64>>> {
    py.detach(|| Ok(quantize_rows_exact(&rows, scale_bits, value_bits)?))
}

#[cfg(feature = "python")]
#[pyo3::pyfunction]
fn merkle_root(py: pyo3::Python<'_>, values: Vec<f64>, nonce: Vec<u8>) -> pyo3::PyResult<Vec<u8>> {
    py.detach(|| Ok(merkle_root_f64(&values, &nonce).to_vec()))
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
    m.add_function(wrap_pyfunction!(compute_delta_quantized, m)?)?;
    m.add_function(wrap_pyfunction!(compute_delta_rows, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_rows, m)?)?;
    m.add_function(wrap_pyfunction!(merkle_root, m)?)?;
    Ok(())
}

#[doc(hidden)]
pub mod bench_support {
    //! Deterministic statement/witness generation for benchmarking only.

    use super::*;

    fn lcg_value(state: &mut u64, magnitude: i64) -> i64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let raw = (*state >> 16) as i64;
        (raw % (2 * magnitude + 1)) - magnitude
    }

    fn div_round(numerator: &BigInt, denominator: &BigInt) -> BigInt {
        let floor_half = denominator / BigInt::from(2u8);
        (numerator + floor_half).div_floor(denominator)
    }

    pub fn bench_statement_and_witness(
        in_dim: usize,
        rank: usize,
        out_dim: usize,
    ) -> (String, String, u32) {
        let fixed_point = FixedPointConfig {
            scale_bits: 20,
            value_bits: 63,
            intermediate_bits: 127,
        };
        let scale = BigInt::one() << fixed_point.scale_bits;
        let scaling_num = 1i64;
        let scaling_den = 1i64;
        let mut state = 0x5eed_5eed_5eed_5eedu64;
        let magnitude = 1i64 << fixed_point.scale_bits;

        let a: Vec<Vec<i64>> = (0..rank)
            .map(|_| {
                (0..in_dim)
                    .map(|_| lcg_value(&mut state, magnitude))
                    .collect()
            })
            .collect();
        let b: Vec<Vec<i64>> = (0..out_dim)
            .map(|_| {
                (0..rank)
                    .map(|_| lcg_value(&mut state, magnitude))
                    .collect()
            })
            .collect();
        let x: Vec<i64> = (0..in_dim)
            .map(|_| lcg_value(&mut state, magnitude))
            .collect();

        let intermediate: Vec<BigInt> = a
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
                    .zip(intermediate.iter())
                    .map(|(w, v)| BigInt::from(*w) * v)
                    .sum();
                let rescaled = div_round(&raw, &scale);
                let scaled = rescaled * BigInt::from(scaling_num);
                let out = div_round(&scaled, &BigInt::from(scaling_den));
                i64::try_from(out).expect("bench delta fits i64")
            })
            .collect();

        let adapter_input = AdapterCommitmentInput {
            schema_version: ARTIFACT_SCHEMA_VERSION,
            in_dim,
            rank,
            out_dim,
            fixed_point: fixed_point.clone(),
            scaling_num,
            scaling_den,
            a: a.clone(),
            b: b.clone(),
        };
        let commitment = adapter_commitment_for_input(&adapter_input).expect("commitment");

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
        let statement_json = serde_json::to_string(&statement).expect("statement json");
        let witness_json = serde_json::to_string(&witness).expect("witness json");
        let circuit = circuit_from_json(&statement_json, &witness_json).expect("circuit");
        let k = k_for(&circuit);
        (statement_json, witness_json, k)
    }

    pub fn prove_verify_once(statement_json: &str, witness_json: &str) -> (f64, f64, usize) {
        let prove_start = std::time::Instant::now();
        let proof = prove_bytes(statement_json, witness_json).expect("prove");
        let prove_ms = prove_start.elapsed().as_secs_f64() * 1000.0;
        let verify_start = std::time::Instant::now();
        assert!(verify_bytes(statement_json, &proof).expect("verify"));
        let verify_ms = verify_start.elapsed().as_secs_f64() * 1000.0;
        (prove_ms, verify_ms, proof.len())
    }
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
    fn window_selection_depends_only_on_shape() {
        let circuit = valid_circuit();
        let (window, k) = circuit.window_and_k();
        assert!((RANGE_WINDOW_MIN..=RANGE_WINDOW_MAX).contains(&window));
        assert!(k >= 8);
        // Witness values must not influence the selected circuit size.
        let mut other = circuit.clone();
        other.a = vec![vec![1, 1]];
        other.b = vec![vec![-1], vec![1]];
        other.x = vec![7, -7];
        other.delta = vec![0, 0];
        assert_eq!(other.window_and_k(), (window, k));
        // The empty circuit used for key generation must agree as well.
        assert_eq!(circuit.without_witnesses().window_and_k(), (window, k));
    }

    #[test]
    fn mock_prover_accepts_multi_limb_range_checks() {
        // value_bits=17 with window selection forces uneven top limbs in the
        // running-sum decomposition; rank 2 exercises both matmul stages.
        let fixed_point = FixedPointConfig {
            scale_bits: 3,
            value_bits: 17,
            intermediate_bits: 40,
        };
        let input = AdapterCommitmentInput {
            schema_version: ARTIFACT_SCHEMA_VERSION,
            in_dim: 5,
            rank: 2,
            out_dim: 3,
            fixed_point: fixed_point.clone(),
            scaling_num: 3,
            scaling_den: 2,
            a: vec![vec![100, -50, 25, -12, 6], vec![-99, 98, -97, 96, -95]],
            b: vec![vec![40, -30], vec![-20, 10], vec![5, -5]],
        };
        let x = vec![64, -32, 16, -8, 4];
        let scale = 1i64 << fixed_point.scale_bits;
        let div_round = |n: i64, d: i64| -> i64 { (n + d / 2).div_euclid(d) };
        let intermediate: Vec<i64> = input
            .a
            .iter()
            .map(|row| {
                let raw: i64 = row.iter().zip(x.iter()).map(|(w, xi)| w * xi).sum();
                div_round(raw, scale)
            })
            .collect();
        let delta: Vec<i64> = input
            .b
            .iter()
            .map(|row| {
                let raw: i64 = row
                    .iter()
                    .zip(intermediate.iter())
                    .map(|(w, v)| w * v)
                    .sum();
                let rescaled = div_round(raw, scale);
                div_round(rescaled * input.scaling_num, input.scaling_den)
            })
            .collect();
        let circuit = LoraCircuit {
            a: input.a.clone(),
            b: input.b.clone(),
            x,
            delta,
            fixed_point,
            scaling_num: input.scaling_num,
            scaling_den: input.scaling_den,
            adapter_commitment: adapter_commitment_for_input(&input).unwrap(),
            statement_digest: "33".repeat(32),
        };
        let instances = public_inputs(&circuit).unwrap();
        let prover = MockProver::run(k_for(&circuit), &circuit, vec![instances]).unwrap();
        assert_eq!(prover.verify(), Ok(()));

        let mut tampered = public_inputs(&circuit).unwrap();
        tampered[circuit.in_dim()] += Fp::from(1);
        let prover = MockProver::run(k_for(&circuit), &circuit, vec![tampered]).unwrap();
        assert!(prover.verify().is_err());
    }

    #[test]
    fn native_delta_helper_matches_circuit_relation() {
        let circuit = valid_circuit();
        let delta = compute_delta_quantized_native(
            &circuit.a,
            &circuit.b,
            &circuit.x,
            circuit.scaling_num,
            circuit.scaling_den,
            circuit.fixed_point.scale_bits,
            circuit.fixed_point.value_bits,
            circuit.fixed_point.intermediate_bits,
        )
        .unwrap();
        assert_eq!(delta, circuit.delta);
    }

    #[test]
    fn merkle_root_handles_padding_rules() {
        let nonce = [7u8; 32];
        assert_eq!(merkle_root_f64(&[], &nonce), MERKLE_EMPTY);
        let single = merkle_root_f64(&[1.5], &nonce);
        let pair = merkle_root_f64(&[1.5, 2.5], &nonce);
        let triple = merkle_root_f64(&[1.5, 2.5, 3.5], &nonce);
        assert_ne!(single, pair);
        assert_ne!(pair, triple);
        // Right-padding with the EMPTY leaf means a singleton tree hashes the
        // leaf against EMPTY rather than returning the leaf itself.
        let mut leaf = blake3::Hasher::new();
        leaf.update(&1.5f64.to_be_bytes());
        leaf.update(&nonce);
        let leaf = *leaf.finalize().as_bytes();
        let mut parent = blake3::Hasher::new();
        parent.update(&leaf);
        parent.update(&MERKLE_EMPTY);
        assert_eq!(single, *parent.finalize().as_bytes());
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
