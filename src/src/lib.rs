use halo2_proofs::{
    circuit::{AssignedCell, Layouter, SimpleFloorPlanner, Value},
    pasta::{vesta, EqAffine, Fp},
    plonk::{
        create_proof, keygen_pk, keygen_vk, verify_proof, Advice, Circuit, Column,
        ConstraintSystem, Error, Instance, Selector, SingleVerifier,
    },
    poly::commitment::Params,
    poly::Rotation,
    transcript::{Blake2bRead, Blake2bWrite, Challenge255},
};
use rand_core::OsRng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

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
    pub lora_commitment: i128,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NativeWitness {
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
    lora_commitment: i128,
}

#[derive(Clone, Debug)]
struct LoraConfig {
    advice: [Column<Advice>; 4],
    instance: Column<Instance>,
    mul: Selector,
    add: Selector,
    div_round: Selector,
    boolean: Selector,
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
            lora_commitment: self.lora_commitment,
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
        meta.create_gate("bounded division witness", |meta| {
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

        LoraConfig {
            advice,
            instance,
            mul,
            add,
            div_round,
            boolean,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        self.validate().map_err(|_| Error::Synthesis)?;

        let (x_cells, delta_cells, binding_cells) = layouter.assign_region(
            || "zklora lora delta relation",
            |mut region| {
                let mut offset = 0usize;
                let scale = self.scale()?;
                let zero = region.assign_advice(
                    || "zero",
                    config.advice[0],
                    offset,
                    || Value::known(Fp::from(0)),
                )?;
                offset += 1;
                let scale_cell = region.assign_advice(
                    || "fixed point scale",
                    config.advice[0],
                    offset,
                    || Value::known(fp_from_i128(scale)),
                )?;
                offset += 1;
                let scaling_num_cell = region.assign_advice(
                    || "scaling numerator",
                    config.advice[0],
                    offset,
                    || Value::known(fp_from_i128(i128::from(self.scaling_num))),
                )?;
                offset += 1;
                let scaling_den_cell = region.assign_advice(
                    || "scaling denominator",
                    config.advice[0],
                    offset,
                    || Value::known(fp_from_i128(i128::from(self.scaling_den))),
                )?;
                offset += 1;

                let mut x_cells = Vec::with_capacity(self.x.len());
                let mut commitment_acc = zero.clone();
                let mut commitment_coefficient = 1i128;
                for (i, value) in self.x.iter().enumerate() {
                    let cell = region.assign_advice(
                        || format!("public x {i}"),
                        config.advice[0],
                        offset,
                        || Value::known(fp_from_i128(i128::from(*value))),
                    )?;
                    x_cells.push(cell);
                    offset += 1;
                }

                let mut intermediate = Vec::with_capacity(self.rank());
                for rank_index in 0..self.rank() {
                    let mut acc = zero.clone();
                    let mut raw_value = 0i128;
                    for input_index in 0..self.in_dim() {
                        let weight_value = self.a[rank_index][input_index];
                        let weight = region.assign_advice(
                            || format!("A[{rank_index}][{input_index}]"),
                            config.advice[1],
                            offset,
                            || Value::known(fp_from_i128(i128::from(weight_value))),
                        )?;
                        offset += 1;
                        let (next_commitment_acc, next_offset) = assign_commitment_term(
                            &mut region,
                            &config,
                            &commitment_acc,
                            &weight,
                            commitment_coefficient,
                            offset,
                        )?;
                        commitment_acc = next_commitment_acc;
                        commitment_coefficient += 1;
                        offset = next_offset;
                        let product = assign_mul(
                            &mut region,
                            &config,
                            &x_cells[input_index],
                            &weight,
                            offset,
                        )?;
                        offset += 1;
                        let next_acc = assign_add(&mut region, &config, &acc, &product, offset)?;
                        offset += 1;
                        acc = next_acc;
                        raw_value += i128::from(weight_value) * i128::from(self.x[input_index]);
                    }
                    let q = div_round_away_from_zero(raw_value, scale)?;
                    let (intermediate_cell, next_offset) = assign_div_round(
                        &mut region,
                        &config,
                        &acc,
                        raw_value,
                        q,
                        scale,
                        &scale_cell,
                        offset,
                    )?;
                    offset = next_offset;
                    intermediate.push((intermediate_cell, q));
                }

                let mut delta_cells = Vec::with_capacity(self.out_dim());
                for out_index in 0..self.out_dim() {
                    let mut acc = zero.clone();
                    let mut raw_value = 0i128;
                    for rank_index in 0..self.rank() {
                        let weight_value = self.b[out_index][rank_index];
                        let weight = region.assign_advice(
                            || format!("B[{out_index}][{rank_index}]"),
                            config.advice[1],
                            offset,
                            || Value::known(fp_from_i128(i128::from(weight_value))),
                        )?;
                        offset += 1;
                        let (next_commitment_acc, next_offset) = assign_commitment_term(
                            &mut region,
                            &config,
                            &commitment_acc,
                            &weight,
                            commitment_coefficient,
                            offset,
                        )?;
                        commitment_acc = next_commitment_acc;
                        commitment_coefficient += 1;
                        offset = next_offset;
                        let product = assign_mul(
                            &mut region,
                            &config,
                            &intermediate[rank_index].0,
                            &weight,
                            offset,
                        )?;
                        offset += 1;
                        let next_acc = assign_add(&mut region, &config, &acc, &product, offset)?;
                        offset += 1;
                        acc = next_acc;
                        raw_value += i128::from(weight_value) * intermediate[rank_index].1;
                    }
                    let rescaled = div_round_away_from_zero(raw_value, scale)?;
                    let (rescaled_cell, next_offset) = assign_div_round(
                        &mut region,
                        &config,
                        &acc,
                        raw_value,
                        rescaled,
                        scale,
                        &scale_cell,
                        offset,
                    )?;
                    offset = next_offset;

                    let scaled_raw = rescaled * i128::from(self.scaling_num);
                    let scaled_raw_cell = assign_mul(
                        &mut region,
                        &config,
                        &rescaled_cell,
                        &scaling_num_cell,
                        offset,
                    )?;
                    offset += 1;

                    let final_delta =
                        div_round_away_from_zero(scaled_raw, i128::from(self.scaling_den))?;
                    let (final_cell, next_offset) = assign_div_round(
                        &mut region,
                        &config,
                        &scaled_raw_cell,
                        scaled_raw,
                        final_delta,
                        i128::from(self.scaling_den),
                        &scaling_den_cell,
                        offset,
                    )?;
                    offset = next_offset;
                    delta_cells.push(final_cell);
                }
                Ok((
                    x_cells,
                    delta_cells,
                    vec![
                        scale_cell,
                        scaling_num_cell,
                        scaling_den_cell,
                        commitment_acc,
                    ],
                ))
            },
        )?;

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

    fn scale(&self) -> Result<i128, Error> {
        if self.fixed_point.scale_bits >= 62 {
            return Err(Error::Synthesis);
        }
        Ok(1i128 << self.fixed_point.scale_bits)
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
        if self.fixed_point.scale_bits >= 62 {
            return Err(NativeError::InvalidDimensions(
                "scale_bits must be less than 62 for the v1 native circuit".to_string(),
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

fn assign_commitment_term(
    region: &mut halo2_proofs::circuit::Region<'_, Fp>,
    config: &LoraConfig,
    acc: &AssignedCell<Fp, Fp>,
    value: &AssignedCell<Fp, Fp>,
    coefficient: i128,
    offset: usize,
) -> Result<(AssignedCell<Fp, Fp>, usize), Error> {
    let coefficient_cell = region.assign_advice(
        || "commitment coefficient",
        config.advice[1],
        offset,
        || Value::known(fp_from_i128(coefficient)),
    )?;
    let product = assign_mul(region, config, value, &coefficient_cell, offset + 1)?;
    let next_acc = assign_add(region, config, acc, &product, offset + 2)?;
    Ok((next_acc, offset + 3))
}

fn assign_div_round(
    region: &mut halo2_proofs::circuit::Region<'_, Fp>,
    config: &LoraConfig,
    raw: &AssignedCell<Fp, Fp>,
    raw_value: i128,
    quotient: i128,
    denominator: i128,
    denominator_cell: &AssignedCell<Fp, Fp>,
    offset: usize,
) -> Result<(AssignedCell<Fp, Fp>, usize), Error> {
    if denominator <= 0 {
        return Err(Error::Synthesis);
    }
    let remainder = raw_value - quotient * denominator;
    let mut next_offset = offset;
    config.div_round.enable(region, offset)?;
    raw.copy_advice(|| "division raw", region, config.advice[0], offset)?;
    let quotient_cell = region.assign_advice(
        || "division quotient",
        config.advice[1],
        offset,
        || Value::known(fp_from_i128(quotient)),
    )?;
    denominator_cell.copy_advice(|| "division denominator", region, config.advice[2], offset)?;
    let remainder_cell = region.assign_advice(
        || "division remainder",
        config.advice[3],
        offset,
        || Value::known(fp_from_i128(remainder)),
    )?;
    next_offset += 1;
    next_offset = range_check_signed_bound(
        region,
        config,
        &remainder_cell,
        remainder,
        denominator / 2,
        next_offset,
    )?;
    Ok((quotient_cell, next_offset))
}

fn range_check_signed_bound(
    region: &mut halo2_proofs::circuit::Region<'_, Fp>,
    config: &LoraConfig,
    value_cell: &AssignedCell<Fp, Fp>,
    value: i128,
    bound: i128,
    mut offset: usize,
) -> Result<usize, Error> {
    if bound < 0 {
        return Err(Error::Synthesis);
    }
    if bound == 0 {
        let zero = region.assign_advice(
            || "zero bound",
            config.advice[1],
            offset,
            || Value::known(Fp::from(0)),
        )?;
        region.constrain_equal(value_cell.cell(), zero.cell())?;
        return Ok(offset + 1);
    }
    if value < -bound || value > bound {
        return Err(Error::Synthesis);
    }

    let shifted = (value + bound) as u128;
    let max = (2 * bound) as u128;
    let diff = max.checked_sub(shifted).ok_or(Error::Synthesis)?;
    let bits = bit_length(max);

    let bound_cell = region.assign_advice(
        || "range bound",
        config.advice[1],
        offset,
        || Value::known(fp_from_i128(bound)),
    )?;
    offset += 1;
    let shifted_cell = assign_add(region, config, value_cell, &bound_cell, offset)?;
    offset += 1;
    offset = range_check_unsigned(region, config, &shifted_cell, shifted, bits, offset)?;

    let max_cell = region.assign_advice(
        || "range max",
        config.advice[1],
        offset,
        || Value::known(fp_from_u128(max)),
    )?;
    offset += 1;
    let diff_cell = region.assign_advice(
        || "range diff",
        config.advice[1],
        offset,
        || Value::known(fp_from_u128(diff)),
    )?;
    offset += 1;
    let sum = assign_add(region, config, &shifted_cell, &diff_cell, offset)?;
    offset += 1;
    region.constrain_equal(sum.cell(), max_cell.cell())?;
    range_check_unsigned(region, config, &diff_cell, diff, bits, offset)
}

fn range_check_unsigned(
    region: &mut halo2_proofs::circuit::Region<'_, Fp>,
    config: &LoraConfig,
    value_cell: &AssignedCell<Fp, Fp>,
    value: u128,
    bits: usize,
    mut offset: usize,
) -> Result<usize, Error> {
    if bits == 0 || bits > 127 {
        return Err(Error::Synthesis);
    }
    let mut acc = region.assign_advice(
        || "range accumulator",
        config.advice[0],
        offset,
        || Value::known(Fp::from(0)),
    )?;
    offset += 1;
    for bit_index in (0..bits).rev() {
        let bit_value = ((value >> bit_index) & 1) as u64;
        config.boolean.enable(region, offset)?;
        let bit = region.assign_advice(
            || "range bit",
            config.advice[0],
            offset,
            || Value::known(Fp::from(bit_value)),
        )?;
        offset += 1;
        let two = region.assign_advice(
            || "range two",
            config.advice[1],
            offset,
            || Value::known(Fp::from(2)),
        )?;
        offset += 1;
        let doubled = assign_mul(region, config, &acc, &two, offset)?;
        offset += 1;
        acc = assign_add(region, config, &doubled, &bit, offset)?;
        offset += 1;
    }
    region.constrain_equal(acc.cell(), value_cell.cell())?;
    Ok(offset)
}

fn bit_length(value: u128) -> usize {
    (u128::BITS - value.leading_zeros()) as usize
}

fn fp_from_u128(value: u128) -> Fp {
    let mask = (1u128 << 32) - 1;
    let mut cursor = value;
    let mut base = Fp::from(1);
    let limb_base = Fp::from(1u64 << 32);
    let mut out = Fp::from(0);
    while cursor > 0 {
        out += base * Fp::from((cursor & mask) as u64);
        cursor >>= 32;
        base *= limb_base;
    }
    out
}

fn fp_from_i128(value: i128) -> Fp {
    if value >= 0 {
        fp_from_u128(value as u128)
    } else {
        -fp_from_u128(value.unsigned_abs())
    }
}

fn public_inputs(circuit: &LoraCircuit) -> Vec<Fp> {
    let mut inputs: Vec<Fp> = circuit
        .x
        .iter()
        .chain(circuit.delta.iter())
        .map(|value| fp_from_i128(i128::from(*value)))
        .collect();
    let scale = 1i128 << circuit.fixed_point.scale_bits;
    inputs.push(fp_from_i128(scale));
    inputs.push(fp_from_i128(i128::from(circuit.scaling_num)));
    inputs.push(fp_from_i128(i128::from(circuit.scaling_den)));
    inputs.push(fp_from_i128(circuit.lora_commitment));
    inputs
}

fn rows_needed(circuit: &LoraCircuit) -> usize {
    let scale_bound_bits = circuit.fixed_point.scale_bits.saturating_add(1) as usize;
    let scaling_bound_bits = bit_length(circuit.scaling_den.unsigned_abs() as u128);
    let range_rows = |bits: usize| 8 * bits.max(1) + 8;
    let div_rows = circuit.rank() * range_rows(scale_bound_bits)
        + circuit.out_dim() * range_rows(scale_bound_bits)
        + circuit.out_dim() * range_rows(scaling_bound_bits);
    let commitment_rows =
        3 * (circuit.rank() * circuit.in_dim() + circuit.out_dim() * circuit.rank());
    5 + circuit.in_dim()
        + 3 * circuit.rank() * circuit.in_dim()
        + circuit.rank()
        + 3 * circuit.out_dim() * circuit.rank()
        + 3 * circuit.out_dim()
        + div_rows
        + commitment_rows
}

fn k_for(circuit: &LoraCircuit) -> u32 {
    let rows = rows_needed(circuit).next_power_of_two();
    rows.trailing_zeros().max(6)
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
        lora_commitment: statement.lora_commitment,
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

fn div_round_away_from_zero(numerator: i128, denominator: i128) -> Result<i128, Error> {
    if denominator <= 0 {
        return Err(Error::Synthesis);
    }
    let sign = if numerator < 0 { -1 } else { 1 };
    let magnitude = numerator.abs();
    let mut quotient = magnitude / denominator;
    let remainder = magnitude % denominator;
    if remainder * 2 >= denominator {
        quotient += 1;
    }
    Ok(quotient * sign)
}

pub fn prove_bytes(statement_json: &str, witness_json: &str) -> Result<Vec<u8>, NativeError> {
    let circuit = circuit_from_json(statement_json, witness_json)?;
    let k = k_for(&circuit);
    let params: Params<EqAffine> = Params::new(k);
    let vk = keygen_vk(&params, &circuit).map_err(|e| NativeError::Halo2(e.to_string()))?;
    let pk = keygen_pk(&params, vk, &circuit).map_err(|e| NativeError::Halo2(e.to_string()))?;
    let instances = public_inputs(&circuit);
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
    let params: Params<EqAffine> = Params::new(k);
    let vk = keygen_vk(&params, &circuit).map_err(|e| NativeError::Halo2(e.to_string()))?;
    let instances = public_inputs(&circuit);
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
#[pyo3::pymodule]
fn _native_prover(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::types::PyModuleMethods;
    use pyo3::wrap_pyfunction;

    m.add_function(wrap_pyfunction!(prove, m)?)?;
    m.add_function(wrap_pyfunction!(verify, m)?)?;
    m.add_function(wrap_pyfunction!(statement_digest, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::dev::MockProver;

    fn valid_circuit() -> LoraCircuit {
        LoraCircuit {
            a: vec![vec![2, -1]],
            b: vec![vec![3], vec![-2]],
            x: vec![4, 5],
            delta: vec![9, -6],
            fixed_point: FixedPointConfig {
                scale_bits: 0,
                value_bits: 32,
                intermediate_bits: 64,
            },
            scaling_num: 1,
            scaling_den: 1,
            lora_commitment: 1,
        }
    }

    #[test]
    fn mock_prover_accepts_valid_lora_relation() {
        let circuit = valid_circuit();
        let instances = public_inputs(&circuit);
        let prover = MockProver::run(k_for(&circuit), &circuit, vec![instances]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }

    #[test]
    fn mock_prover_rejects_tampered_delta() {
        let mut circuit = valid_circuit();
        let mut instances = public_inputs(&circuit);
        instances[circuit.in_dim()] += Fp::from(1);
        let prover = MockProver::run(k_for(&circuit), &circuit, vec![instances]).unwrap();
        assert!(prover.verify().is_err());

        circuit.delta[0] += 1;
        let instances = public_inputs(&circuit);
        let prover = MockProver::run(k_for(&circuit), &circuit, vec![instances]).unwrap();
        assert!(prover.verify().is_err());
    }

    #[test]
    fn real_proof_verifies_for_tiny_relation() {
        let statement = NativeStatement {
            x: vec![4, 5],
            delta: vec![9, -6],
            fixed_point: FixedPointConfig {
                scale_bits: 0,
                value_bits: 32,
                intermediate_bits: 64,
            },
            rank: 1,
            scaling_num: 1,
            scaling_den: 1,
            lora_commitment: 1,
        };
        let witness = NativeWitness {
            a: vec![vec![2, -1]],
            b: vec![vec![3], vec![-2]],
        };
        let statement_json = serde_json::to_string(&statement).unwrap();
        let witness_json = serde_json::to_string(&witness).unwrap();
        let proof = prove_bytes(&statement_json, &witness_json).unwrap();
        assert!(verify_bytes(&statement_json, &proof).unwrap());
    }

    #[test]
    fn real_proof_verifies_fixed_point_scaling_relation() {
        let statement = NativeStatement {
            x: vec![6, -2],
            delta: vec![21, -14],
            fixed_point: FixedPointConfig {
                scale_bits: 2,
                value_bits: 32,
                intermediate_bits: 64,
            },
            rank: 1,
            scaling_num: 1,
            scaling_den: 2,
            lora_commitment: 4,
        };
        let witness = NativeWitness {
            a: vec![vec![8, -4]],
            b: vec![vec![12], vec![-8]],
        };
        let statement_json = serde_json::to_string(&statement).unwrap();
        let witness_json = serde_json::to_string(&witness).unwrap();
        let proof = prove_bytes(&statement_json, &witness_json).unwrap();
        assert!(verify_bytes(&statement_json, &proof).unwrap());

        let mut tampered = statement;
        tampered.scaling_den = 3;
        let tampered_json = serde_json::to_string(&tampered).unwrap();
        assert!(!verify_bytes(&tampered_json, &proof).unwrap());

        let mut tampered_commitment = tampered;
        tampered_commitment.scaling_den = 2;
        tampered_commitment.lora_commitment = 5;
        let tampered_commitment_json = serde_json::to_string(&tampered_commitment).unwrap();
        assert!(!verify_bytes(&tampered_commitment_json, &proof).unwrap());
    }
}
