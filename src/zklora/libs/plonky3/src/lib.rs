use p3_air::{Air, AirBuilder, BaseAir};
use p3_challenger::HashChallenger;
use p3_challenger::SerializingChallenger32;
use p3_circle::CirclePcs;
use p3_commit::ExtensionMmcs;
use p3_field::{extension::BinomialExtensionField, PrimeCharacteristicRing, PrimeField};
use p3_fri::FriParameters;
use p3_keccak::Keccak256Hash;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_mersenne_31::Mersenne31;
use p3_symmetric::CompressionFunctionFromHasher;
use p3_symmetric::SerializingHasher;
use p3_uni_stark::{prove, StarkConfig};

/// Performs matrix multiplication between two matrices of u16 elements.
///
/// # Arguments
/// * `a` - First matrix (m x n)
/// * `b` - Second matrix (n x p)
///
/// # Returns
/// A new matrix (m x p) containing the result of the multiplication
///
/// # Panics
/// Panics if the number of columns in `a` does not match the number of rows in `b`
pub fn matrix_multiply<F: PrimeField>(
    a: &RowMajorMatrix<F>,
    b: &RowMajorMatrix<F>,
) -> RowMajorMatrix<F> {
    assert_eq!(
        a.width(),
        b.height(),
        "Matrix dimensions must be compatible for multiplication"
    );

    let mut result = vec![F::ZERO; a.height() * b.width()];

    for i in 0..a.height() {
        for j in 0..b.width() {
            let mut sum = F::ZERO;
            for k in 0..a.width() {
                sum += a.get(i, k).unwrap() * b.get(k, j).unwrap();
            }
            result[i * b.width() + j] = sum;
        }
    }

    RowMajorMatrix::new(result, b.width())
}

/// Multiplies a vector by a matrix (vector * matrix).
///
/// # Arguments
/// * `a` - A reference to a vector of field elements (length m)
/// * `b` - A reference to a matrix of field elements (m x n)
///
/// # Returns
/// A vector of field elements (length n) representing the result of the multiplication.
///
/// # Panics
/// Panics if the length of the vector does not match the height of the matrix.
pub fn vector_matrix_multiply<F: PrimeField>(a: &Vec<F>, b: &RowMajorMatrix<F>) -> Vec<F> {
    assert_eq!(
        a.len(),
        b.height(),
        "Vector length must match matrix height"
    );
    let mut result = vec![F::ZERO; b.width()];
    for i in 0..b.width() {
        for j in 0..b.height() {
            result[i] += a[j] * b.get(j, i).unwrap();
        }
    }
    result
}

/// AIR (Algebraic Intermediate Representation) for matrix multiplication.
/// This struct represents the configuration and constraints for proving matrix multiplication
/// using an algebraic execution trace. It tracks the dimensions of the input matrices
/// and provides methods for generating and verifying the computation trace.
pub struct VectorMatrixMultiplicationAIR<F: PrimeField> {
    /// Length of vector (number of rows in matrix)
    pub m: usize,
    /// Number of columns in matrix
    pub n: usize,

    pub byte_hash: ByteHash,
    pub field_hash: FieldHash,
    pub compress: MyCompress,
    pub val_mmcs: ValMmcs,
    pub challenge_mmcs: ChallengeMmcs,
    pub config: MyConfig,

    /// Field element type
    _phantom: std::marker::PhantomData<F>,
}

// Field
type Val = Mersenne31;

// This creates a cubic extension field over Val using a binomial basis. It's used for generating challenges in the proof system.
// The reason why we want to extend our field for Challenges, is because the original Field size is too small to be brute-forced to solve the challenge.
type Challenge = BinomialExtensionField<Val, 3>;
// Your choice of Hash Function
type ByteHash = Keccak256Hash;
// A serializer for Hash function, so that it can take Fields as inputs
type FieldHash = SerializingHasher<ByteHash>;
// Defines a compression function type using ByteHash, with 2 input blocks and 32-byte output.
type MyCompress = CompressionFunctionFromHasher<ByteHash, 2, 32>;
// Defines a Merkle tree commitment scheme for field elements with 32 levels.
type ValMmcs = MerkleTreeMmcs<Val, u8, FieldHash, MyCompress, 32>;
// Defines an extension of the Merkle tree commitment scheme for the challenge field.
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
// Defines the challenger type for generating random challenges.
type Challenger = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;
// Defines the polynomial commitment scheme type.
type Pcs = CirclePcs<Val, ValMmcs, ChallengeMmcs>;
// Defines the overall STARK configuration type.
type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;

impl<F: PrimeField> VectorMatrixMultiplicationAIR<F> {
    pub fn new(m: usize, n: usize) -> Self {
        // Declaring an empty hash and its serializer.
        let byte_hash = ByteHash {};
        // Declaring Field hash function, it is used to hash field elements in the proof system
        let field_hash = FieldHash::new(byte_hash);
        // Creates a new instance of the compression function.
        let compress = MyCompress::new(byte_hash);
        // Instantiates the Merkle tree commitment scheme.
        let val_mmcs = ValMmcs::new(field_hash, compress.clone());
        // Creates an instance of the challenge Merkle tree commitment scheme.
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
        // Configures the FRI (Fast Reed-Solomon IOP) protocol parameters.
        let fri_config = FriParameters {
            log_blowup: 1,
            num_queries: 100,
            proof_of_work_bits: 16,
            mmcs: challenge_mmcs.clone(),
            log_final_poly_len: 1,
        };
        // Instantiates the polynomial commitment scheme with the above parameters.
        let pcs = Pcs {
            mmcs: val_mmcs.clone(),
            fri_params: fri_config,
            _phantom: std::marker::PhantomData,
        };
        let challenger = Challenger::from_hasher(vec![], byte_hash);
        // Creates the STARK configuration instance.
        let config = MyConfig::new(pcs, challenger);

        Self {
            m,
            n,
            byte_hash,
            field_hash,
            compress,
            val_mmcs,
            challenge_mmcs,
            config,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Returns the number of columns in the execution trace
    ///
    /// The execution trace for matrix multiplication requires columns to store:
    /// - All elements from vector (m columns)
    /// - All elements from matrix (n * m columns)
    /// - m + (m * n) columns for the selectors
    /// - One column for the running sum of the current element being computed
    /// - One column for the row in the trace that is enabled
    ///
    /// The total width is calculated as: 2 * (m + m * n) + 1
    pub fn trace_width(&self) -> usize {
        2 * (self.m + self.m * self.n) + 1 + 1
    }

    /// Pushes the matrix elements to the trace data with column major order
    fn push_matrix(&self, trace_data: &mut Vec<F>, a: &RowMajorMatrix<F>) {
        for i in 0..self.m {
            for j in 0..self.n {
                trace_data.push(a.get(j, i).unwrap());
            }
        }
    }

    /// Generates a computation trace for matrix multiplication
    ///
    /// # Arguments
    /// * `a` - First matrix (m x n) with field elements
    /// * `b` - Second matrix (n x p) with field elements
    ///
    /// # Returns
    /// A matrix representing the execution trace, where each row is a step in the computation
    /// and columns correspond to:
    /// - Columns 0..(m*n): All elements from matrix A
    /// - Columns (m*n)..(m*n+n*p): All elements from matrix B
    /// - Column trace_width()-4: Row index (i)
    /// - Column trace_width()-3: Column index (j)
    /// - Column trace_width()-2: Current position k in the dot product
    /// - Column trace_width()-1: Current running sum for element C(i,j)
    pub fn generate_trace(&self, v: &Vec<F>, a: &RowMajorMatrix<F>) -> RowMajorMatrix<F> {
        assert_eq!(
            a.height(),
            self.m,
            "Matrix height should match AIR configuration"
        );
        assert_eq!(
            a.width(),
            self.n,
            "Matrix width should match AIR configuration"
        );
        assert_eq!(
            v.len(),
            self.m,
            "Vector length should match AIR configuration"
        );

        // Compute total number of steps needed for the trace
        // For each element V[i], we need m steps to compute the dot product
        let total_rows = self.m * self.n;

        // Initialize the trace matrix with F elements
        let mut trace_data: Vec<F> = Vec::with_capacity(total_rows * self.trace_width());

        let mut vector_selector: Vec<F> = vec![F::ONE]
            .into_iter()
            .chain(std::iter::repeat(F::ZERO).take(self.m - 1))
            .collect();
        let mut matrix_selector: Vec<F> = vec![F::ONE]
            .into_iter()
            .chain(std::iter::repeat(F::ZERO).take(self.m * self.n - 1))
            .collect();

        let mut previous_sum = F::ZERO;

        // Generate the step-by-step trace
        for _ in 0..total_rows {
            trace_data.extend_from_slice(v);
            self.push_matrix(&mut trace_data, a);
            trace_data.extend_from_slice(&vector_selector);
            trace_data.extend_from_slice(&matrix_selector);

            // Find the index in vector_selector where the value is F::ONE
            let vector_index = vector_selector
                .iter()
                .position(|&x| x == F::ONE)
                .expect("vector_selector should contain F::ONE");

            // Get the index in matrix_selector where the value is F::ONE
            let matrix_index = matrix_selector
                .iter()
                .position(|&x| x == F::ONE)
                .expect("matrix_selector should contain F::ONE");

            let running_sum = if vector_index == 1 {
                trace_data[vector_index] * trace_data[self.m + matrix_index] + previous_sum
            } else {
                trace_data[vector_index] * trace_data[self.m + matrix_index]
            };
            trace_data.push(running_sum);
            previous_sum = running_sum.clone();

            trace_data.push(F::ONE);

            vector_selector.rotate_right(1);
            matrix_selector.rotate_right(1);
        }

        // If the trace length is not a power of two, pad it with dummy rows of zeros so that
        // the total number of rows is the next power of two. The last column (running sum)
        // is explicitly set to 0 for these padding rows to indicate that they are inactive.

        let width = self.trace_width();
        let padded_rows = total_rows.next_power_of_two();
        if padded_rows > total_rows {
            for _ in 0..(padded_rows - total_rows) {
                // Push `width` zeros. Since we are inside a PrimeField, F::ZERO is valid.
                // The last column (running sum) remains 0 as well.
                trace_data.extend(std::iter::repeat(F::ZERO).take(width));
            }
        }

        RowMajorMatrix::new(trace_data, width)
    }
}

impl<F: PrimeField> BaseAir<F> for VectorMatrixMultiplicationAIR<F> {
    fn width(&self) -> usize {
        self.trace_width()
    }
}

impl<AB: AirBuilder> Air<AB> for VectorMatrixMultiplicationAIR<AB::F>
where
    AB::F: PrimeField,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let current = main.row_slice(0).unwrap();
        let next = main.row_slice(1).unwrap();

        let v_sel_init = self.n * self.m + self.m;
        let m_sel_init = self.n * self.m + self.m + self.m;
        let matrix_init = self.m;
        let sum = self.trace_width() - 2;
        let enabled = self.trace_width() - 1;

        // Enforce starting state
        // the row is enabled
        builder
            .when_first_row()
            .assert_one(current[enabled].clone());

        // sum equal the first element of the vector times the first element of the matrix
        builder.when_first_row().assert_eq(
            current[sum].clone(),
            current[0].clone() * current[m_sel_init].clone(),
        );

        // The first element of the vector selector is 1
        builder
            .when_first_row()
            .assert_one(current[v_sel_init].clone());
        // The rest of the vector selectors are 0
        for i in 1..self.m {
            builder
                .when_first_row()
                .assert_zero(current[v_sel_init + i].clone());
        }

        // The first element of the matrix selector is 1
        builder
            .when_first_row()
            .assert_one(current[m_sel_init].clone());
        // The rest of the matrix selectors are 0
        for i in 1..self.m {
            builder
                .when_first_row()
                .assert_zero(current[m_sel_init + i].clone());
        }

        // Enforce final enabled row is followed by disabled row
        builder
            .when_transition()
            .when(current[enabled].clone())
            .when(current[sum - 1].clone())
            .assert_zero(next[enabled].clone());

        // Enforce rows are all 0 in the last column after the last enabled row
        builder
            .when_transition()
            .when(AB::Expr::ONE - current[enabled].clone())
            .assert_zero(next[enabled].clone());

        // Enforce booleanity of the vector selector
        for i in 0..self.m {
            builder
                .when_transition()
                .when(current[enabled].clone())
                .assert_bool(current[v_sel_init + i].clone());
        }

        // Enforce booleanity of the matrix selector
        for i in 0..self.m {
            builder
                .when_transition()
                .when(current[enabled].clone())
                .assert_bool(current[m_sel_init + i].clone());
        }

        // Enforce booleanity of the enabled colum
        builder
            .when_transition()
            .assert_bool(current[enabled].clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_field::integers::QuotientMap;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_mersenne_31::Mersenne31; // Import the field implementation from p3-baby-bear

    fn print_trace(trace: &RowMajorMatrix<Mersenne31>) {
        println!("Trace (one row per line):");
        for i in 0..trace.height() {
            let mut row_values = Vec::new();
            for j in 0..trace.width() {
                row_values.push(format!("{}", trace.get(i, j).unwrap()));
            }
            println!("Row {}: [{}]", i, row_values.join(", "));
        }
    }

    #[test]
    fn test_proving() {
        let vector = vec![
            Mersenne31::from_int(1),
            Mersenne31::from_int(2),
            Mersenne31::from_int(3),
        ];
        // [[1, 2], [3, 4]]
        let matrix = RowMajorMatrix::new(
            vec![
                Mersenne31::from_int(1),
                Mersenne31::from_int(2),
                Mersenne31::from_int(3),
                Mersenne31::from_int(4),
                Mersenne31::from_int(5),
                Mersenne31::from_int(6),
                Mersenne31::from_int(7),
                Mersenne31::from_int(8),
                Mersenne31::from_int(9),
            ],
            3,
        );

        let air = VectorMatrixMultiplicationAIR::new(3, 3);
        let trace = air.generate_trace(&vector, &matrix);
        print_trace(&trace);
        let proof = prove(&air.config, &air, trace, &vec![]);
    }

    #[test]
    fn test_trace() {
        let vector = vec![Mersenne31::from_int(1), Mersenne31::from_int(2)];
        // [[1, 2], [3, 4]]
        let matrix = RowMajorMatrix::new(
            vec![
                Mersenne31::from_int(1),
                Mersenne31::from_int(2),
                Mersenne31::from_int(3),
                Mersenne31::from_int(4),
            ],
            2,
        );

        let air = VectorMatrixMultiplicationAIR::new(2, 2);
        let trace = air.generate_trace(&vector, &matrix);
        // Print the trace, one row per line
        print_trace(&trace);

        // Row 0: [1, 2, 1, 3, 2, 4, 1, 0, 1, 0, 0, 0, 1]
        // Row 1: [1, 2, 1, 3, 2, 4, 0, 1, 0, 1, 0, 0, 7]
        // Row 2: [1, 2, 1, 3, 2, 4, 1, 0, 0, 0, 1, 0, 2]
        // Row 3: [1, 2, 1, 3, 2, 4, 0, 1, 0, 0, 0, 1, 10]

        #[rustfmt::skip]
        let correct_trace: RowMajorMatrix<Mersenne31> = RowMajorMatrix::new(
            vec![
                Mersenne31::from_int(1), Mersenne31::from_int(2), Mersenne31::from_int(1), Mersenne31::from_int(3), Mersenne31::from_int(2), Mersenne31::from_int(4), Mersenne31::from_int(1), Mersenne31::from_int(0), Mersenne31::from_int(1), Mersenne31::from_int(0), Mersenne31::from_int(0), Mersenne31::from_int(0), Mersenne31::from_int(1),
                Mersenne31::from_int(1), Mersenne31::from_int(2), Mersenne31::from_int(1), Mersenne31::from_int(3), Mersenne31::from_int(2), Mersenne31::from_int(4), Mersenne31::from_int(0), Mersenne31::from_int(1), Mersenne31::from_int(0), Mersenne31::from_int(1), Mersenne31::from_int(0), Mersenne31::from_int(0), Mersenne31::from_int(7),
                Mersenne31::from_int(1), Mersenne31::from_int(2), Mersenne31::from_int(1), Mersenne31::from_int(3), Mersenne31::from_int(2), Mersenne31::from_int(4), Mersenne31::from_int(1), Mersenne31::from_int(0), Mersenne31::from_int(0), Mersenne31::from_int(0), Mersenne31::from_int(1), Mersenne31::from_int(0), Mersenne31::from_int(2),
                Mersenne31::from_int(1), Mersenne31::from_int(2), Mersenne31::from_int(1), Mersenne31::from_int(3), Mersenne31::from_int(2), Mersenne31::from_int(4), Mersenne31::from_int(0), Mersenne31::from_int(1), Mersenne31::from_int(0), Mersenne31::from_int(0), Mersenne31::from_int(0), Mersenne31::from_int(1), Mersenne31::from_int(10),
            ],
            13,
        );
        assert_eq!(trace, correct_trace);
    }

    #[test]
    fn test_vector_matrix_multiplication() {
        let vector = vec![Mersenne31::from_int(1), Mersenne31::from_int(2)];
        let matrix = RowMajorMatrix::new(
            vec![
                Mersenne31::from_int(1),
                Mersenne31::from_int(2),
                Mersenne31::from_int(3),
                Mersenne31::from_int(4),
            ],
            2,
        );

        let real_result = vec![Mersenne31::from_int(7), Mersenne31::from_int(10)];
        let result = vector_matrix_multiply(&vector, &matrix);
        assert_eq!(result, real_result);
    }

    #[test]
    fn test_matrix_multiplication() {
        // [[1, 2, 3], [4, 5, 6]]
        let matrix_a: RowMajorMatrix<Mersenne31> = RowMajorMatrix::new(
            vec![
                Mersenne31::from_int(1),
                Mersenne31::from_int(2),
                Mersenne31::from_int(3),
                Mersenne31::from_int(4),
                Mersenne31::from_int(5),
                Mersenne31::from_int(6),
            ],
            3,
        );
        // [[1, 2], [3, 4], [5, 6]]
        let matrix_b: RowMajorMatrix<Mersenne31> = RowMajorMatrix::new(
            vec![
                Mersenne31::from_int(1),
                Mersenne31::from_int(2),
                Mersenne31::from_int(3),
                Mersenne31::from_int(4),
                Mersenne31::from_int(5),
                Mersenne31::from_int(6),
            ],
            2,
        );

        let real_result = RowMajorMatrix::new(
            vec![
                Mersenne31::from_int(22),
                Mersenne31::from_int(28),
                Mersenne31::from_int(49),
                Mersenne31::from_int(64),
            ],
            2,
        );
        let result = matrix_multiply::<Mersenne31>(&matrix_a, &matrix_b);
        assert_eq!(result, real_result);
    }
}
