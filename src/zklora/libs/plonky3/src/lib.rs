use p3_air::{Air, AirBuilder, BaseAir};
use p3_baby_bear::BabyBear;
use p3_challenger::HashChallenger;
use p3_challenger::SerializingChallenger32;
use p3_circle::CirclePcs;
use p3_commit::ExtensionMmcs;
use p3_field::{extension::BinomialExtensionField, AbstractField, PrimeField};
use p3_fri::FriConfig;
use p3_keccak::Keccak256Hash;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_symmetric::CompressionFunctionFromHasher;
use p3_symmetric::SerializingHasher32;
use p3_uni_stark::StarkConfig;

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

    let mut result = vec![F::zero(); a.height() * b.width()];

    for i in 0..a.height() {
        for j in 0..b.width() {
            let mut sum = F::zero();
            for k in 0..a.width() {
                sum += a.get(i, k) * b.get(k, j);
            }
            result[i * b.width() + j] = sum;
        }
    }

    RowMajorMatrix::new(result, b.width())
}

/// AIR (Algebraic Intermediate Representation) for matrix multiplication.
/// This struct represents the configuration and constraints for proving matrix multiplication
/// using an algebraic execution trace. It tracks the dimensions of the input matrices
/// and provides methods for generating and verifying the computation trace.
pub struct MatrixMultiplicationAIR<F: PrimeField> {
    /// Number of rows in matrix A
    pub m: usize,
    /// Number of columns in matrix A (and rows in matrix B)
    pub n: usize,
    /// Number of columns in matrix B
    pub p: usize,

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
type Val = BabyBear;

// This creates a cubic extension field over Val using a binomial basis. It's used for generating challenges in the proof system.
// The reason why we want to extend our field for Challenges, is because the original Field size is too small to be brute-forced to solve the challenge.
type Challenge = BinomialExtensionField<Val, 3>;
// Your choice of Hash Function
type ByteHash = Keccak256Hash;
// A serializer for Hash function, so that it can take Fields as inputs
type FieldHash = SerializingHasher32<ByteHash>;
// Defines a compression function type using ByteHash, with 2 input blocks and 32-byte output.
type MyCompress = CompressionFunctionFromHasher<u8, ByteHash, 2, 32>;
// Defines a Merkle tree commitment scheme for field elements with 32 levels.
type ValMmcs = FieldMerkleTreeMmcs<Val, u8, FieldHash, MyCompress, 32>;
// Defines an extension of the Merkle tree commitment scheme for the challenge field.
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
// Defines the challenger type for generating random challenges.
type Challenger = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;
// Defines the polynomial commitment scheme type.
type Pcs = CirclePcs<Val, ValMmcs, ChallengeMmcs>;
// Defines the overall STARK configuration type.
type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;

impl<F: PrimeField> MatrixMultiplicationAIR<F> {
    pub fn new(m: usize, n: usize, p: usize) -> Self {
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
        let fri_config = FriConfig {
            log_blowup: 1,
            num_queries: 100,
            proof_of_work_bits: 16,
            mmcs: challenge_mmcs.clone(),
        };
        // Instantiates the polynomial commitment scheme with the above parameters.
        let pcs = Pcs {
            mmcs: val_mmcs.clone(),
            fri_config,
            _phantom: std::marker::PhantomData,
        };
        // Creates the STARK configuration instance.
        let config = MyConfig::new(pcs);

        Self {
            m,
            n,
            p,
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
    /// - All elements from matrix A (m * n columns)
    /// - All elements from matrix B (n * p columns)
    /// - Three index variables (i, j, k) for tracking the current position in the computation
    /// - One column for the running sum of the current element being computed
    ///
    /// The total width is calculated as: (m * n) + (n * p) + 3 + 1
    pub fn trace_width(&self) -> usize {
        (self.m * self.n) + (self.n * self.p) + 3 + 1
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
    pub fn generate_trace(
        &self,
        a: &RowMajorMatrix<F>,
        b: &RowMajorMatrix<F>,
    ) -> RowMajorMatrix<F> {
        assert_eq!(
            a.height(),
            self.m,
            "Matrix A height should match AIR configuration"
        );
        assert_eq!(
            a.width(),
            self.n,
            "Matrix A width should match AIR configuration"
        );
        assert_eq!(
            b.height(),
            self.n,
            "Matrix B height should match AIR configuration"
        );
        assert_eq!(
            b.width(),
            self.p,
            "Matrix B width should match AIR configuration"
        );

        // Compute total number of steps needed for the trace
        // For each element C[i,j], we need n steps to compute the dot product
        let total_rows = self.m * self.p * self.n;

        // Calculate the new trace width:
        // - Matrix A elements (m*n columns)
        // - Matrix B elements (n*p columns)
        // - Indices i, j, k (3 columns)
        // - Running sum (1 column)

        // Initialize the trace matrix with F elements
        let mut trace_data: Vec<F> = Vec::with_capacity(total_rows * self.trace_width());

        // Generate the step-by-step trace
        for i in 0..self.m {
            for j in 0..self.p {
                let mut running_sum = F::zero();

                for k in 0..self.n {
                    // First, add all elements from matrix A
                    for a_row in 0..self.m {
                        for a_col in 0..self.n {
                            trace_data.push(a.get(a_row, a_col));
                        }
                    }

                    // Then, add all elements from matrix B
                    for b_row in 0..self.n {
                        for b_col in 0..self.p {
                            trace_data.push(b.get(b_row, b_col));
                        }
                    }

                    // Add indices i, j, k
                    trace_data.push(F::from_canonical_u64(i as u64));
                    trace_data.push(F::from_canonical_u64(j as u64));
                    trace_data.push(F::from_canonical_u64(k as u64));

                    // Update running sum
                    running_sum += a.get(i, k) * b.get(k, j);

                    // Add the running sum
                    trace_data.push(running_sum);
                }
            }
        }

        RowMajorMatrix::new(trace_data, self.trace_width())
    }
}

impl<F: PrimeField> BaseAir<F> for MatrixMultiplicationAIR<F> {
    fn width(&self) -> usize {
        self.trace_width()
    }
}

impl<AB: AirBuilder> Air<AB> for MatrixMultiplicationAIR<AB::F>
where
    AB::F: PrimeField,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let current = main.row_slice(0);
        let next = main.row_slice(1);

        let i = self.trace_width() - 4;
        let j = self.trace_width() - 3;
        let k = self.trace_width() - 2;
        let sum = self.trace_width() - 1;

        // Enforce starting state
        // Assert that the sum column of the first element of A and B multiplied
        builder
            .when_first_row()
            .assert_eq(current[sum], current[0] * current[self.m * self.n]);

        // Assert that the row index is 0
        builder
            .when_first_row()
            .assert_eq(current[i], AB::Expr::zero());

        // Assert that the column index is 0
        builder
            .when_first_row()
            .assert_eq(current[j], AB::Expr::zero());

        // Assert that the current position is 0
        builder
            .when_first_row()
            .assert_eq(current[k], AB::Expr::zero());

        // Enforce state transition
        // Assert that the matrices don't change between rows
        for idx in 0..(self.m * self.n + self.n * self.p) {
            builder.when_transition().assert_eq(next[idx], current[idx]);
        }

        //Assert that the multiplication step is correct
        for row_idx in 0..self.m {
            for column_idx in 0..self.p {
                for idx in 0..self.n {
                    // The running sum is applied after the first index in the inner product
                    // of the row of matrix A and the column of matrix B
                    if idx != 0 {
                        builder.when_transition().assert_zero(
                            (current[i] - AB::Expr::from_canonical_usize(row_idx))
                                + (current[j] - AB::Expr::from_canonical_usize(column_idx))
                                + (current[k] - AB::Expr::from_canonical_usize(idx))
                                + (next[sum]
                                    - (current[sum]
                                        + current[row_idx * self.n + idx]
                                            * current
                                                [self.m * self.n + idx * self.p + column_idx])),
                        );
                    // The first number for the running sum is the multiplication of the
                    // first element in row row_idx and the first element in column column_idx
                    } else {
                        builder.when_transition().assert_zero(
                            (current[i] - AB::Expr::from_canonical_usize(row_idx))
                                + (current[j] - AB::Expr::from_canonical_usize(column_idx))
                                + (current[k] - AB::Expr::from_canonical_usize(idx))
                                + (current[sum]
                                    - current[row_idx * self.n + idx]
                                        * current[self.m * self.n + idx * self.p + column_idx]),
                        );
                    }
                }
            }
        }
        //         0  1  2  3  4  5  6  7  8  9 10 11
        // Row 0: [1, 2, 3, 4, 1, 2, 3, 4, 0, 0, 0, 1]
        // Row 1: [1, 2, 3, 4, 1, 2, 3, 4, 0, 0, 1, 7]
        // Row 2: [1, 2, 3, 4, 1, 2, 3, 4, 0, 1, 0, 2]
        // Row 3: [1, 2, 3, 4, 1, 2, 3, 4, 0, 1, 1, 10]
        // Row 4: [1, 2, 3, 4, 1, 2, 3, 4, 1, 0, 0, 3]
        // Row 5: [1, 2, 3, 4, 1, 2, 3, 4, 1, 0, 1, 15]
        // Row 6: [1, 2, 3, 4, 1, 2, 3, 4, 1, 1, 0, 6]
        // Row 7: [1, 2, 3, 4, 1, 2, 3, 4, 1, 1, 1, 22]

        // Assert that if k is not the last index, then i,j stays the same and k is incremented
        builder
            .when_transition()
            .when_ne(current[k], AB::Expr::from_canonical_usize(self.n - 1))
            .assert_zero(
                (next[i] - current[i])
                    + (next[j] - current[j])
                    + (next[k] - current[k] - AB::Expr::one()),
            );

        // Assert that if j is not the last element and k is the last element, then i stays the same, j increments by one, and k is resets to 0
        builder
            .when_transition()
            .when_ne(current[j], AB::Expr::from_canonical_usize(self.p - 1))
            .assert_zero(
                (current[k] - AB::Expr::from_canonical_usize(self.n - 1))
                    + (next[i] - current[i])
                    + (next[j] - current[j] - AB::Expr::one())
                    + (next[k] - AB::Expr::zero()),
            );

        // Assert that if i is not the last element and j is the last element and k is the last element, then i increments by one, j resets to 0, and k resets to 0
        builder
            .when_transition()
            .when_ne(current[i], AB::Expr::from_canonical_usize(self.m - 1))
            .assert_zero(
                (current[j] - AB::Expr::from_canonical_usize(self.p - 1))
                    + (current[k] - AB::Expr::from_canonical_usize(self.n - 1))
                    + (next[i] - current[i] - AB::Expr::one())
                    + (next[j] - AB::Expr::zero())
                    + (next[k] - AB::Expr::zero()),
            );

        // Enforce finale state
        // Assert that columns i,j,k are all 1
        builder
            .when_last_row()
            .assert_eq(current[i], AB::Expr::one());
        builder
            .when_last_row()
            .assert_eq(current[j], AB::Expr::one());
        builder
            .when_last_row()
            .assert_eq(current[k], AB::Expr::one());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_baby_bear::BabyBear; // Import the field implementation from p3-baby-bear
    use p3_matrix::dense::RowMajorMatrix;

    #[test]
    fn test_trace() {
        // [[1, 2], [3, 4]]
        let matrix_a: RowMajorMatrix<BabyBear> = RowMajorMatrix::new(
            vec![
                BabyBear::new(1),
                BabyBear::new(2),
                BabyBear::new(3),
                BabyBear::new(4),
            ],
            2,
        );
        // [[1, 2], [3, 4]]
        let matrix_b: RowMajorMatrix<BabyBear> = RowMajorMatrix::new(
            vec![
                BabyBear::new(1),
                BabyBear::new(2),
                BabyBear::new(3),
                BabyBear::new(4),
            ],
            2,
        );

        let air = MatrixMultiplicationAIR::new(2, 2, 2);
        let trace = air.generate_trace(&matrix_a, &matrix_b);
        // Print the trace, one row per line
        println!("Trace (one row per line):");
        for i in 0..trace.height() {
            let mut row_values = Vec::new();
            for j in 0..trace.width() {
                row_values.push(format!("{}", trace.get(i, j)));
            }
            println!("Row {}: [{}]", i, row_values.join(", "));
        }
        // Row 0: [1, 2, 3, 4, 1, 2, 3, 4, 0, 0, 0, 1]
        // Row 1: [1, 2, 3, 4, 1, 2, 3, 4, 0, 0, 1, 7]
        // Row 2: [1, 2, 3, 4, 1, 2, 3, 4, 0, 1, 0, 2]
        // Row 3: [1, 2, 3, 4, 1, 2, 3, 4, 0, 1, 1, 10]
        // Row 4: [1, 2, 3, 4, 1, 2, 3, 4, 1, 0, 0, 3]
        // Row 5: [1, 2, 3, 4, 1, 2, 3, 4, 1, 0, 1, 15]
        // Row 6: [1, 2, 3, 4, 1, 2, 3, 4, 1, 1, 0, 6]
        // Row 7: [1, 2, 3, 4, 1, 2, 3, 4, 1, 1, 1, 22]
        #[rustfmt::skip]
        let correct_trace: RowMajorMatrix<BabyBear> = RowMajorMatrix::new(
            vec![
                // Row 0
                BabyBear::new(1), BabyBear::new(2), BabyBear::new(3), BabyBear::new(4), // Matrix A
                BabyBear::new(1), BabyBear::new(2), BabyBear::new(3), BabyBear::new(4), // Matrix B
                BabyBear::new(0), BabyBear::new(0), BabyBear::new(0), BabyBear::new(1), // i,j,k,sum
                // Row 1
                BabyBear::new(1), BabyBear::new(2), BabyBear::new(3), BabyBear::new(4), // Matrix A
                BabyBear::new(1), BabyBear::new(2), BabyBear::new(3), BabyBear::new(4), // Matrix B
                BabyBear::new(0), BabyBear::new(0), BabyBear::new(1), BabyBear::new(7), // i,j,k,sum
                // Row 2
                BabyBear::new(1), BabyBear::new(2), BabyBear::new(3), BabyBear::new(4), // Matrix A
                BabyBear::new(1), BabyBear::new(2), BabyBear::new(3), BabyBear::new(4), // Matrix B
                BabyBear::new(0), BabyBear::new(1), BabyBear::new(0), BabyBear::new(2), // i,j,k,sum
                // Row 3
                BabyBear::new(1), BabyBear::new(2), BabyBear::new(3), BabyBear::new(4), // Matrix A
                BabyBear::new(1), BabyBear::new(2), BabyBear::new(3), BabyBear::new(4), // Matrix B
                BabyBear::new(0), BabyBear::new(1), BabyBear::new(1), BabyBear::new(10), // i,j,k,sum
                // Row 4
                BabyBear::new(1), BabyBear::new(2), BabyBear::new(3), BabyBear::new(4), // Matrix A
                BabyBear::new(1), BabyBear::new(2), BabyBear::new(3), BabyBear::new(4), // Matrix B
                BabyBear::new(1), BabyBear::new(0), BabyBear::new(0), BabyBear::new(3), // i,j,k,sum
                // Row 5
                BabyBear::new(1), BabyBear::new(2), BabyBear::new(3), BabyBear::new(4), // Matrix A
                BabyBear::new(1), BabyBear::new(2), BabyBear::new(3), BabyBear::new(4), // Matrix B
                BabyBear::new(1), BabyBear::new(0), BabyBear::new(1), BabyBear::new(15), // i,j,k,sum
                // Row 6
                BabyBear::new(1), BabyBear::new(2), BabyBear::new(3), BabyBear::new(4), // Matrix A
                BabyBear::new(1), BabyBear::new(2), BabyBear::new(3), BabyBear::new(4), // Matrix B
                BabyBear::new(1), BabyBear::new(1), BabyBear::new(0), BabyBear::new(6), // i,j,k,sum
                // Row 7
                BabyBear::new(1), BabyBear::new(2), BabyBear::new(3), BabyBear::new(4), // Matrix A
                BabyBear::new(1), BabyBear::new(2), BabyBear::new(3), BabyBear::new(4), // Matrix B
                BabyBear::new(1), BabyBear::new(1), BabyBear::new(1), BabyBear::new(22), // i,j,k,sum
            ],
            12,
        );
        assert_eq!(trace, correct_trace);
    }

    #[test]
    fn test_matrix_multiplication() {
        // [[1, 2, 3], [4, 5, 6]]
        let matrix_a: RowMajorMatrix<BabyBear> = RowMajorMatrix::new(
            vec![
                BabyBear::new(1),
                BabyBear::new(2),
                BabyBear::new(3),
                BabyBear::new(4),
                BabyBear::new(5),
                BabyBear::new(6),
            ],
            3,
        );
        // [[1, 2], [3, 4], [5, 6]]
        let matrix_b: RowMajorMatrix<BabyBear> = RowMajorMatrix::new(
            vec![
                BabyBear::new(1),
                BabyBear::new(2),
                BabyBear::new(3),
                BabyBear::new(4),
                BabyBear::new(5),
                BabyBear::new(6),
            ],
            2,
        );

        let real_result = RowMajorMatrix::new(
            vec![
                BabyBear::new(22),
                BabyBear::new(28),
                BabyBear::new(49),
                BabyBear::new(64),
            ],
            2,
        );
        let result = matrix_multiply::<BabyBear>(&matrix_a, &matrix_b);
        assert_eq!(result, real_result);
    }
}
