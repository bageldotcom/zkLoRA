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
    assert_eq!(a.len(), b.height(), "Vector length must match matrix height");
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
                let mut running_sum = F::ZERO;

                for k in 0..self.n {
                    // First, add all elements from matrix A
                    for a_row in 0..self.m {
                        for a_col in 0..self.n {
                            trace_data.push(a.get(a_row, a_col).unwrap());
                        }
                    }

                    // Then, add all elements from matrix B
                    for b_row in 0..self.n {
                        for b_col in 0..self.p {
                            trace_data.push(b.get(b_row, b_col).unwrap());
                        }
                    }

                    // Add indices i, j, k
                    trace_data.push(F::from_u64(i as u64));
                    trace_data.push(F::from_u64(j as u64));
                    trace_data.push(F::from_u64(k as u64));

                    // Update running sum
                    running_sum += a.get(i, k).unwrap() * b.get(k, j).unwrap();

                    // Add the running sum
                    trace_data.push(running_sum);
                }
            }
        }

        // Pad the trace to satisfy PCS minimum row requirement (>= 4 rows)
        let row_width = self.trace_width();
        if total_rows < 4 {
            // Duplicate the last row until we reach 4 rows
            let last_row_start = (total_rows - 1) * row_width;
            let last_row_end = total_rows * row_width;
            let last_row: Vec<F> = trace_data[last_row_start..last_row_end].to_vec();
            for _ in total_rows..4 {
                trace_data.extend_from_slice(&last_row);
            }
        }

        RowMajorMatrix::new(trace_data, row_width)
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
        let current = main.row_slice(0).unwrap();
        let next = main.row_slice(1).unwrap();

        let i = self.trace_width() - 4;
        let j = self.trace_width() - 3;
        let k = self.trace_width() - 2;
        let sum = self.trace_width() - 1;

        // Enforce starting state
        // Assert that the sum column of the first element of A and B multiplied
        builder.when_first_row().assert_eq(
            current[sum].clone(),
            current[0].clone() * current[self.m * self.n].clone(),
        );

        // Assert that the row index is 0
        builder
            .when_first_row()
            .assert_eq(current[i].clone(), AB::Expr::ZERO);

        // Assert that the column index is 0
        builder
            .when_first_row()
            .assert_eq(current[j].clone(), AB::Expr::ZERO);

        // Assert that the current position is 0
        builder
            .when_first_row()
            .assert_eq(current[k].clone(), AB::Expr::ZERO);

        // Enforce state transition
        // Assert that the matrices don't change between rows
        for idx in 0..(self.m * self.n + self.n * self.p) {
            builder
                .when_transition()
                .assert_eq(next[idx].clone(), current[idx].clone());
        }

        //Assert that the multiplication step is correct
        for row_idx in 0..self.m {
            for column_idx in 0..self.p {
                for idx in 0..self.n {
                    // The running sum is applied after the first index in the inner product
                    // of the row of matrix A and the column of matrix B
                    /*if idx != 0 {

                        builder.when_transition().assert_zero(
                            (current[i].clone() - AB::Expr::from_usize(row_idx))
                                + (current[j].clone() - AB::Expr::from_usize(column_idx))
                                + (current[k].clone() - AB::Expr::from_usize(idx))
                                + (next[sum].clone()
                                    - (current[sum].clone()
                                        + current[row_idx * self.n + idx].clone()
                                            * current
                                                [self.m * self.n + idx * self.p + column_idx]
                                                .clone())),
                        );
                        // The first number for the running sum is the multiplication of the
                        // first element in row row_idx and the first element in column column_idx
                    } else {
                        builder
                            .when(current[k].clone() - AB::Expr::ZERO)
                            .assert_zero(
                                current[sum].clone()
                                    - current[row_idx * self.n + idx].clone()
                                        * current[self.m * self.n + idx * self.p + column_idx]
                                            .clone(),
                            );
                    }*/
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

        // Enforce lexicographic increment of the (i, j, k) triple across transition rows:
        // flat_next = flat_current + 1, where flat = k + j * n + i * (n * p)
        let n_expr = AB::Expr::from_usize(self.n);
        let np_expr = AB::Expr::from_usize(self.n * self.p);
        let flat_current = current[k].clone()
            + current[j].clone() * n_expr.clone()
            + current[i].clone() * np_expr.clone();
        let flat_next = next[k].clone() + next[j].clone() * n_expr + next[i].clone() * np_expr;
        builder
            .when_transition()
            .assert_eq(flat_next, flat_current + AB::Expr::ONE);

        // Enforce finale state
        // Assert that columns i,j,k are all the max values
        builder
            .when_last_row()
            .assert_eq(current[i].clone(), AB::Expr::from_usize(self.m - 1));
        builder
            .when_last_row()
            .assert_eq(current[j].clone(), AB::Expr::from_usize(self.p - 1));
        builder
            .when_last_row()
            .assert_eq(current[k].clone(), AB::Expr::from_usize(self.n - 1));
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
        // [[1, 2], [3, 4]]
        let matrix_a: RowMajorMatrix<Mersenne31> = RowMajorMatrix::new(
            vec![
                Mersenne31::from_int(1),
                Mersenne31::from_int(2),
                Mersenne31::from_int(3),
                Mersenne31::from_int(4),
            ],
            2,
        );
        // [[1, 2], [3, 4]]
        let matrix_b: RowMajorMatrix<Mersenne31> = RowMajorMatrix::new(
            vec![
                Mersenne31::from_int(1),
                Mersenne31::from_int(2),
                Mersenne31::from_int(3),
                Mersenne31::from_int(4),
            ],
            2,
        );

        let air = MatrixMultiplicationAIR::new(2, 2, 2);
        let trace = air.generate_trace(&matrix_a, &matrix_b);
        print_trace(&trace);
        let proof = prove(&air.config, &air, trace, &vec![]);
    }

    #[test]
    fn test_trace() {
        // [[1, 2], [3, 4]]
        let matrix_a: RowMajorMatrix<Mersenne31> = RowMajorMatrix::new(
            vec![
                Mersenne31::from_int(1),
                Mersenne31::from_int(2),
                Mersenne31::from_int(3),
                Mersenne31::from_int(4),
            ],
            2,
        );
        // [[1, 2], [3, 4]]
        let matrix_b: RowMajorMatrix<Mersenne31> = RowMajorMatrix::new(
            vec![
                Mersenne31::from_int(1),
                Mersenne31::from_int(2),
                Mersenne31::from_int(3),
                Mersenne31::from_int(4),
            ],
            2,
        );

        let air = MatrixMultiplicationAIR::new(2, 2, 2);
        let trace = air.generate_trace(&matrix_a, &matrix_b);
        // Print the trace, one row per line
        print_trace(&trace);

        // Row 0: [1, 2, 3, 4, 1, 2, 3, 4, 0, 0, 0, 1]
        // Row 1: [1, 2, 3, 4, 1, 2, 3, 4, 0, 0, 1, 7]
        // Row 2: [1, 2, 3, 4, 1, 2, 3, 4, 0, 1, 0, 2]
        // Row 3: [1, 2, 3, 4, 1, 2, 3, 4, 0, 1, 1, 10]
        // Row 4: [1, 2, 3, 4, 1, 2, 3, 4, 1, 0, 0, 3]
        // Row 5: [1, 2, 3, 4, 1, 2, 3, 4, 1, 0, 1, 15]
        // Row 6: [1, 2, 3, 4, 1, 2, 3, 4, 1, 1, 0, 6]
        // Row 7: [1, 2, 3, 4, 1, 2, 3, 4, 1, 1, 1, 22]
        #[rustfmt::skip]
        let correct_trace: RowMajorMatrix<Mersenne31> = RowMajorMatrix::new(
            vec![
                // Row 0
                Mersenne31::from_int(1), Mersenne31::from_int(2), Mersenne31::from_int(3), Mersenne31::from_int(4), // Matrix A
                Mersenne31::from_int(1), Mersenne31::from_int(2), Mersenne31::from_int(3), Mersenne31::from_int(4), // Matrix B
                Mersenne31::from_int(0), Mersenne31::from_int(0), Mersenne31::from_int(0), Mersenne31::from_int(1), // i,j,k,sum
                // Row 1
                Mersenne31::from_int(1), Mersenne31::from_int(2), Mersenne31::from_int(3), Mersenne31::from_int(4), // Matrix A
                Mersenne31::from_int(1), Mersenne31::from_int(2), Mersenne31::from_int(3), Mersenne31::from_int(4), // Matrix B
                Mersenne31::from_int(0), Mersenne31::from_int(0), Mersenne31::from_int(1), Mersenne31::from_int(7), // i,j,k,sum
                // Row 2
                Mersenne31::from_int(1), Mersenne31::from_int(2), Mersenne31::from_int(3), Mersenne31::from_int(4), // Matrix A
                Mersenne31::from_int(1), Mersenne31::from_int(2), Mersenne31::from_int(3), Mersenne31::from_int(4), // Matrix B
                Mersenne31::from_int(0), Mersenne31::from_int(1), Mersenne31::from_int(0), Mersenne31::from_int(2), // i,j,k,sum
                // Row 3
                Mersenne31::from_int(1), Mersenne31::from_int(2), Mersenne31::from_int(3), Mersenne31::from_int(4), // Matrix A
                Mersenne31::from_int(1), Mersenne31::from_int(2), Mersenne31::from_int(3), Mersenne31::from_int(4), // Matrix B
                Mersenne31::from_int(0), Mersenne31::from_int(1), Mersenne31::from_int(1), Mersenne31::from_int(10), // i,j,k,sum
                // Row 4
                Mersenne31::from_int(1), Mersenne31::from_int(2), Mersenne31::from_int(3), Mersenne31::from_int(4), // Matrix A
                Mersenne31::from_int(1), Mersenne31::from_int(2), Mersenne31::from_int(3), Mersenne31::from_int(4), // Matrix B
                Mersenne31::from_int(1), Mersenne31::from_int(0), Mersenne31::from_int(0), Mersenne31::from_int(3), // i,j,k,sum
                // Row 5
                Mersenne31::from_int(1), Mersenne31::from_int(2), Mersenne31::from_int(3), Mersenne31::from_int(4), // Matrix A
                Mersenne31::from_int(1), Mersenne31::from_int(2), Mersenne31::from_int(3), Mersenne31::from_int(4), // Matrix B
                Mersenne31::from_int(1), Mersenne31::from_int(0), Mersenne31::from_int(1), Mersenne31::from_int(15), // i,j,k,sum
                // Row 6
                Mersenne31::from_int(1), Mersenne31::from_int(2), Mersenne31::from_int(3), Mersenne31::from_int(4), // Matrix A
                Mersenne31::from_int(1), Mersenne31::from_int(2), Mersenne31::from_int(3), Mersenne31::from_int(4), // Matrix B
                Mersenne31::from_int(1), Mersenne31::from_int(1), Mersenne31::from_int(0), Mersenne31::from_int(6), // i,j,k,sum
                // Row 7
                Mersenne31::from_int(1), Mersenne31::from_int(2), Mersenne31::from_int(3), Mersenne31::from_int(4), // Matrix A
                Mersenne31::from_int(1), Mersenne31::from_int(2), Mersenne31::from_int(3), Mersenne31::from_int(4), // Matrix B
                Mersenne31::from_int(1), Mersenne31::from_int(1), Mersenne31::from_int(1), Mersenne31::from_int(22), // i,j,k,sum
            ],
            12,
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
