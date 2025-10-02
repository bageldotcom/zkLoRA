use p3_field::{PrimeCharacteristicRing, PrimeField};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_mersenne_31::Mersenne31;
use p3_uni_stark::{prove, verify};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::wrap_pyfunction;
use pyo3::Bound;

pub mod vector_matrix_air;

use vector_matrix_air::{VectorMatrixMultiplicationAIR, MyConfig};

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


fn vector_matrix_transform(
    m: usize,
    n: usize,
    v: &Vec<u32>,
    a: &Vec<Vec<u32>>,
) -> (Vec<Mersenne31>, RowMajorMatrix<Mersenne31>) {
    assert_eq!(v.len(), m, "Vector length must be m");
    assert_eq!(a.len(), m, "Matrix must have m rows");
    // Convert vector v from u32 to Mersenne31
    let vector: Vec<Mersenne31> = v.iter().map(|&x| Mersenne31::from_u32(x)).collect();

    // Flatten the matrix a (Vec<Vec<u32>>) into a single Vec<Mersenne31> in row-major order
    let mut matrix_flat: Vec<Mersenne31> = Vec::with_capacity(m * n);
    for row in a {
        assert_eq!(row.len(), n, "Each row of the matrix must have n columns");
        for &val in row {
            matrix_flat.push(Mersenne31::from_u32(val));
        }
    }

    let matrix = RowMajorMatrix::new(matrix_flat, n);
    (vector, matrix)
}

/// Generates a zero-knowledge proof for vector-matrix multiplication.
///
/// This function creates a cryptographic proof that a given vector `v` was correctly
/// multiplied by a matrix `a` to produce a result vector, without revealing the actual
/// computation details. The proof can be verified by anyone without access to the
/// original inputs.
///
/// # Arguments
///
/// * `m` - The number of rows in the matrix (and the length of the input vector).
/// * `n` - The number of columns in the matrix (and the length of the output vector).
/// * `v` - A reference to the input vector of `u32` values to be multiplied.
/// * `a` - A reference to the matrix as a vector of vectors of `u32` values.
///          Must be an `m × n` matrix (m rows, n columns).
///
/// # Returns
///
/// A `Vec<u8>` containing the serialized zero-knowledge proof. This proof can be
/// verified using [`vector_matrix_multiplication_verify`] to confirm that the
/// multiplication was performed correctly without revealing the inputs.
///
/// # Panics
///
/// This function will panic if:
/// - The length of vector `v` is not equal to `m`
/// - The matrix `a` does not have exactly `m` rows
/// - Any row in matrix `a` does not have exactly `n` columns
///
/// # Errors
///
/// This function will return an error if:
/// - The proof generation fails
/// - The proof serialization fails
///
/// # Implementation Details
///
/// The function:
/// 1. Transforms the input vector and matrix into the appropriate field representation
/// 2. Creates a `VectorMatrixMultiplicationAIR` instance for the given dimensions
/// 3. Generates an execution trace for the computation
/// 4. Produces a zero-knowledge proof using the STARK proof system
/// 5. Serializes the proof using bincode for storage/transmission
///
/// The proof is generated using the Plonky3 STARK system with Mersenne31 field elements.
#[pyfunction]
pub fn vector_matrix_multiplication_prove(
    m: usize,
    n: usize,
    v: Vec<u32>,
    a: Vec<Vec<u32>>,
) -> Vec<u8> {
    let (vector, matrix) = vector_matrix_transform(m, n, &v, &a);

    let air = VectorMatrixMultiplicationAIR::new(m, n);
    let trace = air.generate_trace(&vector, &matrix);
    let proof = prove(&air.config, &air, trace, &vec![]);

    let config = bincode::config::standard()
        .with_little_endian()
        .with_fixed_int_encoding();
    bincode::serde::encode_to_vec(proof, config).expect("Failed to serialize proof")
}

// Add a public helper that verifies a serialized proof.
/// Verify a previously generated vector-matrix multiplication proof.
///
/// # Arguments
/// * `m` - Number of rows (length of the vector).
/// * `n` - Number of columns of the matrix.
/// * `proof` - A byte vector containing the serialized proof (as produced by
///             [`vector_matrix_multiplication_prove`]).
///
/// # Returns
/// `true` if the proof is valid, `false` otherwise.
#[pyfunction]
pub fn vector_matrix_multiplication_verify(m: usize, n: usize, proof_bytes: Vec<u8>) -> bool {
    // Deserialize proof bytes
    let config_bin = bincode::config::standard()
        .with_little_endian()
        .with_fixed_int_encoding();

    let (proof_deser, _): (p3_uni_stark::Proof<MyConfig>, usize) =
        match bincode::serde::decode_from_slice(&proof_bytes, config_bin) {
            Ok(res) => res,
            Err(_) => return false, // invalid encoding
        };

    let air = VectorMatrixMultiplicationAIR::new(m, n);
    verify(&air.config, &air, &proof_deser, &vec![]).is_ok()
}

/// The Python module definition.
/// `m` is the module object that will be returned to Python.
#[pymodule]
fn plonky3_py(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(vector_matrix_multiplication_prove, m)?)?;
    m.add_function(wrap_pyfunction!(vector_matrix_multiplication_verify, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_field::integers::QuotientMap;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_mersenne_31::Mersenne31;
    use p3_uni_stark::{prove, verify, Proof, StarkGenericConfig};

    /// Prints the execution trace of a vector-matrix multiplication as a table.
    ///
    /// # Arguments
    ///
    /// * `trace` - A reference to a `RowMajorMatrix<Mersenne31>` representing the execution trace.
    ///
    /// Each row of the trace corresponds to a step in the computation, and each column contains
    /// a value relevant to the AIR (Algebraic Intermediate Representation) for the vector-matrix multiplication.
    /// This function prints each row of the trace, with values separated by commas, for debugging and inspection.
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

    /// Report the size of the serialized proof.
    ///
    /// Serializes the given proof instance using bincode and prints the size in bytes.
    /// Panics if serialization fails.
    fn report_proof_size<SC>(proof: &Proof<SC>)
    where
        SC: StarkGenericConfig,
    {
        let config = bincode::config::standard()
            .with_little_endian()
            .with_fixed_int_encoding();
        let proof_bytes =
            bincode::serde::encode_to_vec(proof, config).expect("Failed to serialize proof");
        println!("Proof size: {} bytes", proof_bytes.len());
    }

    #[test]
    fn test_vector_matrix_multiplication_prove() {
        let vector = vec![1, 2, 3];
        let matrix = vec![vec![1, 2], vec![4, 5], vec![7, 8]];
        let proof = vector_matrix_multiplication_prove(3, 2, vector, matrix);

        let result = vector_matrix_multiplication_verify(3, 2, proof);
        assert!(result);
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

        println!("matrix: {:?}", matrix);

        let air = VectorMatrixMultiplicationAIR::new(3, 3);
        let trace = air.generate_trace(&vector, &matrix);
        println!("trace width: {:?}", trace.width());
        print_trace(&trace);
        let proof = prove(&air.config, &air, trace, &vec![]);
        report_proof_size(&proof);
        let result = verify(&air.config, &air, &proof, &vec![]);
        assert!(result.is_ok());
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
                Mersenne31::from_int(1), Mersenne31::from_int(2), Mersenne31::from_int(1), Mersenne31::from_int(3), Mersenne31::from_int(2), Mersenne31::from_int(4), Mersenne31::from_int(1), Mersenne31::from_int(0), Mersenne31::from_int(1), Mersenne31::from_int(0), Mersenne31::from_int(0), Mersenne31::from_int(0), Mersenne31::from_int(1), Mersenne31::from_int(1),
                Mersenne31::from_int(1), Mersenne31::from_int(2), Mersenne31::from_int(1), Mersenne31::from_int(3), Mersenne31::from_int(2), Mersenne31::from_int(4), Mersenne31::from_int(0), Mersenne31::from_int(1), Mersenne31::from_int(0), Mersenne31::from_int(1), Mersenne31::from_int(0), Mersenne31::from_int(0), Mersenne31::from_int(7), Mersenne31::from_int(1),
                Mersenne31::from_int(1), Mersenne31::from_int(2), Mersenne31::from_int(1), Mersenne31::from_int(3), Mersenne31::from_int(2), Mersenne31::from_int(4), Mersenne31::from_int(1), Mersenne31::from_int(0), Mersenne31::from_int(0), Mersenne31::from_int(0), Mersenne31::from_int(1), Mersenne31::from_int(0), Mersenne31::from_int(2), Mersenne31::from_int(1),
                Mersenne31::from_int(1), Mersenne31::from_int(2), Mersenne31::from_int(1), Mersenne31::from_int(3), Mersenne31::from_int(2), Mersenne31::from_int(4), Mersenne31::from_int(0), Mersenne31::from_int(1), Mersenne31::from_int(0), Mersenne31::from_int(0), Mersenne31::from_int(0), Mersenne31::from_int(1), Mersenne31::from_int(10), Mersenne31::from_int(1),
            ],
            14,
        );
        assert_eq!(trace.width, correct_trace.width);
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
}
