"""JIT-compiled helper functions for Emu."""

import numpy as np

try:
    from numba import jit, njit, prange, float32, int32, float64, int64
    NUMBA_AVAILABLE = True
except ImportError:
    # Define dummy decorator if numba not available
    def jit(*args, **kwargs):
        if callable(args[0]):
            return args[0]
        else:
            def decorator(func):
                return func
            return decorator
    njit = jit
    def prange(*args):
        return range(*args)
    NUMBA_AVAILABLE = False

@njit
def batch_process_alignments(alignments_data, cigar_stats_list, ref_names, query_names, align_lens):
    """
    Process a batch of alignments in parallel using Numba.

    Args:
        alignments_data: List of alignment data dictionaries
        cigar_stats_list: List of CIGAR statistics arrays
        ref_names: List of reference names
        query_names: List of query names
        align_lens: List of alignment lengths

    Returns:
        Processed alignment data
    """
    results = []
    for i in range(len(alignments_data)):
        # Process each alignment
        results.append(process_single_alignment(
            alignments_data[i],
            cigar_stats_list[i],
            ref_names[i],
            query_names[i],
            align_lens[i]
        ))
    return results

@njit
def process_single_alignment(alignment_data, cigar_stats, ref_name, query_name, align_len):
    """
    Process a single alignment with Numba acceleration.

    Args:
        alignment_data: Alignment data dictionary
        cigar_stats: CIGAR statistics array
        ref_name: Reference name
        query_name: Query name
        align_len: Alignment length

    Returns:
        Processed alignment data
    """
    # Extract species_tid from ref_name (format: species_tid:...)
    species_tid = int(ref_name.split(':')[0])

    # Implement alignment-specific processing
    # This is a simplified example - expand for your specific needs
    return {
        'species_tid': species_tid,
        'query_name': query_name,
        'align_len': align_len,
        'processed': True
    }

@njit
def process_p_matrix(matrix_data, matrix_indices, matrix_indptr, row_count, col_count):
    """
    Process a probability matrix with JIT acceleration.

    Args:
        matrix_data: Matrix data array
        matrix_indices: Matrix column indices
        matrix_indptr: Matrix row pointers
        row_count: Number of rows
        col_count: Number of columns

    Returns:
        Processed matrix data
    """
    result = np.zeros((row_count, col_count), dtype=np.float32)

    for i in range(row_count):
        start, end = matrix_indptr[i], matrix_indptr[i+1]
        for j in range(start, end):
            col = matrix_indices[j]
            result[i, col] = matrix_data[j]

    return result

@njit
def efficient_sum(array):
    """
    Efficiently sum array elements with JIT.

    Args:
        array: Array to sum

    Returns:
        Sum of array elements
    """
    result = 0.0
    for i in range(len(array)):
        result += array[i]
    return result

@njit
def vector_dot_product(vec1, vec2):
    """
    Compute dot product of two vectors with JIT.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Dot product result
    """
    result = 0.0
    for i in range(len(vec1)):
        result += vec1[i] * vec2[i]
    return result

@njit
def vector_multiply(vec, scalar):
    """
    Multiply a vector by a scalar with JIT.

    Args:
        vec: Vector
        scalar: Scalar value

    Returns:
        Scaled vector
    """
    result = np.zeros_like(vec)
    for i in range(len(vec)):
        result[i] = vec[i] * scalar
    return result

@njit
def matrix_vector_multiply(matrix, vector):
    """
    Multiply a matrix by a vector with JIT.

    Args:
        matrix: 2D matrix
        vector: 1D vector

    Returns:
        Result vector
    """
    rows, cols = matrix.shape
    result = np.zeros(rows, dtype=matrix.dtype)

    for i in range(rows):
        for j in range(cols):
            result[i] += matrix[i, j] * vector[j]

    return result

@njit
def sparse_matrix_vector_multiply(data, indices, indptr, vector, result_shape):
    """
    Multiply a sparse matrix by a vector with JIT.

    Args:
        data: CSR matrix data array
        indices: CSR matrix column indices
        indptr: CSR matrix row pointers
        vector: Vector to multiply by
        result_shape: Shape of result vector

    Returns:
        Result vector
    """
    result = np.zeros(result_shape, dtype=data.dtype)

    for i in range(len(indptr) - 1):
        start, end = indptr[i], indptr[i+1]
        for j in range(start, end):
            col = indices[j]
            result[i] += data[j] * vector[col]

    return result

@njit(parallel=True)
def parallel_process_chunks(chunks, function):
    """
    Process chunks in parallel with JIT.

    Args:
        chunks: List of data chunks
        function: JIT-compiled function to apply to each chunk

    Returns:
        List of processed chunks
    """
    result = [None] * len(chunks)

    for i in prange(len(chunks)):
        result[i] = function(chunks[i])

    return result