"""Matrix operations for Emu with JIT optimization when available."""

import numpy as np
import logging
from scipy import sparse

from emu.core.jit_utils import NUMBA_AVAILABLE, njit, prange

logger = logging.getLogger(__name__)

# JIT-optimized sparse matrix operations
if NUMBA_AVAILABLE:
    @njit
    def compute_row_max(data, indices, indptr, n_rows):
        """JIT-optimized row maximum calculation for sparse matrix.

        Args:
            data: CSR matrix data array
            indices: CSR matrix column indices
            indptr: CSR matrix row pointers
            n_rows: Number of rows

        Returns:
            Array of row maximums
        """
        row_max = np.full(n_rows, -np.inf)

        for i in range(len(indptr) - 1):
            start, end = indptr[i], indptr[i+1]
            if start < end:  # If row has entries
                row_max_val = -np.inf
                for j in range(start, end):
                    row_max_val = max(row_max_val, data[j])
                row_max[i] = row_max_val

        return row_max

    @njit
    def compute_exp_and_sum(data, indices, indptr, row_max, n_rows):
        """JIT-optimized exponentiation and row sum calculation.

        Args:
            data: CSR matrix data array
            indices: CSR matrix column indices
            indptr: CSR matrix row pointers
            row_max: Array of row maximums
            n_rows: Number of rows

        Returns:
            Tuple containing:
                - Exponentiated data array
                - Array of row sums
        """
        exp_data = np.zeros_like(data)
        row_sums = np.zeros(n_rows)

        for i in range(len(indptr) - 1):
            start, end = indptr[i], indptr[i+1]
            row_sum = 0.0

            for j in range(start, end):
                exp_val = np.exp(data[j] - row_max[i])
                exp_data[j] = exp_val
                row_sum += exp_val

            row_sums[i] = row_sum

        return exp_data, row_sums

    @njit
    def compute_posterior_and_freqs(exp_data, indices, indptr, row_sums, n_rows, n_taxa):
        """JIT-optimized posterior probability and frequency computation.

        Args:
            exp_data: Exponentiated data array
            indices: CSR matrix column indices
            indptr: CSR matrix row pointers
            row_sums: Array of row sums
            n_rows: Number of rows
            n_taxa: Number of taxa (columns)

        Returns:
            Tuple containing:
                - Posterior probabilities array
                - New frequency vector
        """
        posterior_data = np.zeros_like(exp_data)
        new_freq_vector = np.zeros(n_taxa)

        for i in range(len(indptr) - 1):
            start, end = indptr[i], indptr[i+1]
            if start < end and row_sums[i] > 0:
                for j in range(start, end):
                    posterior_data[j] = exp_data[j] / row_sums[i]
                    col = indices[j]
                    new_freq_vector[col] += posterior_data[j]

        # Normalize new frequencies
        total_reads = max(1, n_rows)
        new_freq_vector /= total_reads

        return posterior_data, new_freq_vector

    @njit(parallel=True)
    def parallel_compute_posterior(exp_data, indices, indptr, row_sums, n_rows, n_taxa):
        """Compute posterior probabilities in parallel.

        Args:
            exp_data: Exponentiated data array
            indices: CSR matrix column indices
            indptr: CSR matrix row pointers
            row_sums: Array of row sums
            n_rows: Number of rows
            n_taxa: Number of taxa (columns)

        Returns:
            Tuple containing:
                - Posterior probabilities array
                - New frequency vector
        """
        posterior_data = np.zeros_like(exp_data)
        # Process in tiles to avoid race conditions
        freq_tiles = np.zeros((min(32, n_rows), n_taxa))

        # Process rows in parallel, write to separate freq vectors to avoid race conditions
        for tile_idx in prange(32):  # Fixed number of tiles
            start_row = tile_idx * (n_rows // 32)
            end_row = (tile_idx + 1) * (n_rows // 32) if tile_idx < 31 else n_rows

            # Local frequency accumulator for this tile
            local_freq = np.zeros(n_taxa)

            for i in range(start_row, end_row):
                row_start, row_end = indptr[i], indptr[i+1]
                if row_start < row_end and row_sums[i] > 0:
                    for j in range(row_start, row_end):
                        post_val = exp_data[j] / row_sums[i]
                        posterior_data[j] = post_val
                        col = indices[j]
                        local_freq[col] += post_val

            # Store in the tile array
            freq_tiles[tile_idx] = local_freq

        # Combine all frequency tiles
        new_freq_vector = np.sum(freq_tiles, axis=0) / max(1, n_rows)

        return posterior_data, new_freq_vector

    @njit
    def compute_log_joint(log_prob_data, log_prob_indices, log_prob_indptr, log_freq_vector, n_reads):
        """JIT-optimized computation of log joint probabilities.

        Args:
            log_prob_data: Log probability data array
            log_prob_indices: Log probability column indices
            log_prob_indptr: Log probability row pointers
            log_freq_vector: Log frequency vector
            n_reads: Number of reads

        Returns:
            Log joint probability data array
        """
        log_joint_data = np.copy(log_prob_data)

        # Add log frequencies to each column
        for i in range(len(log_prob_indptr) - 1):
            start, end = log_prob_indptr[i], log_prob_indptr[i+1]
            for j in range(start, end):
                col = log_prob_indices[j]
                log_joint_data[j] += log_freq_vector[col]

        return log_joint_data

    @njit
    def sparse_matrix_vector_multiply(data, indices, indptr, vector, result_shape):
        """Multiply a sparse matrix by a vector with JIT.

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

else:
    # Non-JIT fallbacks that use numpy/scipy operations
    def compute_row_max(data, indices, indptr, n_rows):
        """Calculate row maximums for sparse matrix using scipy operations."""
        # Create a CSR matrix for easier manipulation
        matrix = sparse.csr_matrix((data, indices, indptr), shape=(n_rows, max(indices) + 1))

        # Use scipy sparse max operation
        return matrix.max(axis=1).A.flatten()

    def compute_exp_and_sum(data, indices, indptr, row_max, n_rows):
        """Calculate exp values and row sums using scipy operations."""
        # Create CSR matrix
        matrix = sparse.csr_matrix((data, indices, indptr), shape=(n_rows, max(indices) + 1))

        # Subtract row_max from each row (for numerical stability)
        exp_matrix = matrix.copy()
        for i in range(n_rows):
            row_indices = slice(indptr[i], indptr[i+1])
            exp_matrix.data[row_indices] -= row_max[i]

        # Exponentiate
        exp_matrix.data = np.exp(exp_matrix.data)

        # Calculate row sums
        row_sums = exp_matrix.sum(axis=1).A.flatten()

        return exp_matrix.data, row_sums

    def compute_posterior_and_freqs(exp_data, indices, indptr, row_sums, n_rows, n_taxa):
        """Calculate posterior probabilities and frequencies using scipy operations."""
        # Create CSR matrix for exp values
        exp_matrix = sparse.csr_matrix((exp_data, indices, indptr), shape=(n_rows, n_taxa))

        # Normalize each row to get posterior probabilities
        posterior_matrix = exp_matrix.copy()
        for i in range(n_rows):
            if row_sums[i] > 0:
                row_indices = slice(indptr[i], indptr[i+1])
                posterior_matrix.data[row_indices] /= row_sums[i]

        # Calculate new frequencies (column sums)
        new_freq_vector = posterior_matrix.sum(axis=0).A.flatten() / max(1, n_rows)

        return posterior_matrix.data, new_freq_vector

    def parallel_compute_posterior(exp_data, indices, indptr, row_sums, n_rows, n_taxa):
        """Non-parallel fallback for posterior computation."""
        return compute_posterior_and_freqs(exp_data, indices, indptr, row_sums, n_rows, n_taxa)

    def compute_log_joint(log_prob_data, log_prob_indices, log_prob_indptr, log_freq_vector, n_reads):
        """Compute log joint probabilities using scipy operations."""
        # Create CSR matrix
        log_prob_matrix = sparse.csr_matrix(
            (log_prob_data, log_prob_indices, log_prob_indptr),
            shape=(n_reads, len(log_freq_vector))
        )

        # Add log frequencies to each column
        log_joint_matrix = log_prob_matrix.copy()
        n_taxa = len(log_freq_vector)

        # Loop through columns and add log frequency
        for j in range(n_taxa):
            col_indices = log_prob_matrix.getcol(j).indices
            if len(col_indices) > 0:
                for idx in col_indices:
                    data_idx = log_joint_matrix.indptr[idx] + np.where(log_joint_matrix.indices[
                        log_joint_matrix.indptr[idx]:log_joint_matrix.indptr[idx+1]
                    ] == j)[0][0]
                    log_joint_matrix.data[data_idx] += log_freq_vector[j]

        return log_joint_matrix.data

    def sparse_matrix_vector_multiply(data, indices, indptr, vector, result_shape):
        """Multiply sparse matrix by vector using scipy operations."""
        matrix = sparse.csr_matrix((data, indices, indptr), shape=(result_shape[0], len(vector)))
        return matrix.dot(vector)


def create_sparse_log_joint(log_p_rgs, freq):
    """Create sparse log joint probability matrix from log_p_rgs and freq.

    Args:
        log_p_rgs: Dict mapping query_name to ([tax_ids], [log_scores])
        freq: Dict mapping species_tax_id to likelihood

    Returns:
        Sparse CSR matrix of log joint probabilities
    """
    # Extract all unique reads and taxa
    all_reads = list(log_p_rgs.keys())
    all_taxa = list(freq.keys())
    read_to_idx = {read: i for i, read in enumerate(all_reads)}
    taxon_to_idx = {taxon: i for i, taxon in enumerate(all_taxa)}

    # Prepare data for sparse matrix
    rows = []
    cols = []
    data = []

    # Collect non-zero entries
    for read, (taxa, log_scores) in log_p_rgs.items():
        for taxon, score in zip(taxa, log_scores):
            if taxon in freq and freq[taxon] > 0:
                rows.append(read_to_idx[read])
                cols.append(taxon_to_idx[taxon])
                data.append(score)

    # Create the sparse matrix
    n_reads = len(all_reads)
    n_taxa = len(all_taxa)

    log_prob_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n_reads, n_taxa))

    # Create log frequency vector
    log_freq_vector = np.log(np.maximum(1e-15, [freq.get(taxon, 0) for taxon in all_taxa]))

    # Compute log joint probabilities
    log_joint_data = compute_log_joint(
        log_prob_matrix.data,
        log_prob_matrix.indices,
        log_prob_matrix.indptr,
        log_freq_vector,
        n_reads
    )

    # Create the log joint matrix
    log_joint_matrix = sparse.csr_matrix(
        (log_joint_data, log_prob_matrix.indices, log_prob_matrix.indptr),
        shape=(n_reads, n_taxa)
    )

    return log_joint_matrix