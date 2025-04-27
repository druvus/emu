"""Expectation Maximization strategy implementations for Emu."""

import math
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Set, Optional, Union, Any
from collections import defaultdict
from scipy import sparse
from flatten_dict import unflatten

from emu.core.jit_utils import NUMBA_AVAILABLE
from emu.core.matrix_ops import (
    compute_row_max, compute_exp_and_sum, compute_posterior_and_freqs,
    parallel_compute_posterior, compute_log_joint, create_sparse_log_joint
)
from emu.models.errors import CalculationError

logger = logging.getLogger(__name__)

class EMStrategy(ABC):
    """Base strategy for Expectation Maximization algorithm implementation."""

    @abstractmethod
    def run_em(self, log_p_rgs, freq, max_iterations=100, epsilon=1e-6):
        """Run Expectation Maximization algorithm.

        Args:
            log_p_rgs: Dict mapping query_name to ([tax_ids], [log_scores])
            freq: Dict mapping species_tax_id to likelihood
            max_iterations: Maximum number of iterations
            epsilon: Convergence threshold

        Returns:
            Tuple containing:
                - Updated frequency dict
                - Total log likelihood
                - Dict mapping species_tax_id to {read_id: probability}
        """
        pass

class VectorizedEMStrategy(EMStrategy):
    """Vectorized implementation of EM algorithm."""

    def __init__(self, parallel: bool = True, **kwargs):
        """Initialize vectorized EM strategy.

        Args:
            parallel: Whether to enable parallel processing (when available)
            kwargs: Additional arguments (ignored, for compatibility)
        """
        self.parallel = parallel

    def run_em(self, log_p_rgs, freq, max_iterations=100, epsilon=1e-6):
        """Run vectorized EM algorithm.

        Args:
            log_p_rgs: Dict mapping query_name to ([tax_ids], [log_scores])
            freq: Dict mapping species_tax_id to likelihood
            max_iterations: Maximum number of iterations
            epsilon: Convergence threshold

        Returns:
            Tuple containing:
                - Updated frequency dict
                - Total log likelihood
                - Dict mapping species_tax_id to {read_id: probability}
        """
        try:
            # Set up data structures for sparse matrices
            all_reads = list(log_p_rgs.keys())
            all_taxa = list(freq.keys())
            read_to_idx = {read: i for i, read in enumerate(all_reads)}
            taxon_to_idx = {taxon: i for i, taxon in enumerate(all_taxa)}

            # Create sparse matrices - prepare data
            rows = []
            cols = []
            data = []

            # Prepare data for sparse matrix construction
            for read_idx, read in enumerate(all_reads):
                valid_taxa = []
                log_scores = []

                # Filter only taxa with non-zero frequencies
                for i, taxon in enumerate(log_p_rgs[read][0]):
                    if taxon in freq and freq[taxon] > 0:
                        valid_taxa.append(taxon)
                        log_scores.append(log_p_rgs[read][1][i])

                if valid_taxa:
                    for i, taxon in enumerate(valid_taxa):
                        rows.append(read_idx)
                        cols.append(taxon_to_idx[taxon])
                        data.append(log_scores[i])

            # Create the sparse log probability matrix
            n_reads = len(all_reads)
            n_taxa = len(all_taxa)

            # Use CSR format for better performance with our operations
            log_prob_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n_reads, n_taxa))

            # Convert frequency vector to numpy array
            freq_vector = np.zeros(n_taxa)
            for taxon, freq_val in freq.items():
                freq_vector[taxon_to_idx[taxon]] = freq_val

            # Ensure freq_vector sums to 1
            freq_vector = freq_vector / np.sum(freq_vector)

            # Prepare for iterations
            log_freq_vector = np.log(np.maximum(freq_vector, 1e-15))  # Prevent log(0)
            log_likelihood_prev = -np.inf

            # Get CSR matrix components for operations
            log_prob_data = log_prob_matrix.data
            log_prob_indices = log_prob_matrix.indices
            log_prob_indptr = log_prob_matrix.indptr

            # Perform EM iterations until convergence or max iterations
            for iteration in range(max_iterations):
                # E-step: Calculate log joint probabilities
                log_joint_data = compute_log_joint(
                    log_prob_data, log_prob_indices, log_prob_indptr,
                    log_freq_vector, n_reads
                )

                # Create a view of the log joint matrix with updated data
                log_joint = sparse.csr_matrix(
                    (log_joint_data, log_prob_indices, log_prob_indptr),
                    shape=(n_reads, n_taxa)
                )

                # Calculate row maximums
                row_max = compute_row_max(
                    log_joint_data, log_prob_indices, log_prob_indptr, n_reads
                )

                # Calculate exp(log_joint - row_max) and row sums
                exp_data, row_sums = compute_exp_and_sum(
                    log_joint_data, log_prob_indices, log_prob_indptr,
                    row_max, n_reads
                )

                # Choose appropriate posterior calculation method based on parallel flag
                if self.parallel and NUMBA_AVAILABLE and n_reads > 1000:
                    # Use parallel computation for large matrices
                    posterior_data, new_freq_vector = parallel_compute_posterior(
                        exp_data, log_prob_indices, log_prob_indptr,
                        row_sums, n_reads, n_taxa
                    )
                else:
                    # Use standard computation
                    posterior_data, new_freq_vector = compute_posterior_and_freqs(
                        exp_data, log_prob_indices, log_prob_indptr,
                        row_sums, n_reads, n_taxa
                    )

                # Create posterior matrix from data
                posterior = sparse.csr_matrix(
                    (posterior_data, log_prob_indices, log_prob_indptr),
                    shape=(n_reads, n_taxa)
                )

                # Calculate log-likelihood
                log_likelihood = 0.0
                for i in range(n_reads):
                    if row_sums[i] > 0:
                        log_likelihood += np.log(row_sums[i]) + row_max[i]

                # Check convergence
                rel_change = np.max(np.abs(new_freq_vector - freq_vector) / (freq_vector + epsilon))
                freq_vector = new_freq_vector
                log_freq_vector = np.log(np.maximum(freq_vector, 1e-15))

                # Print progress if verbose
                logger.debug(f"EM Sub-iteration {iteration+1}: log-likelihood = {log_likelihood:.4f}, rel_change = {rel_change:.6f}")

                # Check if converged
                if rel_change < epsilon:
                    logger.debug(f"EM Sub-iterations converged after {iteration+1} iterations")
                    break

                # Check if log-likelihood improved
                if log_likelihood < log_likelihood_prev:
                    logger.warning("Log-likelihood decreased, stopping early")
                    break

                log_likelihood_prev = log_likelihood

            # Convert results back to dictionaries
            frq = {all_taxa[i]: freq_vector[i] for i in range(n_taxa) if freq_vector[i] > 0}

            # Create probability matrix for species-read assignments
            p_sgr_flat = {}
            for i in range(posterior.indptr.size - 1):
                read = all_reads[i]
                start, end = posterior.indptr[i], posterior.indptr[i+1]
                for j in range(start, end):
                    col = posterior.indices[j]
                    taxon = all_taxa[col]
                    p_sgr_flat[(taxon, read)] = posterior.data[j]

            # Convert flat dictionary to nested structure
            p_sgr = unflatten(p_sgr_flat)

            return frq, log_likelihood, p_sgr

        except Exception as e:
            raise CalculationError(f"Error in vectorized expectation maximization: {str(e)}")

class BatchEMStrategy(EMStrategy):
    """Batched implementation of EM algorithm for large datasets."""

    def __init__(self, batch_size: int = 1000, parallel: bool = True, **kwargs):
        """Initialize batch EM strategy.

        Args:
            batch_size: Number of reads to process in each batch
            parallel: Whether to enable parallel processing (when available)
            kwargs: Additional arguments (ignored, for compatibility)
        """
        self.batch_size = batch_size
        self.parallel = parallel

    def run_em(self, log_p_rgs, freq, max_iterations=100, epsilon=1e-6):
        """Run batched EM algorithm.

        Args:
            log_p_rgs: Dict mapping query_name to ([tax_ids], [log_scores])
            freq: Dict mapping species_tax_id to likelihood
            max_iterations: Maximum number of iterations
            epsilon: Convergence threshold

        Returns:
            Tuple containing:
                - Updated frequency dict
                - Total log likelihood
                - Dict mapping species_tax_id to {read_id: probability}
        """
        try:
            all_reads = list(log_p_rgs.keys())
            n_reads = len(all_reads)

            # If dataset is small, use vectorized implementation directly
            if n_reads <= self.batch_size:
                vectorized_strategy = VectorizedEMStrategy(parallel=self.parallel)
                return vectorized_strategy.run_em(log_p_rgs, freq, max_iterations, epsilon)

            # Split reads into batches
            batch_count = (n_reads + self.batch_size - 1) // self.batch_size
            read_batches = np.array_split(all_reads, batch_count)

            logger.info(f"Processing {n_reads} reads in {batch_count} batches of ~{self.batch_size} reads each")

            # Initialize accumulators
            total_log_likelihood = 0.0
            p_sgr_combined = {}
            taxa_counts = defaultdict(float)

            # Create vectorized strategy for batch processing
            vectorized_strategy = VectorizedEMStrategy(parallel=self.parallel)

            # Process each batch
            for batch_idx, batch_reads in enumerate(read_batches):
                logger.debug(f"Processing batch {batch_idx+1}/{batch_count} ({len(batch_reads)} reads)")

                # Create batch subset of log_p_rgs
                batch_log_p_rgs = {read: log_p_rgs[read] for read in batch_reads}

                # Run EM on this batch
                _, batch_log_likelihood, batch_p_sgr = vectorized_strategy.run_em(
                    batch_log_p_rgs, freq, max_iterations=3, epsilon=1e-4)

                # Accumulate log likelihood
                total_log_likelihood += batch_log_likelihood

                # Accumulate taxon counts from posterior probabilities
                for taxon, read_probs in batch_p_sgr.items():
                    taxa_counts[taxon] += sum(read_probs.values())

                    # Update combined probability matrix
                    if taxon not in p_sgr_combined:
                        p_sgr_combined[taxon] = {}
                    p_sgr_combined[taxon].update(read_probs)

            # Calculate final frequencies
            updated_freq = {taxon: count / n_reads for taxon, count in taxa_counts.items()}

            # Ensure frequencies sum to 1
            freq_sum = sum(updated_freq.values())
            if freq_sum == 0:
                logger.warning("Zero frequency sum after batch processing")
                return {}, 0.0, {}
            elif not 0.99 <= freq_sum <= 1.01:
                logger.warning(f"Normalizing frequency vector from {freq_sum} to 1.0")
                updated_freq = {k: v / freq_sum for k, v in updated_freq.items()}

            return updated_freq, total_log_likelihood, p_sgr_combined

        except Exception as e:
            raise CalculationError(f"Error in batch expectation maximization: {str(e)}")

class MemoryEfficientEMStrategy(EMStrategy):
    """Memory-efficient implementation of EM algorithm."""

    def __init__(self, parallel: bool = True, **kwargs):
        """Initialize memory-efficient EM strategy.

        Args:
            parallel: Whether to enable parallel processing (when available)
            kwargs: Additional arguments (ignored, for compatibility)
        """
        self.parallel = parallel

    def run_em(self, log_p_rgs, freq, max_iterations=100, epsilon=1e-6):
        """Run memory-efficient EM algorithm.

        This implementation minimizes memory usage by:
        1. Using sparse data structures consistently
        2. Avoiding creation of unnecessary copies
        3. Using in-place operations where possible

        Args:
            log_p_rgs: Dict mapping query_name to ([tax_ids], [log_scores])
            freq: Dict mapping species_tax_id to likelihood
            max_iterations: Maximum number of iterations
            epsilon: Convergence threshold

        Returns:
            Tuple containing:
                - Updated frequency dict
                - Total log likelihood
                - Dict mapping species_tax_id to {read_id: probability}
        """
        try:
            # Create sparse log joint matrix directly
            log_joint = create_sparse_log_joint(log_p_rgs, freq)
            all_reads = list(log_p_rgs.keys())
            all_taxa = list(freq.keys())
            n_reads = len(all_reads)
            n_taxa = len(all_taxa)

            # Convert frequency vector to numpy array
            freq_vector = np.zeros(n_taxa)
            for i, taxon in enumerate(all_taxa):
                freq_vector[i] = freq.get(taxon, 0)

            # Ensure freq_vector sums to 1
            freq_vector = freq_vector / np.sum(freq_vector)

            # Track changes instead of full history
            prev_freq_vector = np.zeros_like(freq_vector)
            log_likelihood_prev = -np.inf

            # Perform EM iterations
            for iteration in range(max_iterations):
                # Store previous frequencies
                np.copyto(prev_freq_vector, freq_vector)

                # Calculate row maximums (for numerical stability)
                row_max = compute_row_max(
                    log_joint.data, log_joint.indices, log_joint.indptr, n_reads
                )

                # Calculate exp(log_joint - row_max) and row sums
                exp_data, row_sums = compute_exp_and_sum(
                    log_joint.data, log_joint.indices, log_joint.indptr,
                    row_max, n_reads
                )

                # Choose appropriate posterior calculation based on parallel flag
                if self.parallel and NUMBA_AVAILABLE:
                    # Use parallel implementation if available
                    posterior_data, freq_vector = parallel_compute_posterior(
                        exp_data, log_joint.indices, log_joint.indptr,
                        row_sums, n_reads, n_taxa
                    )
                else:
                    # Use standard implementation
                    posterior_data, freq_vector = compute_posterior_and_freqs(
                        exp_data, log_joint.indices, log_joint.indptr,
                        row_sums, n_reads, n_taxa
                    )

                # Calculate log-likelihood
                log_likelihood = 0.0
                for i in range(n_reads):
                    if row_sums[i] > 0:
                        log_likelihood += np.log(row_sums[i]) + row_max[i]

                # Check convergence
                rel_change = np.max(np.abs(freq_vector - prev_freq_vector) /
                                   (prev_freq_vector + epsilon))

                logger.debug(f"EM iteration {iteration+1}: log-likelihood = {log_likelihood:.4f}, rel_change = {rel_change:.6f}")

                # Check if converged
                if rel_change < epsilon:
                    logger.debug(f"EM converged after {iteration+1} iterations")
                    break

                # Check if log-likelihood improved
                if log_likelihood < log_likelihood_prev:
                    logger.warning("Log-likelihood decreased, stopping early")
                    # Revert to previous frequencies
                    np.copyto(freq_vector, prev_freq_vector)
                    break

                log_likelihood_prev = log_likelihood

                # Update log joint matrix with new frequencies (in-place)
                log_freq_vector = np.log(np.maximum(freq_vector, 1e-15))
                log_joint_new = create_sparse_log_joint(log_p_rgs,
                                                      {all_taxa[i]: freq_vector[i]
                                                       for i in range(n_taxa)})
                log_joint = log_joint_new

            # Create posterior matrix
            posterior = sparse.csr_matrix(
                (posterior_data, log_joint.indices, log_joint.indptr),
                shape=(n_reads, n_taxa)
            )

            # Convert results back to dictionaries
            frq = {all_taxa[i]: freq_vector[i] for i in range(n_taxa) if freq_vector[i] > 0}

            # Create probability matrix for species-read assignments
            p_sgr_flat = {}
            for i in range(posterior.indptr.size - 1):
                read = all_reads[i]
                start, end = posterior.indptr[i], posterior.indptr[i+1]
                for j in range(start, end):
                    col = posterior.indices[j]
                    taxon = all_taxa[col]
                    p_sgr_flat[(taxon, read)] = posterior.data[j]

            # Convert flat dictionary to nested structure
            p_sgr = unflatten(p_sgr_flat)

            return frq, log_likelihood, p_sgr

        except Exception as e:
            raise CalculationError(f"Error in memory efficient expectation maximization: {str(e)}")

def create_em_strategy(strategy_type: str = "vectorized", batch_size: int = 1000,
                  parallel: bool = True, **kwargs) -> EMStrategy:
    """Factory function to create appropriate EM strategy.

    Args:
        strategy_type: Type of strategy to create ("vectorized", "batch", "memory_efficient")
        batch_size: Batch size for batch processing strategies
        parallel: Whether to enable parallel processing (when available)
        kwargs: Additional arguments for the strategy

    Returns:
        EMStrategy object
    """
    from emu.core.jit_utils import NUMBA_AVAILABLE

    # Log information about parallel processing
    if parallel and NUMBA_AVAILABLE:
        logger.info("Parallel processing enabled for EM algorithm")
    elif parallel and not NUMBA_AVAILABLE:
        logger.info("Parallel processing requested but numba not available - using sequential processing")
        parallel = False
    elif not parallel:
        logger.info("Using sequential processing for EM algorithm (parallel disabled)")

    # Create appropriate strategy based on type
    if strategy_type.lower() == "vectorized":
        return VectorizedEMStrategy(parallel=parallel, **kwargs)
    elif strategy_type.lower() == "batch":
        return BatchEMStrategy(batch_size=batch_size, parallel=parallel, **kwargs)
    elif strategy_type.lower() == "memory_efficient":
        return MemoryEfficientEMStrategy(parallel=parallel, **kwargs)
    else:
        # Default to vectorized
        return VectorizedEMStrategy(parallel=parallel, **kwargs)