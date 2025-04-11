"""Abundance estimation core functionality."""

import math
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional, Union, Any
from collections import defaultdict
from operator import add, mul

import numpy as np
import pandas as pd
import pysam
from flatten_dict import unflatten
from Bio import SeqIO
from scipy import sparse

from emu.models.errors import AlignmentError, CalculationError
from emu.core.utils import (
    get_align_stats, get_align_len, CIGAR_OPS, CIGAR_OPS_ALL,
    TAXONOMY_RANKS, RANKS_PRINTOUT
)

logger = logging.getLogger(__name__)

def get_cigar_op_log_probabilities(
    sam_path: Path
) -> Tuple[List[float], List[int], Dict[str, int]]:
    """Calculate log probabilities for each CIGAR operation using vectorized operations."""
    try:
        cigar_stats_primary = np.zeros(len(CIGAR_OPS), dtype=np.int32)
        dict_longest_align = {}

        # Open SAM file
        sam_pysam = pysam.AlignmentFile(str(sam_path))

        alignment_count = 0
        primary_alignment_count = 0

        # Process alignments
        for alignment in sam_pysam.fetch():
            alignment_count += 1
            align_len = get_align_len(alignment)

            # Track longest alignment per read using vectorized max
            query_name = alignment.query_name
            dict_longest_align[query_name] = max(
                dict_longest_align.get(query_name, 0),
                align_len
            )

            # Process primary alignments only
            if (not alignment.is_secondary and
                not alignment.is_supplementary and
                alignment.reference_name):
                primary_alignment_count += 1
                current_stats = np.array(get_align_stats(alignment), dtype=np.int32)
                cigar_stats_primary += current_stats

        logger.debug(f"Total alignments: {alignment_count}, Primary alignments: {primary_alignment_count}")

        # Check if any probabilities are 0, if so, remove
        zero_locs = np.where(cigar_stats_primary == 0)[0].tolist()
        if zero_locs:
            cigar_stats_primary = np.delete(cigar_stats_primary, zero_locs)

        # Calculate log probabilities
        n_char = np.sum(cigar_stats_primary)

        # IMPORTANT: When no alignments found, use default uniform distribution
        if n_char == 0:
            logger.warning("No valid CIGAR operations found in alignments, using default probabilities")
            # Use uniform distribution as a fallback
            remaining_ops = len(CIGAR_OPS) - len(zero_locs)
            return [math.log(1.0/max(1, remaining_ops)) for _ in range(max(1, remaining_ops))], zero_locs, dict_longest_align

        # Vectorized computation of log probabilities
        probs = cigar_stats_primary / n_char
        log_probs = np.log(probs).tolist()

        return log_probs, zero_locs, dict_longest_align

    except Exception as e:
        raise AlignmentError(f"Error calculating CIGAR operation probabilities: {str(e)}")

def compute_log_prob_rgs(
    alignment: pysam.AlignedSegment,
    cigar_stats: List[int],
    log_p_cigar_op: List[float],
    dict_longest_align: Dict[str, int],
    align_len: int
) -> Tuple[float, str, int]:
    """
    Calculate log(L(r|s)) = log(P(cigar_op)) × n_cigar_op for CIGAR_OPS.

    Args:
        alignment: Pysam alignment to score
        cigar_stats: List of cigar stats to compute
        log_p_cigar_op: List of cigar_op probabilities
        dict_longest_align: Dict of max alignment length for each query read
        align_len: Number of columns in the alignment

    Returns:
        Tuple containing:
            - log_score: log(L(r|s))
            - query_name: Query name in alignment
            - species_tid: Species-level taxonomy id

    Raises:
        AlignmentError: If there's an issue with the alignment
    """
    try:
        ref_name, query_name = alignment.reference_name, alignment.query_name

        if align_len == 0:
            raise AlignmentError(f"Alignment length is zero for {query_name}")

        # Vectorized computation of log score
        cigar_stats_np = np.array(cigar_stats)
        log_p_cigar_op_np = np.array(log_p_cigar_op)
        log_score = np.sum(log_p_cigar_op_np * cigar_stats_np) * \
                    (dict_longest_align[query_name]/align_len)

        species_tid = int(ref_name.split(":")[0])

        return log_score, query_name, species_tid

    except Exception as e:
        raise AlignmentError(f"Error computing log probability: {str(e)}")

def log_prob_rgs_dict(
    sam_path: Path,
    log_p_cigar_op: List[float],
    dict_longest_align: Dict[str, int],
    p_cigar_op_zero_locs: Optional[List[int]] = None
) -> Tuple[Dict[str, Tuple[List[int], List[float]]], Set[str], Set[str]]:
    """
    Create dict containing log(L(read|seq)) for all pairwise alignments in sam file.

    Args:
        sam_path: Path to sam file
        log_p_cigar_op: Probability for each cigar operation
        dict_longest_align: Dict of max alignment length for each query read
        p_cigar_op_zero_locs: List of indices where probability == 0

    Returns:
        Tuple containing:
            - Dict mapping query_name to ([tax_ids], [log_scores])
            - Set of unmapped read names
            - Set of mapped read names

    Raises:
        AlignmentError: If there's an issue with the alignments
    """
    try:
        # Convert to numpy arrays for vectorized operations
        log_p_cigar_op_np = np.array(log_p_cigar_op)

        # Calculate log(L(read|seq)) for all alignments
        log_p_rgs = {}
        unmapped_set = set()

        # Open SAM file
        sam_filename = pysam.AlignmentFile(str(sam_path), 'rb')

        # Process based on whether there are zero probability locations
        if not p_cigar_op_zero_locs:
            for alignment in sam_filename.fetch():
                align_len = get_align_len(alignment)
                if alignment.reference_name and align_len:
                    cigar_stats = get_align_stats(alignment)
                    log_score, query_name, species_tid = compute_log_prob_rgs(
                        alignment, cigar_stats, log_p_cigar_op,
                        dict_longest_align, align_len
                    )

                    if query_name not in log_p_rgs:
                        log_p_rgs[query_name] = ([species_tid], [log_score])
                    elif query_name in log_p_rgs:
                        if species_tid not in log_p_rgs[query_name][0]:
                            log_p_rgs[query_name] = (
                                log_p_rgs[query_name][0] + [species_tid],
                                log_p_rgs[query_name][1] + [log_score]
                            )
                        else:
                            logprgs_idx = log_p_rgs[query_name][0].index(species_tid)
                            if log_p_rgs[query_name][1][logprgs_idx] < log_score:
                                log_p_rgs[query_name][1][logprgs_idx] = log_score

                else:
                    unmapped_set.add(alignment.query_name)
        else:
            # Convert zero locations to numpy array for vectorized filtering
            p_cigar_op_zero_locs_np = np.array(p_cigar_op_zero_locs)

            for alignment in sam_filename.fetch():
                align_len = get_align_len(alignment)
                if alignment.reference_name and align_len:
                    cigar_stats = np.array(get_align_stats(alignment))

                    # Vectorized check for zero operations
                    if np.sum(cigar_stats[p_cigar_op_zero_locs_np]) == 0:
                        # Vectorized deletion of zero locations
                        filtered_cigar_stats = np.delete(cigar_stats, p_cigar_op_zero_locs_np)

                        log_score, query_name, species_tid = compute_log_prob_rgs(
                            alignment, filtered_cigar_stats.tolist(), log_p_cigar_op,
                            dict_longest_align, align_len
                        )

                        if query_name not in log_p_rgs:
                            log_p_rgs[query_name] = ([species_tid], [log_score])
                        elif query_name in log_p_rgs and species_tid not in log_p_rgs[query_name][0]:
                            log_p_rgs[query_name] = (
                                log_p_rgs[query_name][0] + [species_tid],
                                log_p_rgs[query_name][1] + [log_score]
                            )
                        else:
                            logprgs_idx = log_p_rgs[query_name][0].index(species_tid)
                            if log_p_rgs[query_name][1][logprgs_idx] < log_score:
                                log_p_rgs[query_name][1][logprgs_idx] = log_score
                else:
                    unmapped_set.add(alignment.query_name)

        # Calculate mapped and unmapped sets
        mapped_set = set(log_p_rgs.keys())
        unmapped_set = unmapped_set - mapped_set
        unmapped_count = len(unmapped_set)

        logger.info(f"Unmapped read count: {unmapped_count}")

        return log_p_rgs, unmapped_set, mapped_set

    except Exception as e:
        raise AlignmentError(f"Error building log probability dictionary: {str(e)}")

def expectation_maximization_vectorized(
    log_p_rgs: Dict[str, Tuple[List[int], List[float]]],
    freq: Dict[int, float],
    max_iterations: int = 100,
    epsilon: float = 1e-6
) -> Tuple[Dict[int, float], float, Dict[int, Dict[str, float]]]:
    """
    Vectorized implementation of one iteration of the EM algorithm.

    Updates the relative abundance estimation in freq based on probabilities in log_p_rgs.

    Args:
        log_p_rgs: Dict mapping query_name to ([tax_ids], [log_scores])
        freq: Dict mapping species_tax_id to likelihood
        max_iterations: Maximum number of EM iterations within a single call
        epsilon: Convergence threshold for relative change in frequencies

    Returns:
        Tuple containing:
            - Updated frequency dict
            - Total log likelihood
            - Dict mapping species_tax_id to {read_id: probability}

    Raises:
        CalculationError: If there's an issue with the calculations
    """
    try:
        # Set up data structures for sparse matrices
        all_reads = list(log_p_rgs.keys())
        all_taxa = list(freq.keys())
        read_to_idx = {read: i for i, read in enumerate(all_reads)}
        taxon_to_idx = {taxon: i for i, taxon in enumerate(all_taxa)}

        # Create sparse matrices
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

        # Perform EM iterations until convergence or max iterations
        for iteration in range(max_iterations):
            # E-step: Calculate log probabilities
            # Add log frequencies to log probabilities
            log_joint = log_prob_matrix.copy()
            log_joint_data = log_joint.data

            # Add log frequencies to each column (taxon)
            for i in range(log_joint.indptr.size - 1):
                start, end = log_joint.indptr[i], log_joint.indptr[i+1]
                for j in range(start, end):
                    col = log_joint.indices[j]
                    log_joint_data[j] += log_freq_vector[col]

            # Calculate log-sum-exp for each row (read)
            row_max = np.zeros(n_reads)
            for i in range(log_joint.indptr.size - 1):
                if log_joint.indptr[i] < log_joint.indptr[i+1]:  # If row has any entries
                    start, end = log_joint.indptr[i], log_joint.indptr[i+1]
                    row_max[i] = max(log_joint_data[j] for j in range(start, end))

            # Calculate exp(log_joint - row_max) for each entry
            exp_joint = log_joint.copy()
            exp_joint_data = exp_joint.data
            for i in range(exp_joint.indptr.size - 1):
                if exp_joint.indptr[i] < exp_joint.indptr[i+1]:  # If row has any entries
                    start, end = exp_joint.indptr[i], exp_joint.indptr[i+1]
                    for j in range(start, end):
                        exp_joint_data[j] = np.exp(log_joint_data[j] - row_max[i])

            # Calculate row sums (normalization)
            row_sums = np.zeros(n_reads)
            for i in range(exp_joint.indptr.size - 1):
                if exp_joint.indptr[i] < exp_joint.indptr[i+1]:  # If row has any entries
                    start, end = exp_joint.indptr[i], exp_joint.indptr[i+1]
                    row_sums[i] = sum(exp_joint_data[j] for j in range(start, end))

            # Normalize to get posterior probabilities
            posterior = exp_joint.copy()
            posterior_data = posterior.data
            for i in range(posterior.indptr.size - 1):
                if posterior.indptr[i] < posterior.indptr[i+1] and row_sums[i] > 0:  # If row has any entries
                    start, end = posterior.indptr[i], posterior.indptr[i+1]
                    for j in range(start, end):
                        posterior_data[j] /= row_sums[i]

            # Calculate log-likelihood
            log_likelihood = 0.0
            for i in range(n_reads):
                if row_sums[i] > 0:
                    log_likelihood += np.log(row_sums[i]) + row_max[i]

            # M-step: Update frequencies
            new_freq_vector = np.zeros(n_taxa)
            for i in range(posterior.indptr.size - 1):
                if posterior.indptr[i] < posterior.indptr[i+1]:  # If row has any entries
                    start, end = posterior.indptr[i], posterior.indptr[i+1]
                    for j in range(start, end):
                        col = posterior.indices[j]
                        new_freq_vector[col] += posterior_data[j]

            # Normalize new frequencies
            new_freq_vector /= max(1, n_reads)

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
                p_sgr_flat[(taxon, read)] = posterior_data[j]

        # Convert flat dictionary to nested structure
        p_sgr = unflatten(p_sgr_flat)

        return frq, log_likelihood, p_sgr

    except Exception as e:
        raise CalculationError(f"Error in expectation maximization: {str(e)}")

def batch_expectation_maximization(
    log_p_rgs: Dict[str, Tuple[List[int], List[float]]],
    freq: Dict[int, float],
    batch_size: int = 1000
) -> Tuple[Dict[int, float], float, Dict[int, Dict[str, float]]]:
    """
    EM algorithm that processes reads in batches for large datasets.

    Args:
        log_p_rgs: Dict mapping query_name to ([tax_ids], [log_scores])
        freq: Dict mapping species_tax_id to likelihood
        batch_size: Number of reads to process in each batch

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
        if n_reads <= batch_size:
            return expectation_maximization_vectorized(log_p_rgs, freq)

        # Split reads into batches
        batch_count = (n_reads + batch_size - 1) // batch_size
        read_batches = np.array_split(all_reads, batch_count)

        logger.info(f"Processing {n_reads} reads in {batch_count} batches of ~{batch_size} reads each")

        # Initialize accumulators
        total_log_likelihood = 0.0
        p_sgr_combined = {}
        taxa_counts = defaultdict(float)

        # Process each batch
        for batch_idx, batch_reads in enumerate(read_batches):
            logger.debug(f"Processing batch {batch_idx+1}/{batch_count} ({len(batch_reads)} reads)")

            # Create batch subset of log_p_rgs
            batch_log_p_rgs = {read: log_p_rgs[read] for read in batch_reads}

            # Run EM on this batch
            _, batch_log_likelihood, batch_p_sgr = expectation_maximization_vectorized(
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
        total_taxa_count = sum(taxa_counts.values())
        if total_taxa_count == 0:
            raise CalculationError("No valid taxa counts after batch processing")

        updated_freq = {taxon: count / n_reads for taxon, count in taxa_counts.items()}

        # Ensure frequencies sum to 1
        freq_sum = sum(updated_freq.values())
        if not 0.99 <= freq_sum <= 1.01:
            logger.warning(f"Normalizing frequency vector from {freq_sum} to 1.0")
            updated_freq = {k: v / freq_sum for k, v in updated_freq.items()}

        return updated_freq, total_log_likelihood, p_sgr_combined

    except Exception as e:
        raise CalculationError(f"Error in batch expectation maximization: {str(e)}")

def expectation_maximization_iterations(
    log_p_rgs: Dict[str, Tuple[List[int], List[float]]],
    db_ids: List[int],
    lli_thresh: float,
    input_threshold: float,
    max_iterations: int = 100,
    rel_tol: float = 1e-4,
    batch_size: int = 1000
) -> Tuple[Dict[int, float], Optional[Dict[int, float]], Dict[int, Dict[str, float]]]:
    """
    Full expectation maximization algorithm with batching for large datasets.

    Args:
        log_p_rgs: Dict mapping query_name to ([tax_ids], [log_scores])
        db_ids: List of each unique species taxonomy id in database
        lli_thresh: Log likelihood increase minimum to continue EM iterations
        input_threshold: Minimum relative abundance in output
        max_iterations: Maximum number of iterations
        rel_tol: Relative tolerance for early stopping
        batch_size: Number of reads to process in each batch

    Returns:
        Tuple containing:
            - Dict mapping species_tax_id to estimated likelihood
            - Dict with only values above threshold, or None if not needed
            - Dict mapping species_tax_id to {read_id: probability}
    """
    try:
        n_db = len(db_ids)
        n_reads = len(log_p_rgs)

        logger.info(f"Mapped read count: {n_reads}")

        # Check if there are enough reads
        if n_reads == 0:
            raise CalculationError("0 reads mapped")

        # Determine whether to use batching based on dataset size
        use_batching = n_reads > batch_size
        em_function = batch_expectation_maximization if use_batching else expectation_maximization_vectorized

        if use_batching:
            logger.info(f"Using batch processing with batch size {batch_size} for {n_reads} reads")

        # Initialize frequency vector
        freq = dict.fromkeys(db_ids, 1 / n_db)

        # Keep track of frequency history for convergence detection
        freq_history = []
        counter = 1

        # Set output abundance threshold
        freq_thresh = 1/(n_reads + 1)
        if n_reads > 1000:
            freq_thresh = 10/n_reads

        # Perform iterations of the expectation_maximization algorithm
        total_log_likelihood = -math.inf

        # Early stopping variables
        consecutive_small_improvements = 0
        required_consecutive_small_improvements = 3  # Number of consecutive small improvements to trigger early stopping

        while counter <= max_iterations:
            # Use the appropriate EM implementation based on dataset size
            if use_batching:
                freq, updated_log_likelihood, _ = batch_expectation_maximization(
                    log_p_rgs, freq, batch_size=batch_size
                )
            else:
                freq, updated_log_likelihood, _ = expectation_maximization_vectorized(
                    log_p_rgs, freq, max_iterations=5, epsilon=1e-6
                )

            # Check f vector sums to 1
            freq_sum = sum(freq.values())
            if not .9 <= freq_sum <= 1.1:
                # Normalize if slightly off
                freq = {k: v/freq_sum for k, v in freq.items()}
                logger.warning(f"Normalized frequency vector from {freq_sum} to 1.0")

            # Track frequency history for convergence detection
            freq_history.append(dict(freq))

            # Confirm log likelihood increase
            log_likelihood_diff = updated_log_likelihood - total_log_likelihood

            # Handle the case when total_log_likelihood is -inf in the first iteration
            if counter == 1 or math.isinf(abs(total_log_likelihood)):
                rel_improvement = 1.0  # Set a default value for the first iteration
            else:
                rel_improvement = log_likelihood_diff / abs(total_log_likelihood)

            logger.info(f"EM Iteration {counter}: log-likelihood = {updated_log_likelihood:.4f}, "
                        f"change = {log_likelihood_diff:.6f}, relative = {rel_improvement:.6f}")

            total_log_likelihood = updated_log_likelihood

            if log_likelihood_diff < 0:
                logger.warning("Log likelihood decreased, reverting to previous iteration")
                freq = freq_history[-2]  # Revert to previous frequency
                break

            # Check for early stopping based on small relative improvements
            if rel_improvement < rel_tol:
                consecutive_small_improvements += 1
                if consecutive_small_improvements >= required_consecutive_small_improvements:
                    logger.info(f"Early stopping triggered after {counter} iterations due to small relative improvements")
                    break
            else:
                consecutive_small_improvements = 0

            # Exit loop if log likelihood increase less than threshold
            if log_likelihood_diff < lli_thresh:
                logger.info(f"Stopping after {counter} iterations: log likelihood increase below threshold")
                break

            # Check for convergence in frequency vectors
            if counter >= 3:
                # Calculate maximum relative change in frequencies
                prev_freq = freq_history[-2]
                max_rel_change = max(
                    abs(freq.get(k, 0) - prev_freq.get(k, 0)) / (prev_freq.get(k, 1e-10))
                    for k in set(freq) | set(prev_freq)
                )

                if max_rel_change < rel_tol:
                    logger.info(f"Early stopping triggered after {counter} iterations due to frequency convergence")
                    break

            counter += 1

        logger.info(f"Total number of EM iterations: {counter}")

        # Remove tax id if less than the frequency threshold
        freq = {k: v for k, v in freq.items() if v >= freq_thresh}

        # Final EM iteration with filtered frequencies
        if use_batching:
            freq_full, updated_log_likelihood, p_sgr = batch_expectation_maximization(
                log_p_rgs, freq, batch_size=batch_size
            )
        else:
            freq_full, updated_log_likelihood, p_sgr = expectation_maximization_vectorized(
                log_p_rgs, freq, max_iterations=10, epsilon=1e-8
            )

        # Apply threshold if needed
        freq_set_thresh = None
        if freq_thresh < input_threshold:
            freq_filtered = {k: v for k, v in freq_full.items() if v >= input_threshold}
            if freq_filtered:  # Only calculate if there are species above threshold
                if use_batching:
                    freq_set_thresh, _, _ = batch_expectation_maximization(
                        log_p_rgs, freq_filtered, batch_size=batch_size
                    )
                else:
                    freq_set_thresh, _, _ = expectation_maximization_vectorized(
                        log_p_rgs, freq_filtered, max_iterations=10, epsilon=1e-8
                    )

        return freq_full, freq_set_thresh, p_sgr

    except Exception as e:
        raise CalculationError(f"Error in expectation maximization iterations: {str(e)}")

def expectation_maximization(
    log_p_rgs: Dict[str, Tuple[List[int], List[float]]],
    freq: Dict[int, float]
) -> Tuple[Dict[int, float], float, Dict[int, Dict[str, float]]]:
    """
    One iteration of the EM algorithm.

    Updates the relative abundance estimation in freq based on probabilities in log_p_rgs.

    Args:
        log_p_rgs: Dict mapping query_name to ([tax_ids], [log_scores])
        freq: Dict mapping species_tax_id to likelihood

    Returns:
        Tuple containing:
            - Updated frequency dict
            - Total log likelihood
            - Dict mapping species_tax_id to {read_id: probability}

    Raises:
        CalculationError: If there's an issue with the calculations
    """
    # For backwards compatibility, we'll use the vectorized implementation
    return expectation_maximization_vectorized(log_p_rgs, freq)

def inspect_sam_file(sam_path):
    """Inspect the first few lines of a SAM file for debugging."""
    logger.debug(f"Inspecting SAM file: {sam_path}")
    try:
        with open(sam_path, 'r') as f:
            header_lines = 0
            for line in f:
                if line.startswith('@'):
                    header_lines += 1
                else:
                    logger.debug(f"First alignment line: {line.strip()}")
                    break
            logger.debug(f"SAM file has {header_lines} header lines")

            f.seek(0)
            alignment_count = sum(1 for line in f if not line.startswith('@'))
            logger.debug(f"SAM file has {alignment_count} alignment lines")
    except Exception as e:
        logger.error(f"Error inspecting SAM file: {str(e)}")

def generate_alignments(
    in_file_list: List[Path],
    out_basename: Path,
    database: Path,
    seq_type: str = 'map-ont',
    threads: int = 3,
    N: int = 50,
    K: int = 500000000,
    mm2_forward_only: bool = False
) -> Path:
    """
    Generate .sam alignment file using minimap2.

    Args:
        in_file_list: List of paths to input sequences
        out_basename: Path and basename for output files
        database: Path to database directory
        seq_type: Sequencing type for minimap2
        threads: Number of threads for minimap2
        N: Max number of secondary alignments for minimap2
        K: Minibatch size for minimap2
        mm2_forward_only: Whether to use forward transcript strand only

    Returns:
        Path to SAM alignment file

    Raises:
        AlignmentError: If there's an issue generating alignments
    """
    try:
        # Convert paths to strings for command building
        input_files_str = " ".join(str(f) for f in in_file_list)

        # Check if input is already a SAM file
        if in_file_list[0].suffix == '.sam':
            return in_file_list[0]

        # Create output SAM file path
        sam_align_file = f"{out_basename}_emu_alignments.sam"
        db_sequence_file = database / 'species_taxid.fasta'

        # Build minimap2 command
        if mm2_forward_only:
            cmd = (
                f"minimap2 -ax {seq_type} -t {threads} -N {N} -p .9 -u f -K {K} "
                f"{db_sequence_file} {input_files_str} -o {sam_align_file}"
            )
        else:
            cmd = (
                f"minimap2 -ax {seq_type} -t {threads} -N {N} -p .9 -K {K} "
                f"{db_sequence_file} {input_files_str} -o {sam_align_file}"
            )

        # Run minimap2
        logger.info(f"Running alignment with command: {cmd}")
        subprocess.check_output(cmd, shell=True)

        # Inspect the SAM file for debugging
        inspect_sam_file(sam_align_file)

        return Path(sam_align_file)

    except subprocess.CalledProcessError as e:
        raise AlignmentError(f"Error running minimap2: {str(e)}")
    except Exception as e:
        raise AlignmentError(f"Error generating alignments: {str(e)}")