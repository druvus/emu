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
from emu.core.jit_utils import NUMBA_AVAILABLE, sum_array
from emu.core.matrix_ops import (
    compute_row_max, compute_exp_and_sum, compute_posterior_and_freqs,
    parallel_compute_posterior, compute_log_joint
)
from emu.core.em_strategy import (
    create_em_strategy, VectorizedEMStrategy, BatchEMStrategy, MemoryEfficientEMStrategy
)
from emu.core.batch_processing import (
    BatchProcessor, AlignmentBatchProcessor, ReadBatchProcessor
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

        # Use batch processing for alignments
        alignment_processor = AlignmentBatchProcessor(batch_size=10000, show_progress=True)

        # Create batched processing function
        def process_batch(alignments):
            batch_cigar_stats = np.zeros(len(CIGAR_OPS), dtype=np.int32)
            batch_longest_align = {}
            primary_count = 0

            for alignment in alignments:
                align_len = get_align_len(alignment)

                # Track longest alignment per read
                query_name = alignment.query_name
                batch_longest_align[query_name] = max(
                    batch_longest_align.get(query_name, 0),
                    align_len
                )

                # Process primary alignments only
                if (not alignment.is_secondary and
                    not alignment.is_supplementary and
                    alignment.reference_name):
                    primary_count += 1
                    current_stats = np.array(get_align_stats(alignment), dtype=np.int32)
                    batch_cigar_stats += current_stats

            return batch_cigar_stats, batch_longest_align, primary_count

        # Process all alignments in batches
        total_alignments = 0
        primary_alignments = 0

        for batch in alignment_processor.process(sam_pysam.fetch(), lambda x: [x]):
            batch_stats, batch_longest, batch_primary = process_batch(batch)
            cigar_stats_primary += batch_stats
            dict_longest_align.update(batch_longest)
            primary_alignments += batch_primary
            total_alignments += len(batch)

        logger.debug(f"Total alignments: {total_alignments}, Primary alignments: {primary_alignments}")

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

        # Compute log score
        if NUMBA_AVAILABLE:
            # Use JIT-optimized dot product
            cigar_stats_np = np.array(cigar_stats, dtype=np.float64)
            log_p_cigar_op_np = np.array(log_p_cigar_op, dtype=np.float64)
            log_score = np.dot(cigar_stats_np, log_p_cigar_op_np) * \
                        (dict_longest_align[query_name]/align_len)
        else:
            # Fallback to standard dot product
            log_score = sum(c * p for c, p in zip(cigar_stats, log_p_cigar_op)) * \
                        (dict_longest_align[query_name]/align_len)

        # Extract species taxonomy ID from reference name
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
        log_p_cigar_op_np = np.array(log_p_cigar_op, dtype=np.float64)

        # Initialize data structures
        log_p_rgs = {}
        unmapped_set = set()

        # Open SAM file
        sam_filename = pysam.AlignmentFile(str(sam_path), 'rb')

        # Use batch processing
        alignment_processor = AlignmentBatchProcessor(batch_size=5000, show_progress=True)

        # Create batch processing function
        def process_log_prob_batch(batch_alignments):
            batch_log_p_rgs = {}
            batch_unmapped = set()

            # Process zero locations case separately for efficiency
            if not p_cigar_op_zero_locs:
                for alignment in batch_alignments:
                    align_len = get_align_len(alignment)
                    if alignment.reference_name and align_len:
                        cigar_stats = get_align_stats(alignment)
                        log_score, query_name, species_tid = compute_log_prob_rgs(
                            alignment, cigar_stats, log_p_cigar_op_np,
                            dict_longest_align, align_len
                        )

                        if query_name not in batch_log_p_rgs:
                            batch_log_p_rgs[query_name] = ([species_tid], [log_score])
                        elif species_tid not in batch_log_p_rgs[query_name][0]:
                            batch_log_p_rgs[query_name] = (
                                batch_log_p_rgs[query_name][0] + [species_tid],
                                batch_log_p_rgs[query_name][1] + [log_score]
                            )
                        else:
                            logprgs_idx = batch_log_p_rgs[query_name][0].index(species_tid)
                            if batch_log_p_rgs[query_name][1][logprgs_idx] < log_score:
                                batch_log_p_rgs[query_name][1][logprgs_idx] = log_score
                    else:
                        batch_unmapped.add(alignment.query_name)

            else:
                # Use vectorized operations when handling zero locations
                p_cigar_op_zero_locs_np = np.array(p_cigar_op_zero_locs, dtype=np.int32)

                for alignment in batch_alignments:
                    align_len = get_align_len(alignment)
                    if alignment.reference_name and align_len:
                        cigar_stats = np.array(get_align_stats(alignment), dtype=np.int32)

                        # Vectorized check for zero operations
                        if np.sum(cigar_stats[p_cigar_op_zero_locs_np]) == 0:
                            # Vectorized deletion of zero locations
                            filtered_cigar_stats = np.delete(cigar_stats, p_cigar_op_zero_locs_np)

                            log_score, query_name, species_tid = compute_log_prob_rgs(
                                alignment, filtered_cigar_stats.tolist(), log_p_cigar_op_np,
                                dict_longest_align, align_len
                            )

                            if query_name not in batch_log_p_rgs:
                                batch_log_p_rgs[query_name] = ([species_tid], [log_score])
                            elif species_tid not in batch_log_p_rgs[query_name][0]:
                                batch_log_p_rgs[query_name] = (
                                    batch_log_p_rgs[query_name][0] + [species_tid],
                                    batch_log_p_rgs[query_name][1] + [log_score]
                                )
                            else:
                                logprgs_idx = batch_log_p_rgs[query_name][0].index(species_tid)
                                if batch_log_p_rgs[query_name][1][logprgs_idx] < log_score:
                                    batch_log_p_rgs[query_name][1][logprgs_idx] = log_score
                    else:
                        batch_unmapped.add(alignment.query_name)

            return batch_log_p_rgs, batch_unmapped

        # Process all alignments in batches
        processed_count = 0

        for batch in alignment_processor.process(sam_filename.fetch(), lambda x: [x]):
            batch_log_p_rgs, batch_unmapped = process_log_prob_batch(batch)

            # Merge batch results into overall results
            log_p_rgs.update(batch_log_p_rgs)
            unmapped_set.update(batch_unmapped)

            processed_count += len(batch)

            # Log progress for very large files
            if processed_count % 100000 == 0:
                logger.info(f"Processed {processed_count} alignments")

        # Calculate mapped and unmapped sets
        mapped_set = set(log_p_rgs.keys())
        unmapped_set = unmapped_set - mapped_set
        unmapped_count = len(unmapped_set)

        logger.info(f"Unmapped read count: {unmapped_count}")

        return log_p_rgs, unmapped_set, mapped_set

    except Exception as e:
        raise AlignmentError(f"Error building log probability dictionary: {str(e)}")

def expectation_maximization_iterations(
    log_p_rgs: Dict[str, Tuple[List[int], List[float]]],
    db_ids: List[int],
    lli_thresh: float,
    input_threshold: float,
    max_iterations: int = 100,
    rel_tol: float = 1e-4,
    batch_size: int = 1000,
    em_strategy: str = "auto",
    parallel: bool = True
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
        em_strategy: Strategy to use for EM algorithm ("auto", "vectorized", "batch", "memory_efficient")
        parallel: Whether to use parallel processing when available

    Returns:
        Tuple containing:
            - Dict mapping species_tax_id to estimated likelihood
            - Dict with only values above threshold, or None if not needed
            - Dict mapping species_tax_id to {read_id: probability}
    """
    try:
        from emu.core.em_strategy import create_em_strategy

        n_db = len(db_ids)
        n_reads = len(log_p_rgs)

        logger.info(f"Mapped read count: {n_reads}")

        # Check if there are enough reads
        if n_reads == 0:
            raise CalculationError("0 reads mapped")

        # Determine appropriate EM strategy if set to auto
        if em_strategy == "auto":
            if n_reads > 5000:
                # Use batch processing for large datasets
                strategy_type = "batch"
                logger.info(f"Using batch EM strategy with batch size {batch_size}")
            elif n_reads > 1000:
                # Use memory-efficient processing for medium datasets
                strategy_type = "memory_efficient"
                logger.info("Using memory-efficient EM strategy")
            else:
                # Use vectorized processing for small datasets
                strategy_type = "vectorized"
                logger.info("Using vectorized EM strategy")
        else:
            strategy_type = em_strategy
            logger.info(f"Using specified EM strategy: {strategy_type}")

        # Create EM strategy with appropriate parameters
        strategy = create_em_strategy(
            strategy_type=strategy_type,
            batch_size=batch_size,
            parallel=parallel
        )

        # Initialize frequency vector
        freq = dict.fromkeys(db_ids, 1 / n_db)

        # Keep track of frequency history (efficiently)
        freq_changes = []
        counter = 1

        # Set output abundance threshold
        freq_thresh = 1/(n_reads + 1)
        if n_reads > 1000:
            freq_thresh = 10/n_reads

        # Perform iterations of the expectation_maximization algorithm
        total_log_likelihood = -math.inf

        # Early stopping variables
        consecutive_small_improvements = 0
        required_consecutive_small_improvements = 3

        while counter <= max_iterations:
            # Run EM iteration with appropriate strategy
            freq, updated_log_likelihood, _ = strategy.run_em(
                log_p_rgs, freq, max_iterations=5, epsilon=1e-6
            )

            # Check f vector sums to 1
            freq_sum = sum(freq.values())
            if not .9 <= freq_sum <= 1.1:
                # Normalize if slightly off
                freq = {k: v/freq_sum for k, v in freq.items()}
                logger.warning(f"Normalized frequency vector from {freq_sum} to 1.0")

            # Track frequency changes efficiently
            if len(freq_changes) > 0:
                # Only store significant changes
                changes = {}
                for k, v in freq.items():
                    prev_v = freq_changes[-1].get(k, 0)
                    if abs(v - prev_v) > rel_tol * max(1e-10, prev_v):
                        changes[k] = v
                freq_changes.append(changes)
            else:
                freq_changes.append(dict(freq))

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
                # Reconstruct full frequency dict from changes
                freq = dict(freq_changes[-2])
                for i in range(len(freq_changes) - 3, -1, -1):
                    freq.update(freq_changes[i])
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
                prev_freq = {}
                for i in range(len(freq_changes) - 2, -1, -1):
                    prev_changes = freq_changes[i]
                    for k, v in prev_changes.items():
                        if k not in prev_freq:
                            prev_freq[k] = v

                # Calculate max relative change only for significant taxa
                significant_taxa = set(freq) | set(prev_freq)
                if significant_taxa:
                    max_rel_change = max(
                        abs(freq.get(k, 0) - prev_freq.get(k, 0)) / max(1e-10, prev_freq.get(k, 0))
                        for k in significant_taxa
                    )

                    if max_rel_change < rel_tol:
                        logger.info(f"Early stopping triggered after {counter} iterations due to frequency convergence")
                        break

            counter += 1

        logger.info(f"Total number of EM iterations: {counter}")

        # Remove tax id if less than the frequency threshold
        freq = {k: v for k, v in freq.items() if v >= freq_thresh}

        # Final EM iteration with filtered frequencies
        freq_full, updated_log_likelihood, p_sgr = strategy.run_em(
            log_p_rgs, freq, max_iterations=10, epsilon=1e-8
        )

        # Apply threshold if needed
        freq_set_thresh = None
        if freq_thresh < input_threshold:
            freq_filtered = {k: v for k, v in freq_full.items() if v >= input_threshold}
            if freq_filtered:  # Only calculate if there are species above threshold
                freq_set_thresh, _, _ = strategy.run_em(
                    log_p_rgs, freq_filtered, max_iterations=10, epsilon=1e-8
                )

        return freq_full, freq_set_thresh, p_sgr

    except Exception as e:
        raise CalculationError(f"Error in expectation maximization iterations: {str(e)}")

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

        # Build minimap2 command with optimized parameters
        base_cmd = (
            f"minimap2 -ax {seq_type} -t {threads} -N {N} -p .9 -K {K} "
        )

        if mm2_forward_only:
            base_cmd += "-u f "

        # Add input and output files
        cmd = f"{base_cmd} {db_sequence_file} {input_files_str} -o {sam_align_file}"

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