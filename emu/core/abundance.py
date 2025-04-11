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

from emu.models.errors import AlignmentError, CalculationError
from emu.core.utils import (
    get_align_stats, get_align_len, CIGAR_OPS, CIGAR_OPS_ALL,
    TAXONOMY_RANKS, RANKS_PRINTOUT
)

logger = logging.getLogger(__name__)

def get_cigar_op_log_probabilities(
    sam_path: Path
) -> Tuple[List[float], List[int], Dict[str, int]]:
    """Calculate log probabilities for each CIGAR operation."""
    try:
        cigar_stats_primary = [0] * len(CIGAR_OPS)
        dict_longest_align = {}

        # Open SAM file
        sam_pysam = pysam.AlignmentFile(str(sam_path))

        alignment_count = 0
        primary_alignment_count = 0

        # Process alignments
        for alignment in sam_pysam.fetch():
            alignment_count += 1
            align_len = get_align_len(alignment)

            # Track longest alignment per read
            if alignment.query_name not in dict_longest_align:
                dict_longest_align[alignment.query_name] = align_len
            elif dict_longest_align[alignment.query_name] < align_len:
                dict_longest_align[alignment.query_name] = align_len

            # Process primary alignments only
            if (not alignment.is_secondary and
                not alignment.is_supplementary and
                alignment.reference_name):
                primary_alignment_count += 1
                current_stats = get_align_stats(alignment)
                cigar_stats_primary = list(map(add, cigar_stats_primary, current_stats))

        logger.debug(f"Total alignments: {alignment_count}, Primary alignments: {primary_alignment_count}")

        # Check if any probabilities are 0, if so, remove
        zero_locs = [i for i, e in enumerate(cigar_stats_primary) if e == 0]
        if zero_locs:
            for i in sorted(zero_locs, reverse=True):
                del cigar_stats_primary[i]

        # Calculate log probabilities
        n_char = sum(cigar_stats_primary)

        # IMPORTANT: When no alignments found, use default uniform distribution
        if n_char == 0:
            logger.warning("No valid CIGAR operations found in alignments, using default probabilities")
            # Use uniform distribution as a fallback
            remaining_ops = len(CIGAR_OPS) - len(zero_locs)
            return [math.log(1.0/max(1, remaining_ops)) for _ in range(max(1, remaining_ops))], zero_locs, dict_longest_align

        log_probs = [math.log(x) for x in np.array(cigar_stats_primary)/n_char]
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

        log_score = sum(list(map(mul, log_p_cigar_op, cigar_stats))) * \
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
            # Handle case with zero probability locations
            for alignment in sam_filename.fetch():
                align_len = get_align_len(alignment)
                if alignment.reference_name and align_len:
                    cigar_stats = get_align_stats(alignment)
                    if sum(cigar_stats[x] for x in p_cigar_op_zero_locs) == 0:
                        for i in sorted(p_cigar_op_zero_locs, reverse=True):
                            del cigar_stats[i]
                        log_score, query_name, species_tid = compute_log_prob_rgs(
                            alignment, cigar_stats, log_p_cigar_op,
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
    try:
        p_sgr_flat = {}
        logpr_sum, n_reads = 0, 0

        for read in log_p_rgs:
            valid_seqs, log_p_rns = [], []
            # Check if sequences were found in frequency vector
            for seq in range(len(log_p_rgs[read][0])):
                s_val = log_p_rgs[read][0][seq]
                if s_val in freq and freq[s_val] != 0:
                    # Calculate log(L(r|s))+log(f(s)) for each sequence
                    logprns_val = log_p_rgs[read][1][seq] + math.log(freq[s_val])
                    valid_seqs.append(s_val)
                    log_p_rns.append(logprns_val)

            if len(valid_seqs) != 0:
                # Calculate fixed multiplier, c
                logc = -np.max(log_p_rns)
                # Calculate exp(log(L(r|s) * f(s) * c))
                prnsc = np.exp(log_p_rns + logc)
                # Calculate sum of (L(r|s) * f(s) * c) for each read
                prc = np.sum(prnsc)
                # Add to sum of log likelihood
                logpr_sum += (np.log(prc) - logc)
                n_reads += 1

                # Calculate P(s|r) for each sequence
                for seq in enumerate(valid_seqs):
                    p_sgr_flat[(seq[1], read)] = prnsc[seq[0]] / prc

        # Convert flat dictionary to nested structure
        p_sgr = unflatten(p_sgr_flat)

        # Calculate updated frequency vector
        if n_reads == 0:
            raise CalculationError("No valid reads for expectation maximization")

        frq = {tax_id: sum(read_id.values()) / n_reads
               for tax_id, read_id in p_sgr.items()}

        return frq, logpr_sum, p_sgr

    except Exception as e:
        raise CalculationError(f"Error in expectation maximization: {str(e)}")

def expectation_maximization_iterations(
    log_p_rgs: Dict[str, Tuple[List[int], List[float]]],
    db_ids: List[int],
    lli_thresh: float,
    input_threshold: float
) -> Tuple[Dict[int, float], Optional[Dict[int, float]], Dict[int, Dict[str, float]]]:
    """
    Full expectation maximization algorithm for alignments.

    Args:
        log_p_rgs: Dict mapping query_name to ([tax_ids], [log_scores])
        db_ids: List of each unique species taxonomy id in database
        lli_thresh: Log likelihood increase minimum to continue EM iterations
        input_threshold: Minimum relative abundance in output

    Returns:
        Tuple containing:
            - Dict mapping species_tax_id to estimated likelihood
            - Dict with only values above threshold, or None if not needed
            - Dict mapping species_tax_id to {read_id: probability}

    Raises:
        CalculationError: If there's an issue with the calculations
    """
    try:
        n_db = len(db_ids)
        n_reads = len(log_p_rgs)

        logger.info(f"Mapped read count: {n_reads}")

        # Check if there are enough reads
        if n_reads == 0:
            raise CalculationError("0 reads mapped")

        # Initialize frequency vector
        freq = dict.fromkeys(db_ids, 1 / n_db)
        counter = 1

        # Set output abundance threshold
        freq_thresh = 1/(n_reads + 1)
        if n_reads > 1000:
            freq_thresh = 10/n_reads

        # Perform iterations of the expectation_maximization algorithm
        total_log_likelihood = -math.inf

        while True:
            freq, updated_log_likelihood, _ = expectation_maximization(log_p_rgs, freq)

            # Check f vector sums to 1
            freq_sum = sum(freq.values())
            if not .9 <= freq_sum <= 1.1:
                raise CalculationError(f"Frequency vector sums to {freq_sum}, rather than 1")

            # Confirm log likelihood increase
            log_likelihood_diff = updated_log_likelihood - total_log_likelihood
            total_log_likelihood = updated_log_likelihood

            if log_likelihood_diff < 0:
                raise CalculationError("Total log likelihood decreased from prior iteration")

            # Exit loop if log likelihood increase less than threshold
            if log_likelihood_diff < lli_thresh:
                logger.info(f"Number of EM iterations: {counter}")

                # Remove tax id if less than the frequency threshold
                freq = {k: v for k, v in freq.items() if v >= freq_thresh}

                freq_full, updated_log_likelihood, p_sgr = expectation_maximization(log_p_rgs, freq)

                freq_set_thresh = None
                if freq_thresh < input_threshold:
                    freq = {k: v for k, v in freq_full.items() if v >= input_threshold}
                    freq_set_thresh, updated_log_likelihood, p_sgr = \
                        expectation_maximization(log_p_rgs, freq)

                return freq_full, freq_set_thresh, p_sgr

            counter += 1

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