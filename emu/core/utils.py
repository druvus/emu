"""Utility functions for Emu."""

import gzip
import math
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional, Union, Iterator, Any, BinaryIO

import numpy as np
import pysam
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from emu.models.errors import InputError, AlignmentError

# Logger configuration
logger = logging.getLogger(__name__)

# Static global variables
CIGAR_OPS = [1, 2, 4, 10]
CIGAR_OPS_ALL = [0, 1, 2, 4]
TAXONOMY_RANKS = ['species', 'genus', 'family', 'order', 'class', 'phylum', 'clade', 'superkingdom']
RANKS_PRINTOUT = ['tax_id'] + TAXONOMY_RANKS + ['subspecies', 'species subgroup', 'species group']
RANKS_ORDER = ['tax_id'] + TAXONOMY_RANKS[:6] + TAXONOMY_RANKS[7:]

def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    Configure logging for the Emu application.

    Args:
        verbose: Whether to enable verbose (DEBUG) logging

    Returns:
        Logger instance
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger('emu')

def validate_input(path: Path) -> None:
    """
    Validate input file is either: fasta, fastq, or sam alignment file.

    Args:
        path: Path to input file

    Raises:
        InputError: If the input file is not in a valid format
    """
    # Check if input is a SAM file
    sam_pysam = None
    try:
        sam_pysam = pysam.AlignmentFile(str(path))
    except (ValueError, OSError):
        pass
    if sam_pysam:
        return

    # Check if input is a FASTA/Q file
    fasta_rd, fastq_rd = None, None
    try:
        fasta_rd = SeqIO.to_dict(SeqIO.parse(str(path), "fasta"))
        fastq_rd = SeqIO.to_dict(SeqIO.parse(str(path), "fastq"))
    except (UnicodeDecodeError, ValueError):
        pass
    if not (fasta_rd or fastq_rd):
        raise InputError(f"Input file '{path}' must be in a valid format: fasta, fastq, or sam")

def get_align_stats(alignment: pysam.AlignedSegment) -> List[int]:
    """
    Retrieve list of inquired cigar stats (I,D,S,X) for alignment.

    Args:
        alignment: Alignment of interest

    Returns:
        List of counts for each cigar operation defined in (I,D,S,X)
    """
    cigar_stats = alignment.get_cigar_stats()[0]
    n_mismatch = cigar_stats[10] - cigar_stats[1] - cigar_stats[2]
    return [cigar_stats[1], cigar_stats[2], cigar_stats[4], n_mismatch]

def get_align_len(alignment: pysam.AlignedSegment) -> int:
    """
    Retrieve number of columns in alignment.

    Args:
        alignment: Alignment of interest

    Returns:
        Number of columns in alignment
    """
    return sum(alignment.get_cigar_stats()[0][cigar_op] for cigar_op in CIGAR_OPS_ALL)