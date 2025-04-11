"""File format parsers for Emu."""

import gzip
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional, Union, Any, BinaryIO

import pandas as pd
from Bio import SeqIO

from emu.models.errors import InputError

logger = logging.getLogger(__name__)

def parse_taxonomy_file(taxonomy_path: Path) -> pd.DataFrame:
    """
    Parse taxonomy file into a pandas DataFrame.

    Args:
        taxonomy_path: Path to taxonomy.tsv file

    Returns:
        DataFrame containing taxonomy information indexed by tax_id

    Raises:
        InputError: If there's an issue with the taxonomy file
    """
    try:
        df_taxonomy = pd.read_csv(
            taxonomy_path,
            sep='\t',
            index_col='tax_id',
            dtype=str
        )
        return df_taxonomy
    except Exception as e:
        raise InputError(f"Error parsing taxonomy file: {str(e)}")

def output_sequences(
    in_path: Path,
    seq_output_path: Path,
    input_type: str,
    keep_ids: Set[str]
) -> None:
    """
    Output specified list of sequences from input_file based on sequence id.

    Args:
        in_path: Path to input fasta or fastq
        seq_output_path: Output path for fasta/q of unclassified sequences
        input_type: fasta or fastq
        keep_ids: Set of sequence id strings

    Raises:
        InputError: If there's an issue with the input or output files
    """
    try:
        # Determine how to open the file based on extension
        if str(in_path).endswith(".gz"):
            open_func = lambda p: gzip.open(p, "rt", encoding="utf-8")
        else:
            open_func = lambda p: open(p, "r", encoding="utf-8")

        # Create output file path
        out_file_path = f"{seq_output_path}.{input_type}"

        # Open input and output files
        with open_func(in_path) as in_file, \
                open(out_file_path, "w", encoding="utf-8") as out_seq_file:
            # Parse the FASTA/FASTQ file and filter by read IDs
            filtered_sequences = (seq for seq in SeqIO.parse(in_file, input_type)
                                if seq.id in keep_ids)

            # Write the filtered sequences to the output file
            count = SeqIO.write(filtered_sequences, out_seq_file, input_type)
            logger.info(f"Wrote {count} sequences to {out_file_path}")

    except Exception as e:
        raise InputError(f"Error outputting sequences: {str(e)}")