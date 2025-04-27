"""File format parsers for Emu."""

import gzip
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional, Union, Any, BinaryIO
from abc import ABC, abstractmethod

import pandas as pd
from Bio import SeqIO

from emu.models.errors import InputError

logger = logging.getLogger(__name__)

class Parser(ABC):
    """Base parser class for different file formats."""

    @abstractmethod
    def parse(self, path: Path):
        """Parse file at the given path.

        Args:
            path: Path to file

        Returns:
            Parsed data
        """
        pass

class TaxonomyParser(Parser):
    """Parser for taxonomy files."""

    def parse(self, path: Path) -> pd.DataFrame:
        """Parse taxonomy file into a pandas DataFrame.

        Args:
            path: Path to taxonomy.tsv file

        Returns:
            DataFrame containing taxonomy information indexed by tax_id

        Raises:
            InputError: If there's an issue with the taxonomy file
        """
        try:
            df_taxonomy = pd.read_csv(
                path,
                sep='\t',
                index_col='tax_id',
                dtype=str
            )
            return df_taxonomy
        except Exception as e:
            raise InputError(f"Error parsing taxonomy file: {str(e)}")

class SequenceParser(Parser):
    """Parser for sequence files (FASTA/FASTQ)."""

    def parse(self, path: Path, format: Optional[str] = None) -> Dict[str, Any]:
        """Parse FASTA/FASTQ file into a dictionary.

        Args:
            path: Path to sequence file
            format: Optional format type ('fasta' or 'fastq'). If None, inferred from extension.

        Returns:
            Dictionary mapping sequence IDs to SeqRecord objects

        Raises:
            InputError: If there's an issue with the sequence file
        """
        try:
            # Determine format if not specified
            if format is None:
                suffix = path.suffix.lower()
                if suffix in ('.fasta', '.fa'):
                    format = 'fasta'
                elif suffix in ('.fastq', '.fq'):
                    format = 'fastq'
                else:
                    raise InputError(f"Couldn't determine format for file: {path}")

            # Determine how to open the file based on extension
            if str(path).endswith(".gz"):
                open_func = lambda p: gzip.open(p, "rt", encoding="utf-8")
            else:
                open_func = lambda p: open(p, "r", encoding="utf-8")

            # Parse the file
            with open_func(path) as file:
                return SeqIO.to_dict(SeqIO.parse(file, format))
        except Exception as e:
            raise InputError(f"Error parsing sequence file: {str(e)}")

class SAMParser(Parser):
    """Parser for SAM/BAM files."""

    def parse(self, path: Path) -> Any:
        """Parse SAM/BAM file using pysam.

        Args:
            path: Path to SAM/BAM file

        Returns:
            pysam.AlignmentFile object

        Raises:
            InputError: If there's an issue with the SAM/BAM file
        """
        try:
            import pysam
            return pysam.AlignmentFile(str(path), "r")
        except Exception as e:
            raise InputError(f"Error parsing SAM/BAM file: {str(e)}")

class MappingParser(Parser):
    """Parser for mapping files (e.g., seq2taxid)."""

    def parse(self, path: Path) -> Dict[str, str]:
        """Parse mapping file into a dictionary.

        Args:
            path: Path to mapping file

        Returns:
            Dictionary mapping first column to second column

        Raises:
            InputError: If there's an issue with the mapping file
        """
        try:
            result = {}
            with open(path, encoding="utf8") as file:
                for line in file:
                    parts = line.rstrip().split("\t")
                    if len(parts) >= 2:
                        result[parts[0]] = parts[1]
            return result
        except Exception as e:
            raise InputError(f"Error parsing mapping file: {str(e)}")

# Factory function to get appropriate parser
def get_parser(file_type: str) -> Parser:
    """Get appropriate parser for file type.

    Args:
        file_type: Type of file to parse

    Returns:
        Parser object
    """
    parsers = {
        'taxonomy': TaxonomyParser(),
        'sequence': SequenceParser(),
        'sam': SAMParser(),
        'mapping': MappingParser()
    }

    return parsers.get(file_type, Parser())

def parse_taxonomy_file(taxonomy_path: Path) -> pd.DataFrame:
    """Parse taxonomy file into a pandas DataFrame.

    Legacy wrapper for TaxonomyParser.

    Args:
        taxonomy_path: Path to taxonomy.tsv file

    Returns:
        DataFrame containing taxonomy information indexed by tax_id

    Raises:
        InputError: If there's an issue with the taxonomy file
    """
    parser = TaxonomyParser()
    return parser.parse(taxonomy_path)

def output_sequences(
    in_path: Path,
    seq_output_path: Path,
    input_type: str,
    keep_ids: Set[str]
) -> None:
    """Output specified list of sequences from input_file based on sequence id.

    Args:
        in_path: Path to input fasta or fastq
        seq_output_path: Output path for fasta/q of unclassified sequences
        input_type: fasta or fastq
        keep_ids: Set of sequence id strings

    Raises:
        InputError: If there's an issue with the input or output files
    """
    try:
        # Use a more memory-efficient approach for large files
        # Determine how to open the file based on extension
        if str(in_path).endswith(".gz"):
            open_func = lambda p: gzip.open(p, "rt", encoding="utf-8")
        else:
            open_func = lambda p: open(p, "r", encoding="utf-8")

        # Create output file path
        out_file_path = f"{seq_output_path}.{input_type}"

        # Track the number of sequences written
        count = 0

        # Process in a memory-efficient way by streaming
        with open_func(in_path) as in_file, \
                open(out_file_path, "w", encoding="utf-8") as out_seq_file:

            # Create a generator for the sequences
            for seq in SeqIO.parse(in_file, input_type):
                if seq.id in keep_ids:
                    SeqIO.write(seq, out_seq_file, input_type)
                    count += 1

                    # Log progress for very large files
                    if count % 10000 == 0:
                        logger.debug(f"Processed {count} matching sequences")

        logger.info(f"Wrote {count} sequences to {out_file_path}")

    except Exception as e:
        raise InputError(f"Error outputting sequences: {str(e)}")