#!/usr/bin/env python3
"""Command-line interface for Emu."""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional, Union, Any

import pandas as pd

from emu import __version__
from emu.core.utils import setup_logging, TAXONOMY_RANKS
from emu.models.config import EmuConfig
from emu.models.errors import EmuError, InputError

logger = logging.getLogger(__name__)

def create_parser() -> argparse.ArgumentParser:
    """
    Create and return the main argument parser for Emu.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Emu: species-level taxonomic abundance for full-length 16S reads",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--version', '-v',
        action='version',
        version=f'%(prog)s v{__version__}'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help='Emu commands',
        required=True
    )

    # Abundance command
    abundance_parser = subparsers.add_parser(
        "abundance",
        help="Generate relative abundance estimates",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    abundance_parser.add_argument(
        'input_file',
        type=str,
        nargs='+',
        help='filepath to input nt sequence file'
    )
    abundance_parser.add_argument(
        '--type', '-x',
        choices=['map-ont', 'map-pb', 'sr', 'lr:hq', 'map-hifi', 'splice:hq'],
        default='map-ont',
        help='short-read: sr, Pac-Bio:map-pb, ONT:map-ont, ...'
    )
    abundance_parser.add_argument(
        '--min-abundance', '-a',
        type=float,
        default=0.0001,
        help='min species abundance in results'
    )
    abundance_parser.add_argument(
        '--db',
        type=str,
        default=None,
        help='path to emu database containing required files'
    )
    abundance_parser.add_argument(
        '--N', '-N',
        type=int,
        default=50,
        help='minimap max number of secondary alignments per read'
    )
    abundance_parser.add_argument(
        '--K', '-K',
        type=int,
        default=500000000,
        help='minibatch size for minimap2 mapping'
    )
    abundance_parser.add_argument(
        '--mm2-forward-only',
        action="store_true",
        help='force minimap2 to consider the forward transcript strand only'
    )
    abundance_parser.add_argument(
        '--output-dir',
        type=str,
        default="./results",
        help='output directory name'
    )
    abundance_parser.add_argument(
        '--output-basename',
        type=str,
        help='basename for all emu output files'
    )
    abundance_parser.add_argument(
        '--keep-files',
        action="store_true",
        help='keep working files in output-dir'
    )
    abundance_parser.add_argument(
        '--keep-counts',
        action="store_true",
        help='include estimated read counts in output'
    )
    abundance_parser.add_argument(
        '--keep-read-assignments',
        action="store_true",
        help='output file of read assignment distribution'
    )
    abundance_parser.add_argument(
        '--output-unclassified',
        action="store_true",
        help='output unclassified sequences'
    )
    abundance_parser.add_argument(
        '--threads',
        type=int,
        default=3,
        help='threads utilized by minimap'
    )

    # Build-database command
    build_db_parser = subparsers.add_parser(
        "build-database",
        help="Build custom Emu database",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    build_db_parser.add_argument(
        'db_name',
        type=str,
        help='custom database name'
    )
    build_db_parser.add_argument(
        '--sequences',
        type=str,
        required=True,
        help='path to fasta of database sequences'
    )
    build_db_parser.add_argument(
        '--seq2tax',
        type=str,
        required=True,
        help='path to tsv mapping species tax id to fasta sequence headers'
    )
    taxonomy_group = build_db_parser.add_mutually_exclusive_group(required=True)
    taxonomy_group.add_argument(
        '--ncbi-taxonomy',
        type=str,
        help='path to directory containing both a names.dmp and nodes.dmp file'
    )
    taxonomy_group.add_argument(
        '--taxonomy-list',
        type=str,
        help='path to .tsv file mapping full lineage to corresponding taxid'
    )

    # Collapse-taxonomy command
    collapse_parser = subparsers.add_parser(
        "collapse-taxonomy",
        help="Collapse emu output at specified taxonomic rank",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    collapse_parser.add_argument(
        'input_path',
        type=str,
        help='emu output filepath'
    )
    collapse_parser.add_argument(
        'rank',
        type=str,
        help='collapsed taxonomic rank'
    )

    # Combine-outputs command
    combine_parser = subparsers.add_parser(
        "combine-outputs",
        help="Combine Emu rel abundance outputs to a single table",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    combine_parser.add_argument(
        'dir_path',
        type=str,
        help='path to directory containing Emu output files'
    )
    combine_parser.add_argument(
        'rank',
        type=str,
        help='taxonomic rank to include in combined table'
    )
    combine_parser.add_argument(
        '--split-tables',
        action="store_true",
        help='two output tables:abundances and taxonomy lineages'
    )
    combine_parser.add_argument(
        '--counts',
        action="store_true",
        help='counts rather than abundances in output table'
    )

    return parser

def validate_input_files(input_files: List[str]) -> List[Path]:
    """
    Validate input files and ensure they exist with correct extensions.

    Args:
        input_files: List of input file paths as strings

    Returns:
        List of validated Path objects

    Raises:
        InputError: If any input file cannot be found
    """
    validated_files = []

    for input_file in input_files:
        input_path = Path(input_file)

        # If the file exists, add it directly
        if input_path.exists():
            validated_files.append(input_path)
            continue

        # Try common file extensions if the file doesn't exist as specified
        found = False
        for ext in ['.fa', '.fasta', '.fq', '.fastq']:
            test_path = Path(f"{input_file}{ext}")
            if test_path.exists():
                logger.debug(f"Found file with extension: {test_path}")
                validated_files.append(test_path)
                found = True
                break

        # If we still haven't found the file, try removing extension and trying again
        if not found and '.' in input_file:
            base_name = input_file.rsplit('.', 1)[0]
            for ext in ['.fa', '.fasta', '.fq', '.fastq']:
                test_path = Path(f"{base_name}{ext}")
                if test_path.exists():
                    logger.debug(f"Found file with different extension: {test_path}")
                    validated_files.append(test_path)
                    found = True
                    break

        # If we still can't find the file, raise an error
        if not found:
            raise InputError(f"Input file not found: {input_file}")

    return validated_files

def run_abundance(config: EmuConfig) -> None:
    """
    Run the abundance command.

    Args:
        config: Configuration for the abundance command
    """
    from emu.io.parsers import parse_taxonomy_file
    from emu.core.abundance import (
        generate_alignments, get_cigar_op_log_probabilities,
        log_prob_rgs_dict, expectation_maximization_iterations
    )
    from emu.io.writers import freq_to_lineage_df, output_read_assignments
    from emu.io.parsers import output_sequences

    # Validate input files
    try:
        validated_input_files = validate_input_files(
            [str(f) for f in config.input_files]
        )
        config.input_files = validated_input_files
    except InputError as e:
        logger.error(f"Input file error: {str(e)}")
        raise

    # Create output directory if it doesn't exist
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Load taxonomy data
    taxonomy_path = config.database_dir / "taxonomy.tsv"
    df_taxonomy = parse_taxonomy_df(taxonomy_path)
    db_species_tids = df_taxonomy.index.tolist()

    # Set up output paths
    if not config.output_basename:
        out_basename = config.output_dir / "-".join([f.stem for f in config.input_files])
    else:
        out_basename = config.output_dir / config.output_basename

    # Generate alignments and perform EM algorithm
    sam_file = generate_alignments(
        config.input_files,
        out_basename,
        config.database_dir,
        config.seq_type,
        config.threads,
        config.N,
        config.K,
        config.mm2_forward_only
    )

    log_prob_cigar_op, locs_p_cigar_zero, longest_align_dict = get_cigar_op_log_probabilities(sam_file)

    log_prob_rgs, set_unmapped, set_mapped = log_prob_rgs_dict(
        sam_file,
        log_prob_cigar_op,
        longest_align_dict,
        locs_p_cigar_zero
    )

    f_full, f_set_thresh, read_dist = expectation_maximization_iterations(
        log_prob_rgs,
        db_species_tids,
        0.01,
        config.min_abundance
    )

    # Calculate classified and unclassified reads
    classified_reads = {read_id for inner_dict in read_dist.values() for read_id in inner_dict}
    mapped_unclassified = set_mapped - classified_reads
    logger.info(f"Unclassified mapped read count: {len(mapped_unclassified)}")

    # Generate output files
    freq_to_lineage_df(
        f_full,
        f"{out_basename}_rel-abundance",
        df_taxonomy,
        len(set_mapped),
        len(set_unmapped),
        len(mapped_unclassified),
        config.keep_counts
    )

    # Output read assignment distributions if requested
    if config.keep_read_assignments:
        output_read_assignments(
            read_dist,
            f"{out_basename}_read-assignment-distributions"
        )

    # Output threshold-filtered results if available
    if f_set_thresh:
        freq_to_lineage_df(
            f_set_thresh,
            f"{out_basename}_rel-abundance-threshold-{config.min_abundance}",
            df_taxonomy,
            len(set_mapped),
            len(set_unmapped),
            len(mapped_unclassified),
            config.keep_counts
        )

    # Output unmapped and unclassified sequences if requested
    if config.output_unclassified:
        input_file = config.input_files[0]
        input_filetype = "fastq" if input_file.suffix.lower() in [".fastq", ".fq"] else "fasta"

        output_sequences(
            input_file,
            f"{out_basename}_unmapped",
            input_filetype,
            set_unmapped
        )

        output_sequences(
            input_file,
            f"{out_basename}_unclassified_mapped",
            input_filetype,
            mapped_unclassified
        )

    # Clean up temporary files
    if not config.keep_files and sam_file.exists():
        sam_file.unlink()
        logger.info(f"Removed temporary file: {sam_file}")

def run_build_database(config: EmuConfig) -> None:
    """
    Run the build-database command.

    Args:
        config: Configuration for the build-database command
    """
    from emu.core.database import (
        create_names_dict, create_nodes_dict, create_species_seq2tax_dict,
        create_direct_seq2tax_dict, create_unique_seq_dict, create_reduced_fasta
    )
    from Bio import SeqIO

    # Create database directory
    custom_db_path = Path.cwd() / config.db_name
    custom_db_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Emu custom database generating at path: {custom_db_path}")

    # Set up seq2tax dict for either NCBI or direct taxonomy
    if config.ncbi_taxonomy:
        dict_names = create_names_dict(config.ncbi_taxonomy / 'names.dmp')
        dict_nodes = create_nodes_dict(config.ncbi_taxonomy / 'nodes.dmp')
        seq2tax = create_species_seq2tax_dict(config.seq2tax, dict_nodes)
    else:
        seq2tax = create_direct_seq2tax_dict(config.seq2tax)

    # Get unique database IDs and create fasta with desired format
    db_unique_ids = set(seq2tax.values())
    dict_fasta = create_unique_seq_dict(config.sequences, seq2tax)
    fasta_records = create_reduced_fasta(dict_fasta, config.db_name)

    # Write the species_taxid.fasta file
    SeqIO.write(fasta_records, custom_db_path / 'species_taxid.fasta', "fasta")

    # Build taxonomy for database
    output_taxonomy_location = custom_db_path / "taxonomy.tsv"

    if config.ncbi_taxonomy:
        build_ncbi_taxonomy(db_unique_ids, dict_nodes, dict_names, output_taxonomy_location)
    else:
        build_direct_taxonomy(db_unique_ids, config.taxonomy_list, output_taxonomy_location)

    logger.info("Database creation successful")

def run_collapse_taxonomy(config: EmuConfig) -> None:
    """
    Run the collapse-taxonomy command.

    Args:
        config: Configuration for the collapse-taxonomy command
    """
    from emu.io.writers import collapse_rank

    collapse_rank(config.input_path, config.rank)

def run_combine_outputs(config: EmuConfig) -> None:
    """
    Run the combine-outputs command.

    Args:
        config: Configuration for the combine-outputs command
    """
    from emu.io.writers import combine_outputs

    combine_outputs(config.dir_path, config.rank, config.split_tables, config.counts)

def parse_taxonomy_df(taxonomy_path: Path) -> pd.DataFrame:
    """
    Parse taxonomy file into a pandas DataFrame.

    Args:
        taxonomy_path: Path to taxonomy.tsv file

    Returns:
        DataFrame containing taxonomy information indexed by tax_id
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

def build_ncbi_taxonomy(
    unique_tids: Set[str],
    nodes_dict: Dict[str, Tuple[str, str]],
    names_dict: Dict[str, str],
    filepath: Path
) -> None:
    """
    Creates a tsv file with tax lineage for each id in unique_tids.

    Args:
        unique_tids: Set of unique taxids in database sequences
        nodes_dict: Dict of nodes.dmp with 'tax_id' as keys
        names_dict: Dict of names.dmp with 'tax_id' as keys
        filepath: Path to output taxonomy file
    """
    from emu.core.taxonomy import lineage_dict_from_tid
    from emu.core.utils import RANKS_PRINTOUT

    with open(filepath, "w", encoding="utf8") as file:
        # Write the header to the file
        dummy_str = '\t'.join(['%s',] * len(RANKS_PRINTOUT)) + '\n'
        file.write(dummy_str % tuple(RANKS_PRINTOUT))

        # Write each lineage as a row in the file
        for tid in unique_tids:
            lst = lineage_dict_from_tid(tid, nodes_dict, names_dict)
            file.write(dummy_str % lst)

    logger.info(f"NCBI taxonomy file created: {filepath}")

def build_direct_taxonomy(
    tid_set: Set[str],
    lineage_path: Path,
    taxonomy_file: Path
) -> None:
    """
    Create a tsv file containing taxid and lineage.

    Args:
        tid_set: Set of taxids to include
        lineage_path: Path to file with lineage and taxid
        taxonomy_file: Path to output taxonomy file
    """
    from emu.core.utils import RANKS_PRINTOUT

    with open(taxonomy_file, 'w', encoding="utf8") as tax_output_file:
        with open(lineage_path, encoding="utf8") as file:
            first_line = file.readline()
            tax_output_file.write(f"{RANKS_PRINTOUT[0]}\t")
            tax_output_file.write(first_line.split("\t", 1)[1])

            for line in file:
                tax_id = line.split("\t", 1)[0]
                if tax_id in tid_set:  # Only add if taxid is in fasta
                    tax_output_file.write(line)

    logger.info(f"Direct taxonomy file created: {taxonomy_file}")

def main() -> int:
    """
    Main entry point for Emu command-line interface.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.verbose)

    try:
        # Create configuration
        config = EmuConfig(args)

        # Dispatch to appropriate command handler
        if config.command == 'abundance':
            run_abundance(config)
        elif config.command == 'build-database':
            run_build_database(config)
        elif config.command == 'collapse-taxonomy':
            run_collapse_taxonomy(config)
        elif config.command == 'combine-outputs':
            run_combine_outputs(config)
        else:
            logger.error(f"Unknown command: {config.command}")
            return 1

        return 0

    except EmuError as e:
        logger.error(f"Error: {str(e)}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())