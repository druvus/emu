"""File writers for Emu."""

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional, Union, Any

import pandas as pd

from emu.models.errors import InputError

logger = logging.getLogger(__name__)

def freq_to_lineage_df(
    freq: Dict[int, float],
    tsv_output_path: Path,
    taxonomy_df: pd.DataFrame,
    mapped_count: int,
    unmapped_count: int,
    mapped_unclassified_count: int,
    counts: bool = False
) -> pd.DataFrame:
    """
    Convert frequency dictionary to a DataFrame with taxonomic lineages.

    Args:
        freq: Dict mapping species_tax_id to estimated likelihood
        tsv_output_path: Path to output .tsv file
        taxonomy_df: DataFrame of all database sequence taxonomy
        mapped_count: Number of mapped reads
        unmapped_count: Number of unmapped reads
        mapped_unclassified_count: Number of mapped but unclassified reads
        counts: Whether to include estimated counts in output

    Returns:
        DataFrame with lineage and abundances

    Raises:
        InputError: If there's an issue with creating the output
    """
    try:
        # Add tax lineage for values in freq
        results_df = pd.DataFrame(
            zip(
                list(freq.keys()) + ['unmapped', 'mapped_unclassified'],
                list(freq.values()) + [0, 0]
            ),
            columns=["tax_id", "abundance"]
        ).set_index('tax_id')

        # Join with taxonomy information
        results_df = results_df.join(taxonomy_df, how='left').reset_index()

        # Add estimated counts if requested
        if counts:
            classified_count = mapped_count - mapped_unclassified_count
            counts_series = pd.concat([
                (results_df["abundance"] * classified_count)[:-2],
                pd.Series(unmapped_count),
                pd.Series(mapped_unclassified_count)
            ], ignore_index=True)
            results_df["estimated counts"] = counts_series

        # Write to TSV file
        output_file = f"{tsv_output_path}.tsv"
        results_df.to_csv(output_file, sep='\t', index=False)
        logger.info(f"Wrote results to {output_file}")

        return results_df

    except Exception as e:
        raise InputError(f"Error creating lineage DataFrame: {str(e)}")

def output_read_assignments(
    p_sgr: Dict[int, Dict[str, float]],
    tsv_output_path: Path
) -> pd.DataFrame:
    """
    Output file of read assignment distributions.

    Args:
        p_sgr: Dict mapping species_tax_id to {read_id: probability}
        tsv_output_path: Path to output .tsv file

    Returns:
        DataFrame of read assignment distributions

    Raises:
        InputError: If there's an issue with creating the output
    """
    try:
        dist_df = pd.DataFrame(p_sgr)
        output_file = f"{tsv_output_path}.tsv"
        dist_df.to_csv(output_file, sep='\t')
        logger.info(f"Wrote read assignments to {output_file}")
        return dist_df

    except Exception as e:
        raise InputError(f"Error creating read assignments: {str(e)}")

def collapse_rank(path: Path, rank: str) -> None:
    """
    Stores a version of Emu output collapsed at the specified taxonomic rank.

    Args:
        path: Path to Emu output
        rank: Taxonomic rank for collapsed abundance

    Raises:
        InputError: If there's an issue with the input or output files
    """
    from emu.core.utils import TAXONOMY_RANKS

    try:
        df_emu = pd.read_csv(path, sep='\t')

        if rank not in TAXONOMY_RANKS:
            raise InputError(f"Specified rank must be in list: {TAXONOMY_RANKS}")

        keep_ranks = TAXONOMY_RANKS[TAXONOMY_RANKS.index(rank):]

        for keep_rank in list(keep_ranks):  # Create a copy to modify safely
            if keep_rank not in df_emu.columns:
                keep_ranks.remove(keep_rank)

        if "estimated counts" in df_emu.columns:
            df_emu_copy = df_emu[['abundance', 'estimated counts'] + keep_ranks]
            df_emu_copy = df_emu_copy.replace({'-': 0})
            df_emu_copy = df_emu_copy.astype({'abundance': 'float', 'estimated counts': 'float'})
        else:
            df_emu_copy = df_emu[['abundance'] + keep_ranks]
            df_emu_copy = df_emu_copy.replace({'-': 0})
            df_emu_copy = df_emu_copy.astype({'abundance': 'float'})

        df_emu_copy = df_emu_copy.groupby(keep_ranks, dropna=False).sum()

        output_path = f"{path.stem}-{rank}.tsv"
        df_emu_copy.to_csv(output_path, sep='\t')

        logger.info(f"File generated: {output_path}")

    except Exception as e:
        raise InputError(f"Error collapsing taxonomy ranks: {str(e)}")

def combine_outputs(
    dir_path: Path,
    rank: str,
    split_files: bool = False,
    count_table: bool = False
) -> pd.DataFrame:
    """
    Combines multiple Emu output relative abundance tables into a single table.

    Args:
        dir_path: Path of directory containing Emu output files
        rank: Taxonomic rank to combine files on
        split_files: Whether to split into separate abundance and taxonomy files
        count_table: Whether to use estimated counts instead of abundance

    Returns:
        DataFrame of the combined relative abundance files

    Raises:
        InputError: If there's an issue with the input or output files
    """
    from emu.core.utils import RANKS_ORDER

    try:
        keep_ranks = RANKS_ORDER[RANKS_ORDER.index(rank):]
        df_combined_full = pd.DataFrame(columns=keep_ranks, dtype=str)
        metric = 'estimated counts' if count_table else 'abundance'

        for file in dir_path.glob('*.tsv'):
            if 'rel-abundance' in file.name:
                name = file.stem.replace('_rel-abundance', '')

                df_sample = pd.read_csv(file, sep='\t', dtype=str)
                df_sample[[metric]] = df_sample[[metric]].apply(pd.to_numeric)

                if rank in df_sample.columns and metric in df_sample.columns:
                    # Check which keep_ranks are in df_sample
                    keep_ranks_sample = [value for value in keep_ranks
                                        if value in set(df_sample.columns)]

                    if df_sample.at[len(df_sample.index)-1, 'tax_id'] == 'unmapped':
                        df_sample.at[len(df_sample.index)-1, rank] = 'unmapped'

                    df_sample_reduced = df_sample[keep_ranks_sample + [metric]].rename(
                        columns={metric: name}
                    )

                    # Sum metric within df_sample_reduced if same tax lineage
                    df_sample_reduced = df_sample_reduced.groupby(
                        keep_ranks_sample, dropna=False
                    ).sum().reset_index()

                    df_sample_reduced = df_sample_reduced.astype(object)
                    df_sample_reduced[[name]] = df_sample_reduced[[name]].apply(pd.to_numeric)

                    df_combined_full = pd.merge(df_combined_full, df_sample_reduced, how='outer')

        # Organize and sort the combined dataframe
        df_combined_full = df_combined_full.set_index(rank).sort_index().reset_index()

        # Determine output filename suffix
        filename_suffix = "-counts" if count_table else ""

        # Create output files
        if split_files:
            abundance_out_path = dir_path / f"emu-combined-abundance-{rank}{filename_suffix}.tsv"
            tax_out_path = dir_path / f"emu-combined-taxonomy-{rank}.tsv"

            df_combined_full[keep_ranks].to_csv(tax_out_path, sep='\t', index=False)
            logger.info(f"Combined taxonomy table generated: {tax_out_path}")

            keep_ranks_copy = list(keep_ranks)
            keep_ranks_copy.remove(rank)
            df_combined_full.drop(columns=keep_ranks_copy).to_csv(
                abundance_out_path, sep='\t', index=False
            )
            logger.info(f"Combined abundance table generated: {abundance_out_path}")
        else:
            out_path = dir_path / f"emu-combined-{rank}{filename_suffix}.tsv"
            df_combined_full.to_csv(out_path, sep='\t', index=False)
            logger.info(f"Combined table generated: {out_path}")

        return df_combined_full

    except Exception as e:
        raise InputError(f"Error combining outputs: {str(e)}")