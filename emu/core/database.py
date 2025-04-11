"""Database operations for Emu."""

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional, Union, Any

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from emu.models.errors import DatabaseError, TaxonomyError
from emu.core.taxonomy import get_species_tid, lineage_dict_from_tid
from emu.core.utils import TAXONOMY_RANKS

logger = logging.getLogger(__name__)

def create_names_dict(names_path: Path) -> Dict[str, str]:
    """
    Convert names.dmp file into a dictionary.

    Args:
        names_path: Path to names.dmp file

    Returns:
        Dictionary of names.dmp with 'tax_id' as keys, 'name_txt' as values

    Raises:
        DatabaseError: If there's an issue with loading names.dmp
    """
    try:
        name_headers = ['tax_id', 'name_txt', 'name_class']
        names_df = pd.read_csv(
            names_path,
            sep='|',
            index_col=False,
            header=None,
            dtype=str
        ).drop([2, 4], axis=1)

        names_df.columns = name_headers
        for col in names_df.columns:
            names_df[col] = names_df[col].str.strip()

        names_df = names_df[names_df["name_class"] == "scientific name"]
        return dict(zip(names_df['tax_id'], names_df['name_txt']))
    except Exception as e:
        raise DatabaseError(f"Error loading names.dmp file: {str(e)}")

def create_nodes_dict(nodes_path: Path) -> Dict[str, Tuple[str, str]]:
    """
    Convert nodes.dmp file into a dictionary.

    Args:
        nodes_path: Path to nodes.dmp file

    Returns:
        Dictionary of nodes.dmp with 'tax_id' as keys, tuple ('parent_taxid', 'rank') as values

    Raises:
        DatabaseError: If there's an issue with loading nodes.dmp
    """
    try:
        node_headers = ['tax_id', 'parent_tax_id', 'rank']
        nodes_df = pd.read_csv(
            nodes_path,
            sep='|',
            header=None,
            dtype=str
        )[[0, 1, 2]]

        nodes_df.columns = node_headers
        for col in nodes_df.columns:
            nodes_df[col] = nodes_df[col].str.strip()

        return dict(zip(nodes_df['tax_id'], tuple(zip(nodes_df['parent_tax_id'], nodes_df['rank']))))
    except Exception as e:
        raise DatabaseError(f"Error loading nodes.dmp file: {str(e)}")

def create_species_seq2tax_dict(
    seq2tax_path: Path,
    nodes_dict: Dict[str, Tuple[str, str]]
) -> Dict[str, str]:
    """
    Convert seqid-taxid mapping in seq2tax_path to dict mapping seqid to species level taxid.

    Args:
        seq2tax_path: Path to seqid-taxid mapping file
        nodes_dict: Dict of nodes.dmp with 'tax_id' as keys

    Returns:
        Dictionary mapping seqid to species taxid

    Raises:
        DatabaseError: If there's an issue with the mapping file
    """
    try:
        seq2tax_dict = {}
        species_id_dict = {}

        with open(seq2tax_path, encoding="utf8") as file:
            # Unpack values in each line of the file
            for line in file:
                (seqid, tid) = line.rstrip().split("\t")
                # Retrieve the species taxid from the species_id_dict if already in dictionary
                if tid in species_id_dict:
                    species_tid = species_id_dict[tid]
                # Find the species taxid if not in the dictionary
                else:
                    species_tid = get_species_tid(tid, nodes_dict)
                    species_id_dict[tid] = species_tid
                seq2tax_dict[seqid] = species_tid

        return seq2tax_dict
    except Exception as e:
        raise DatabaseError(f"Error creating species-taxid mapping: {str(e)}")

def create_direct_seq2tax_dict(seq2tax_path: Path) -> Dict[str, str]:
    """
    Convert seqid-taxid mapping in seq2tax_path to dict mapping seqid to corresponding taxid.

    Args:
        seq2tax_path: Path to seqid-taxid mapping file

    Returns:
        Dictionary mapping seqid to taxid

    Raises:
        DatabaseError: If there's an issue with the mapping file
    """
    try:
        seq2_taxid = {}
        with open(seq2tax_path, encoding="utf8") as file:
            for line in file:
                (seqid, taxid) = line.rstrip().split("\t")
                seq2_taxid[seqid] = taxid
        return seq2_taxid
    except Exception as e:
        raise DatabaseError(f"Error creating direct seqid-taxid mapping: {str(e)}")

def create_unique_seq_dict(
    db_fasta_path: Path,
    seq2tax_dict: Dict[str, str]
) -> Dict[Seq, Dict[str, List[str]]]:
    """
    Creates dict of unique sequences to track sequences connected to each species taxid.

    Args:
        db_fasta_path: Path to fasta file of database sequences
        seq2tax_dict: Dict mapping seqid to species taxid

    Returns:
        Dict mapping sequence to {species_taxid: [list of sequence ids]}

    Raises:
        DatabaseError: If there's an issue with the database fasta file
    """
    try:
        fasta_dict = {}
        # Traverse through the species taxids
        for record in SeqIO.parse(str(db_fasta_path), "fasta"):
            tid = seq2tax_dict.get(record.id)
            if tid:
                # If sequence already in the dictionary, add more ids if found
                if record.seq in fasta_dict:
                    if tid in fasta_dict[record.seq].keys():
                        fasta_dict[record.seq][tid] += [record.description]
                    # Create inner species taxid dictionary and add id
                    else:
                        fasta_dict[record.seq][tid] = [record.description]
                elif record.seq.reverse_complement() in fasta_dict:
                    if tid in fasta_dict[record.seq.reverse_complement()].keys():
                        fasta_dict[record.seq.reverse_complement()][tid] += [record.description]
                    else:
                        fasta_dict[record.seq.reverse_complement()][tid] = [record.description]
                else:
                    fasta_dict[record.seq] = {tid: [record.description]}
        return fasta_dict
    except Exception as e:
        raise DatabaseError(f"Error creating unique sequence dictionary: {str(e)}")

def create_reduced_fasta(
    fasta_dict: Dict[Seq, Dict[str, List[str]]],
    db_name: str
) -> List[SeqRecord]:
    """
    Creates fasta file records of taxid for each sequence.

    Args:
        fasta_dict: Dict mapping sequence to {species_taxid: [list of sequence ids]}
        db_name: Name to represent database

    Returns:
        List of sequences for output fasta file
    """
    records = []
    count = 1

    for seq, tid_dict in fasta_dict.items():
        for taxid, descriptions in tid_dict.items():
            records.append(
                SeqRecord(
                    seq,
                    id=f"{taxid}:{db_name}:{count}",
                    description=f"{descriptions}"
                )
            )
            count += 1

    return records