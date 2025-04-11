"""Taxonomy-related functionality."""

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional, Union, Any

from emu.models.errors import TaxonomyError
from emu.core.utils import TAXONOMY_RANKS, RANKS_PRINTOUT
from emu.models.taxonomic import TaxonomicLineage

logger = logging.getLogger(__name__)

def lineage_dict_from_tid(
    taxid: str,
    nodes_dict: Dict[str, Tuple[str, str]],
    names_dict: Dict[str, str]
) -> Tuple:
    """
    For each given taxid, traverse the node_dict to build the lineage.

    Args:
        taxid: Tax ID to retrieve lineage dict
        nodes_dict: Dict of nodes.dmp with 'tax_id' as keys
        names_dict: Dict of names.dmp with 'tax_id' as keys

    Returns:
        A tuple containing the scientific name for each taxonomic rank

    Raises:
        TaxonomyError: If there's an issue with the taxonomy
    """
    try:
        # Initialize list and record tax id
        lineage_list = [taxid] + [""] * (len(RANKS_PRINTOUT) - 1)
        # Traverse the nodes to create the lineage
        current_taxid = taxid
        while names_dict[current_taxid] != "root":
            tup = nodes_dict[current_taxid]
            # Find the name for each taxonomic rank
            if tup[1] in RANKS_PRINTOUT:  # Check rank in printout list
                idx = RANKS_PRINTOUT.index(tup[1])
                lineage_list[idx] = names_dict[current_taxid]
            current_taxid = tup[0]
        return tuple(lineage_list)
    except KeyError as e:
        raise TaxonomyError(f"Taxonomy ID {taxid} not found in taxonomy database: {str(e)}")
    except Exception as e:
        raise TaxonomyError(f"Error building lineage for taxonomy ID {taxid}: {str(e)}")

def get_species_tid(tid: str, nodes_dict: Dict[str, Tuple[str, str]]) -> str:
    """
    Get lowest taxid down to species-level in lineage for taxid [tid].

    Args:
        tid: Taxid for species level or more specific
        nodes_dict: Dict of nodes.dmp with 'tax_id' as keys

    Returns:
        Species taxid in lineage

    Raises:
        TaxonomyError: If there's an issue with the taxonomy
    """
    if str(tid) not in nodes_dict.keys():
        raise TaxonomyError(f"Taxid:{tid} not found in nodes file")

    start_id = tid
    current_id = tid

    while nodes_dict[str(current_id)][1] not in TAXONOMY_RANKS:
        if nodes_dict[str(current_id)][1] == "no rank":
            raise TaxonomyError(f"Taxid:{start_id} does not have an acceptable taxonomy label in lineage. Acceptable taxonomy labels: {TAXONOMY_RANKS}")
        current_id = nodes_dict[str(current_id)][0]

    return current_id