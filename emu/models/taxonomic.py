"""Data models for taxonomy."""

from typing import List, Dict, Tuple, Set, Optional, Union, Any
from dataclasses import dataclass

@dataclass
class TaxonomicLineage:
    """Represents a taxonomic lineage."""
    tax_id: str
    species: Optional[str] = None
    genus: Optional[str] = None
    family: Optional[str] = None
    order: Optional[str] = None
    class_: Optional[str] = None
    phylum: Optional[str] = None
    clade: Optional[str] = None
    superkingdom: Optional[str] = None
    subspecies: Optional[str] = None
    species_subgroup: Optional[str] = None
    species_group: Optional[str] = None

    def as_tuple(self) -> Tuple:
        """Convert to tuple for backwards compatibility."""
        return (
            self.tax_id,
            self.species or "",
            self.genus or "",
            self.family or "",
            self.order or "",
            self.class_ or "",
            self.phylum or "",
            self.clade or "",
            self.superkingdom or "",
            self.subspecies or "",
            self.species_subgroup or "",
            self.species_group or ""
        )

    @classmethod
    def from_tuple(cls, data: Tuple) -> 'TaxonomicLineage':
        """Create from tuple for backwards compatibility."""
        return cls(
            tax_id=data[0],
            species=data[1] if data[1] else None,
            genus=data[2] if data[2] else None,
            family=data[3] if data[3] else None,
            order=data[4] if data[4] else None,
            class_=data[5] if data[5] else None,
            phylum=data[6] if data[6] else None,
            clade=data[7] if data[7] else None,
            superkingdom=data[8] if data[8] else None,
            subspecies=data[9] if data[9] else None,
            species_subgroup=data[10] if data[10] else None,
            species_group=data[11] if data[11] else None
        )