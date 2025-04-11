"""Error classes for Emu."""

class EmuError(Exception):
    """Base class for Emu exceptions."""
    pass

class DatabaseError(EmuError):
    """Raised when there's an issue with the database."""
    pass

class TaxonomyError(EmuError):
    """Raised when there's an issue with taxonomy."""
    pass

class InputError(EmuError):
    """Raised when there's an issue with input files."""
    pass

class AlignmentError(EmuError):
    """Raised when there's an issue with alignments."""
    pass

class CalculationError(EmuError):
    """Raised when there's an issue with calculations."""
    pass