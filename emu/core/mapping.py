"""Optimized mapping functionality for Emu."""

import os
import logging
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional, Union, Any

from emu.models.errors import AlignmentError

logger = logging.getLogger(__name__)

class Mapper:
    """Base class for sequence mapping tools."""

    def __init__(self, threads: int = 3, N: int = 50, K: int = 500000000, **kwargs):
        """Initialize mapper.

        Args:
            threads: Number of threads to use
            N: Maximum number of secondary alignments
            K: Minibatch size for mapping
            kwargs: Additional arguments (ignored, for future compatibility)
        """
        self.threads = threads
        self.N = N
        self.K = K

    def check_mapper_available(self) -> bool:
        """Check if mapper executable is available.

        Returns:
            True if mapper is available, False otherwise
        """
        raise NotImplementedError

    def generate_alignments(
        self,
        in_file_list: List[Path],
        out_basename: Path,
        database: Path,
        seq_type: str = 'map-ont',
        forward_only: bool = False
    ) -> Path:
        """Generate alignments from input sequences.

        Args:
            in_file_list: List of paths to input sequences
            out_basename: Path and basename for output files
            database: Path to database directory
            seq_type: Sequencing type (e.g., 'map-ont', 'sr')
            forward_only: Whether to use forward transcript strand only

        Returns:
            Path to SAM alignment file

        Raises:
            AlignmentError: If there's an issue with the alignment
        """
        raise NotImplementedError

class Minimap2Mapper(Mapper):
    """Mapper using standard minimap2."""

    def check_mapper_available(self) -> bool:
        """Check if minimap2 executable is available.

        Returns:
            True if minimap2 is available, False otherwise
        """
        try:
            result = subprocess.run(
                ["minimap2", "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )
            if result.returncode == 0:
                logger.info("minimap2 found in PATH")
                return True
            else:
                logger.warning(f"minimap2 check failed with return code: {result.returncode}")
                return False
        except FileNotFoundError:
            logger.warning("minimap2 not found in PATH")
            return False
        except Exception as e:
            logger.warning(f"Error checking minimap2 availability: {str(e)}")
            return False

    def generate_alignments(
        self,
        in_file_list: List[Path],
        out_basename: Path,
        database: Path,
        seq_type: str = 'map-ont',
        forward_only: bool = False
    ) -> Path:
        """Generate alignments using minimap2.

        Args:
            in_file_list: List of paths to input sequences
            out_basename: Path and basename for output files
            database: Path to database directory
            seq_type: Sequencing type (e.g., 'map-ont', 'sr')
            forward_only: Whether to use forward transcript strand only

        Returns:
            Path to SAM alignment file

        Raises:
            AlignmentError: If there's an issue with the alignment
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

            # Build minimap2 command with optimized parameters
            base_cmd = (
                f"minimap2 -ax {seq_type} -t {self.threads} -N {self.N} -p .9 -K {self.K} "
            )

            if forward_only:
                base_cmd += "-u f "

            # Add input and output files
            cmd = f"{base_cmd} {db_sequence_file} {input_files_str} -o {sam_align_file}"

            # Run minimap2
            logger.info(f"Running alignment with command: {cmd}")
            subprocess.check_output(cmd, shell=True)

            # Inspect the SAM file
            self._inspect_sam_file(sam_align_file)

            return Path(sam_align_file)

        except subprocess.CalledProcessError as e:
            raise AlignmentError(f"Error running minimap2: {str(e)}")
        except Exception as e:
            raise AlignmentError(f"Error generating alignments: {str(e)}")

    def _inspect_sam_file(self, sam_path):
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

class MM2PlusMapper(Mapper):
    """Mapper using mm2-plus for improved performance."""

    def __init__(self, threads: int = 3, N: int = 50, K: int = 500000000, mm2plus_path: Optional[str] = None, **kwargs):
        """Initialize mm2-plus mapper.

        Args:
            threads: Number of threads to use
            N: Maximum number of secondary alignments
            K: Minibatch size for mapping
            mm2plus_path: Path to mm2-plus executable (if not in PATH)
            kwargs: Additional arguments (ignored, for compatibility)
        """
        super().__init__(threads, N, K)
        self.mm2plus_path = mm2plus_path or "mm2-plus"

    def check_mapper_available(self) -> bool:
        """Check if mm2-plus executable is available.

        Returns:
            True if mm2-plus is available, False otherwise
        """
        try:
            # Check if explicit path provided
            if self.mm2plus_path != "mm2-plus":
                if not os.path.exists(self.mm2plus_path):
                    logger.warning(f"mm2-plus not found at specified path: {self.mm2plus_path}")
                    return False

            result = subprocess.run(
                [self.mm2plus_path, "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )
            if result.returncode == 0:
                logger.info(f"mm2-plus found: {self.mm2plus_path}")
                return True
            else:
                logger.warning(f"mm2-plus check failed with return code: {result.returncode}")
                return False
        except FileNotFoundError:
            logger.warning("mm2-plus not found in PATH")
            return False
        except Exception as e:
            logger.warning(f"Error checking mm2-plus availability: {str(e)}")
            return False

    def generate_alignments(
        self,
        in_file_list: List[Path],
        out_basename: Path,
        database: Path,
        seq_type: str = 'map-ont',
        forward_only: bool = False
    ) -> Path:
        """Generate alignments using mm2-plus.

        Args:
            in_file_list: List of paths to input sequences
            out_basename: Path and basename for output files
            database: Path to database directory
            seq_type: Sequencing type (e.g., 'map-ont', 'sr')
            forward_only: Whether to use forward transcript strand only

        Returns:
            Path to SAM alignment file

        Raises:
            AlignmentError: If there's an issue with the alignment
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

            # Build mm2-plus command
            # mm2-plus has enhanced performance and memory efficiency
            # -M flag enables memory profiling (available in mm2-plus)
            base_cmd = (
                f"{self.mm2plus_path} -ax {seq_type} -t {self.threads} -N {self.N} "
                f"-p .9 -K {self.K} -M "  # -M enables memory profiling
            )

            if forward_only:
                base_cmd += "-u f "

            # Add input and output files
            cmd = f"{base_cmd} {db_sequence_file} {input_files_str} -o {sam_align_file}"

            # Run mm2-plus
            logger.info(f"Running alignment with mm2-plus: {cmd}")

            # Capture and log memory usage information from mm2-plus
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            if result.stderr:
                for line in result.stderr.split('\n'):
                    if "memory" in line.lower() or "peak" in line.lower():
                        logger.info(f"mm2-plus memory info: {line}")

            # Verify the SAM file was created
            if not os.path.exists(sam_align_file):
                raise AlignmentError(f"mm2-plus did not create output file: {sam_align_file}")

            # Inspect the SAM file
            self._inspect_sam_file(sam_align_file)

            return Path(sam_align_file)

        except subprocess.CalledProcessError as e:
            raise AlignmentError(f"Error running mm2-plus: {str(e)}")
        except Exception as e:
            raise AlignmentError(f"Error generating alignments with mm2-plus: {str(e)}")

    def _inspect_sam_file(self, sam_path):
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

def create_mapper(mapper_type: str = "auto", **kwargs) -> Mapper:
    """Factory function to create appropriate mapper.

    Args:
        mapper_type: Type of mapper to create ("minimap2", "mm2-plus", or "auto")
        kwargs: Additional arguments for the mapper

    Returns:
        Mapper object

    Raises:
        AlignmentError: If no suitable mapper is available
    """
    # Extract mm2plus_path from kwargs
    mm2plus_path = kwargs.pop('mm2plus_path', None)

    # Initialize mappers - only pass mm2plus_path to MM2PlusMapper
    minimap2_mapper = Minimap2Mapper(**kwargs)
    mm2plus_mapper = MM2PlusMapper(mm2plus_path=mm2plus_path, **kwargs)

    if mapper_type.lower() == "auto":
        # Try mm2-plus first, fall back to minimap2
        if mm2plus_mapper.check_mapper_available():
            logger.info("Using mm2-plus for mapping (better performance)")
            return mm2plus_mapper
        elif minimap2_mapper.check_mapper_available():
            logger.info("Using standard minimap2 for mapping")
            return minimap2_mapper
        else:
            raise AlignmentError("No mapping tool available. Please install mm2-plus or minimap2")
    elif mapper_type.lower() == "mm2-plus":
        if mm2plus_mapper.check_mapper_available():
            logger.info("Using mm2-plus for mapping")
            return mm2plus_mapper
        else:
            raise AlignmentError("mm2-plus not available. Please install from https://github.com/at-cg/mm2-plus")
    elif mapper_type.lower() == "minimap2":
        if minimap2_mapper.check_mapper_available():
            logger.info("Using standard minimap2 for mapping")
            return minimap2_mapper
        else:
            raise AlignmentError("minimap2 not available. Please install it via conda or your package manager")
    else:
        raise AlignmentError(f"Unknown mapper type: {mapper_type}")

def generate_alignments(
    in_file_list: List[Path],
    out_basename: Path,
    database: Path,
    seq_type: str = 'map-ont',
    threads: int = 3,
    N: int = 50,
    K: int = 500000000,
    mm2_forward_only: bool = False,
    mapper_type: str = "auto",
    mm2plus_path: Optional[str] = None
) -> Path:
    """
    Generate alignments using the most efficient available mapper.

    This function will use mm2-plus if available, falling back to minimap2 if necessary.

    Args:
        in_file_list: List of paths to input sequences
        out_basename: Path and basename for output files
        database: Path to database directory
        seq_type: Sequencing type for minimap2
        threads: Number of threads for alignment
        N: Max number of secondary alignments
        K: Minibatch size for mapping
        mm2_forward_only: Whether to use forward transcript strand only
        mapper_type: Type of mapper to use ("minimap2", "mm2-plus", or "auto")
        mm2plus_path: Optional custom path to mm2-plus executable

    Returns:
        Path to SAM alignment file

    Raises:
        AlignmentError: If there's an issue generating alignments
    """
    # Check if input is already a SAM file
    if in_file_list and in_file_list[0].suffix == '.sam':
        # If input is already a SAM file, return it directly
        return in_file_list[0]

    # Create mapper with specified parameters
    try:
        mapper = create_mapper(
            mapper_type=mapper_type,
            threads=threads,
            N=N,
            K=K,
            mm2plus_path=mm2plus_path
        )
    except Exception as e:
        logger.error(f"Error creating mapper: {str(e)}")
        # Fall back to direct minimap2 execution in case of mapper creation failure
        raise AlignmentError(f"Failed to create mapper: {str(e)}")

    # Generate alignments using selected mapper
    return mapper.generate_alignments(
        in_file_list=in_file_list,
        out_basename=out_basename,
        database=database,
        seq_type=seq_type,
        forward_only=mm2_forward_only
    )