"""Configuration management for Emu."""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

from emu.models.errors import EmuError

class ConfigError(EmuError):
    """Raised when there's an issue with configuration."""
    pass

class EmuConfig:
    """Centralized configuration for Emu."""

    def __init__(self, args: Optional[Any] = None):
        """
        Initialize configuration from args and environment.

        Args:
            args: Arguments from argparse

        Raises:
            ConfigError: If required configuration is missing
        """
        # Command-specific configuration - get this first
        self.command = getattr(args, 'command', None)

        # Common configuration
        self.threads = getattr(args, 'threads', 3)
        self.verbose = getattr(args, 'verbose', False)

        # Database directory - only required for some commands
        self.database_dir = getattr(args, 'db', None) or os.environ.get("EMU_DATABASE_DIR")
        if self.command in ['abundance', 'collapse-taxonomy', 'combine-outputs'] and not self.database_dir:
            raise ConfigError("Database directory not specified. Either 'export EMU_DATABASE_DIR=<path_to_database>' or utilize '--db' parameter.")
        if self.database_dir:
            self.database_dir = Path(self.database_dir)

        # Abundance command configuration
        if self.command == 'abundance':
            self.input_files = [Path(f) for f in getattr(args, 'input_file', [])]
            self.seq_type = getattr(args, 'type', 'map-ont')
            self.min_abundance = getattr(args, 'min_abundance', 0.0001)
            self.N = getattr(args, 'N', 50)
            self.K = getattr(args, 'K', 500000000)
            self.mm2_forward_only = getattr(args, 'mm2_forward_only', False)
            self.output_dir = Path(getattr(args, 'output_dir', "./results"))
            self.output_basename = getattr(args, 'output_basename', None)
            self.keep_files = getattr(args, 'keep_files', False)
            self.keep_counts = getattr(args, 'keep_counts', False)
            self.keep_read_assignments = getattr(args, 'keep_read_assignments', False)
            self.output_unclassified = getattr(args, 'output_unclassified', False)

        # Build-database command configuration
        elif self.command == 'build-database':
            self.db_name = getattr(args, 'db_name', None)
            self.sequences = Path(getattr(args, 'sequences', ''))
            self.seq2tax = Path(getattr(args, 'seq2tax', ''))
            self.ncbi_taxonomy = getattr(args, 'ncbi_taxonomy', None)
            if self.ncbi_taxonomy:
                self.ncbi_taxonomy = Path(self.ncbi_taxonomy)
            self.taxonomy_list = getattr(args, 'taxonomy_list', None)
            if self.taxonomy_list:
                self.taxonomy_list = Path(self.taxonomy_list)

        # Collapse-taxonomy command configuration
        elif self.command == 'collapse-taxonomy':
            self.input_path = Path(getattr(args, 'input_path', ''))
            self.rank = getattr(args, 'rank', '')

        # Combine-outputs command configuration
        elif self.command == 'combine-outputs':
            self.dir_path = Path(getattr(args, 'dir_path', ''))
            self.rank = getattr(args, 'rank', '')
            self.split_tables = getattr(args, 'split_tables', False)
            self.counts = getattr(args, 'counts', False)