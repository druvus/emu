"""Batch processing utilities for Emu."""

import logging
import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Callable, Iterable, TypeVar

from emu.core.jit_utils import NUMBA_AVAILABLE, parallel_process_chunks

logger = logging.getLogger(__name__)

# Type variables for generics
T = TypeVar('T')
U = TypeVar('U')

class BatchProcessor:
    """Generic framework for batch processing of any dataset."""

    def __init__(self, batch_size: int = 1000, show_progress: bool = False):
        """Initialize a batch processor.

        Args:
            batch_size: Number of items to process in each batch
            show_progress: Whether to log progress after each batch
        """
        self.batch_size = batch_size
        self.show_progress = show_progress

    def process(self, items: Iterable[T], process_func: Callable[[List[T], Any], List[U]],
                *args, **kwargs) -> List[U]:
        """Process items in batches.

        Args:
            items: Iterable of items to process
            process_func: Function to apply to each batch
            args, kwargs: Additional arguments for process_func

        Returns:
            Combined results from all batches
        """
        results = []
        batch = []
        processed_count = 0

        # Process items in batches
        for item in items:
            batch.append(item)

            if len(batch) >= self.batch_size:
                batch_results = process_func(batch, *args, **kwargs)
                results.extend(batch_results)
                processed_count += len(batch)

                if self.show_progress:
                    logger.info(f"Processed {processed_count} items")

                batch = []

        # Process remaining items
        if batch:
            batch_results = process_func(batch, *args, **kwargs)
            results.extend(batch_results)
            processed_count += len(batch)

            if self.show_progress:
                logger.info(f"Processed {processed_count} items (final batch)")

        return results

class AlignmentBatchProcessor(BatchProcessor):
    """Specialized processor for alignment data."""

    def process_alignments(self, alignments, process_func, *args, **kwargs):
        """Process alignments in batches with specialized handling.

        This method provides optimized memory handling for alignment data,
        which can be memory-intensive.

        Args:
            alignments: Iterable of alignment objects
            process_func: Function to process each batch of alignments
            args, kwargs: Additional arguments for process_func

        Returns:
            Combined results from all batches
        """
        return self.process(alignments, process_func, *args, **kwargs)

    def parallel_process_alignments(self, alignments, process_func, *args, **kwargs):
        """Process alignments in parallel batches if numba is available.

        Args:
            alignments: Iterable of alignment objects
            process_func: JIT-compiled function to process each alignment
            args, kwargs: Additional arguments for process_func

        Returns:
            Combined results from all batches
        """
        # Collect all alignments first
        all_alignments = list(alignments)

        # Create batches
        num_batches = (len(all_alignments) + self.batch_size - 1) // self.batch_size
        batches = [all_alignments[i*self.batch_size:(i+1)*self.batch_size]
                  for i in range(num_batches)]

        if NUMBA_AVAILABLE:
            # Process batches in parallel
            all_results = []
            for batch_idx, batch in enumerate(batches):
                batch_results = parallel_process_chunks(batch, process_func)
                all_results.extend(batch_results)

                if self.show_progress:
                    logger.info(f"Processed batch {batch_idx+1}/{num_batches} ({len(batch)} alignments)")

            return all_results
        else:
            # Fall back to sequential processing
            return self.process(all_alignments, lambda batch, *a, **kw:
                              [process_func(item, *a, **kw) for item in batch],
                              *args, **kwargs)

class ReadBatchProcessor(BatchProcessor):
    """Specialized processor for read data."""

    def process_em_data(self, log_p_rgs, freq, process_func, max_iterations=3, epsilon=1e-4):
        """Process EM algorithm data in batches.

        Args:
            log_p_rgs: Dict mapping query_name to ([tax_ids], [log_scores])
            freq: Dict mapping species_tax_id to likelihood
            process_func: Function to process each batch
            max_iterations: Maximum number of EM iterations per batch
            epsilon: Convergence threshold

        Returns:
            Updated frequency dict and other results
        """
        all_reads = list(log_p_rgs.keys())
        n_reads = len(all_reads)

        # If dataset is small, process directly
        if n_reads <= self.batch_size:
            return process_func(log_p_rgs, freq, max_iterations, epsilon)

        # Split reads into batches
        batch_count = (n_reads + self.batch_size - 1) // self.batch_size
        read_batches = np.array_split(all_reads, batch_count)

        if self.show_progress:
            logger.info(f"Processing {n_reads} reads in {batch_count} batches of ~{self.batch_size} reads each")

        # Initialize accumulators
        total_log_likelihood = 0.0
        taxa_counts = {}
        p_sgr_combined = {}

        # Process each batch
        for batch_idx, batch_reads in enumerate(read_batches):
            if self.show_progress:
                logger.info(f"Processing batch {batch_idx+1}/{batch_count} ({len(batch_reads)} reads)")

            # Create batch subset
            batch_log_p_rgs = {read: log_p_rgs[read] for read in batch_reads}

            # Run EM on this batch
            _, batch_log_likelihood, batch_p_sgr = process_func(
                batch_log_p_rgs, freq, max_iterations, epsilon)

            # Accumulate log likelihood
            total_log_likelihood += batch_log_likelihood

            # Accumulate taxon counts from posterior probabilities
            for taxon, read_probs in batch_p_sgr.items():
                if taxon not in taxa_counts:
                    taxa_counts[taxon] = 0.0

                taxa_counts[taxon] += sum(read_probs.values())

                # Update combined probability matrix
                if taxon not in p_sgr_combined:
                    p_sgr_combined[taxon] = {}
                p_sgr_combined[taxon].update(read_probs)

        # Calculate final frequencies
        updated_freq = {taxon: count / n_reads for taxon, count in taxa_counts.items()}

        # Ensure frequencies sum to 1
        freq_sum = sum(updated_freq.values())
        if freq_sum == 0:
            logger.warning("Zero frequency sum after batch processing")
        elif not 0.99 <= freq_sum <= 1.01:
            logger.warning(f"Normalizing frequency vector from {freq_sum} to 1.0")
            updated_freq = {k: v / freq_sum for k, v in updated_freq.items()}

        return updated_freq, total_log_likelihood, p_sgr_combined