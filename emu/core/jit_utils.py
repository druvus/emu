"""Centralized JIT compilation utilities for Emu."""

import numpy as np
import logging

logger = logging.getLogger(__name__)

# Try to import numba, but provide fallbacks if not available
try:
    from numba import jit, njit, prange, float32, int32, float64, int64
    NUMBA_AVAILABLE = True
    logger.debug("Numba is available for JIT compilation")
except ImportError:
    NUMBA_AVAILABLE = False
    logger.debug("Numba not available, using fallback implementations")

    def jit(*args, **kwargs):
        """Fallback for jit when numba not available."""
        if len(args) > 0 and callable(args[0]):
            return args[0]
        else:
            def decorator(func):
                return func
            return decorator

    njit = jit

    def prange(*args):
        """Fallback for prange."""
        return range(*args)

    # Define type aliases for consistency
    float32 = float
    int32 = int
    float64 = float
    int64 = int

# Common JIT-optimized functions for matrix operations
if NUMBA_AVAILABLE:
    @njit
    def sum_array(arr):
        """JIT-optimized sum of array elements."""
        result = 0.0
        for i in range(len(arr)):
            result += arr[i]
        return result

    @njit
    def vector_dot_product(vec1, vec2):
        """Compute dot product of two vectors with JIT."""
        result = 0.0
        for i in range(len(vec1)):
            result += vec1[i] * vec2[i]
        return result

    @njit
    def vector_multiply(vec, scalar):
        """Multiply a vector by a scalar with JIT."""
        result = np.zeros_like(vec)
        for i in range(len(vec)):
            result[i] = vec[i] * scalar
        return result

    @njit(parallel=True)
    def parallel_process_chunks(chunks, function):
        """Process chunks in parallel with JIT.

        Args:
            chunks: List of data chunks
            function: JIT-compiled function to apply to each chunk

        Returns:
            List of processed chunks
        """
        result = [None] * len(chunks)

        for i in prange(len(chunks)):
            result[i] = function(chunks[i])

        return result
else:
    # Non-JIT fallbacks
    def sum_array(arr):
        """Standard sum of array elements."""
        return sum(arr)

    def vector_dot_product(vec1, vec2):
        """Standard dot product."""
        return np.dot(vec1, vec2)

    def vector_multiply(vec, scalar):
        """Standard vector multiplication."""
        return vec * scalar

    def parallel_process_chunks(chunks, function):
        """Standard sequential processing of chunks."""
        return [function(chunk) for chunk in chunks]

# CIGAR operation constants as numpy arrays for JIT functions
CIGAR_OPS_NP = np.array([1, 2, 4, 10], dtype=np.int32)  # I, D, S, X
CIGAR_OPS_ALL_NP = np.array([0, 1, 2, 4], dtype=np.int32)  # M, I, D, S