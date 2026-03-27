"""
openGJK GPU - Python wrapper for GPU-accelerated collision detection

High-performance GJK/EPA algorithms running on NVIDIA GPUs with CUDA.
"""

from .opengjk_gpu import (
    USE_32BITS,
    PolytopeArray,
    SimplexArray,
    GpuBatch,
    compute_minimum_distance,
    compute_epa,
    compute_gjk_epa,
)

__version__ = "3.2.0"
__all__ = [
    "USE_32BITS",
    "PolytopeArray",
    "SimplexArray",
    "GpuBatch",
    "compute_minimum_distance",
    "compute_epa",
    "compute_gjk_epa",
]
