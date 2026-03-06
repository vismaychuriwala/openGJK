"""
openGJK GPU - Python wrapper for GPU-accelerated collision detection

High-performance GJK/EPA algorithms running on NVIDIA GPUs with CUDA.
"""

from .opengjk_gpu import (
    PolytopeArray,
    SimplexArray,
    GpuBatch,
    compute_minimum_distance,
    compute_minimum_distance_indexed,
    compute_epa,
    compute_gjk_epa,
)

__version__ = "3.1.0"
__all__ = [
    "PolytopeArray",
    "SimplexArray",
    "GpuBatch",
    "compute_minimum_distance",
    "compute_minimum_distance_indexed",
    "compute_epa",
    "compute_gjk_epa",
]
