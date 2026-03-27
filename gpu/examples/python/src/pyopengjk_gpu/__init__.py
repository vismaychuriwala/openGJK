"""
openGJK GPU - Python wrapper for GPU-accelerated collision detection

High-performance GJK/EPA algorithms running on NVIDIA GPUs with CUDA.
"""

from .opengjk_gpu import (
    USE_32BITS,
    PolytopeArray,
    SimplexArray,
    GpuBatch,
    IndexedBatch,
    compute_minimum_distance,
    compute_minimum_distance_indexed,
    computeCollisionInformation,
    compute_gjk_epa,
)

__version__ = "3.1.0"
__all__ = [
    "USE_32BITS",
    "PolytopeArray",
    "SimplexArray",
    "GpuBatch",
    "IndexedBatch",
    "compute_minimum_distance",
    "compute_minimum_distance_indexed",
    "computeCollisionInformation",
    "compute_gjk_epa",
]
