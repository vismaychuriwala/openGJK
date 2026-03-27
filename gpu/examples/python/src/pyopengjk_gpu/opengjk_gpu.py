"""
openGJK GPU - Python wrapper

GPU-accelerated GJK collision detection library for batch processing.

Copyright 2022-2026 Mattia Montanari, University of Oxford
Copyright 2025-2026 Vismay Churiwala, Marcus Hedlund
SPDX-License-Identifier: GPL-3.0-only
"""

import ctypes
import os
import sys
import numpy as np
from typing import List, Union


# ============================================================================
# Precision
# ============================================================================

USE_32BITS = True  # Must match USE_32BITS compile flag

if USE_32BITS:
    gkFloat = ctypes.c_float
    DTYPE = np.float32
else:
    gkFloat = ctypes.c_double
    DTYPE = np.float64


# ============================================================================
# C Structure Definitions
# ============================================================================

class gkPolytope(ctypes.Structure):
    _fields_ = [
        ("numpoints", ctypes.c_int),
        ("s",         gkFloat * 3),
        ("s_idx",     ctypes.c_int),
        ("coord",     ctypes.POINTER(gkFloat)),
    ]


class gkSimplex(ctypes.Structure):
    _fields_ = [
        ("nvrtx",     ctypes.c_int),
        ("vrtx",      (gkFloat * 3) * 4),
        ("vrtx_idx",  (ctypes.c_int * 2) * 4),
        ("witnesses", (gkFloat * 3) * 2),
    ]


class gkCollisionPair(ctypes.Structure):
    _fields_ = [
        ("idx1", ctypes.c_int),
        ("idx2", ctypes.c_int),
    ]


# ============================================================================
# Load Library
# ============================================================================

def _find_library():
    module_dir = os.path.dirname(os.path.abspath(__file__))
    search_paths = [
        module_dir,
        os.path.join(module_dir, "..", "..", "..", "..", "..", "build", "gpu", "Release"),
        os.path.join(module_dir, "..", "..", "..", "..", "..", "build", "gpu"),
        os.path.join(module_dir, "..", "..", "..", "..", "Release"),
        os.path.join(module_dir, "..", "..", "..", ".."),
    ]
    if sys.platform == "win32":
        lib_names = ["openGJK_GPU.dll"]
    elif sys.platform == "darwin":
        lib_names = ["libopenGJK_GPU.dylib", "libopenGJK_GPU.so"]
    else:
        lib_names = ["libopenGJK_GPU.so"]

    for path in search_paths:
        for name in lib_names:
            full = os.path.join(path, name)
            if os.path.exists(full):
                return full

    raise RuntimeError(
        "Could not find openGJK_GPU shared library. Searched:\n" +
        "\n".join(f"  {p}/{n}" for p in search_paths for n in lib_names)
    )


_lib = ctypes.CDLL(_find_library())


# ============================================================================
# CUDA Runtime (device memory management for GpuBatch)
# ============================================================================

_cudart = None

def _get_cudart():
    global _cudart
    if _cudart is not None:
        return _cudart
    names = (
        ["cudart64_120.dll", "cudart64_12.dll", "cudart64_110.dll"]
        if sys.platform == "win32"
        else ["libcudart.so.12", "libcudart.so.11", "libcudart.so"]
    )
    for name in names:
        try:
            lib = ctypes.CDLL(name)
            lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
            lib.cudaMalloc.restype  = ctypes.c_int
            lib.cudaFree.argtypes   = [ctypes.c_void_p]
            lib.cudaFree.restype    = ctypes.c_int
            _cudart = lib
            return lib
        except OSError:
            pass
    raise RuntimeError("Could not load CUDA runtime library (cudart)")

def _cuda_malloc(ptr_ref, size):
    _get_cudart().cudaMalloc(ptr_ref, size)

def _cuda_free(ptr):
    if ptr.value:
        _get_cudart().cudaFree(ptr)
        ptr.value = None


# ============================================================================
# Function Signatures
# ============================================================================

# --- High-level (used only for indexed, which has no mid-level equivalent) ---

_lib.compute_minimum_distance_indexed.argtypes = [
    ctypes.c_int,                    # num_polytopes
    ctypes.c_int,                    # num_pairs
    ctypes.POINTER(gkPolytope),      # polytopes
    ctypes.POINTER(gkCollisionPair), # pairs
    ctypes.POINTER(gkSimplex),       # simplices
    ctypes.POINTER(gkFloat),         # distances
]
_lib.compute_minimum_distance_indexed.restype = None

# --- Mid-level: GJK ---

_lib.allocate_and_copy_device_arrays.argtypes = [
    ctypes.c_int,                    # n
    ctypes.POINTER(gkPolytope),      # bd1 (host)
    ctypes.POINTER(gkPolytope),      # bd2 (host)
    ctypes.POINTER(ctypes.c_void_p), # d_bd1 (out)
    ctypes.POINTER(ctypes.c_void_p), # d_bd2 (out)
    ctypes.POINTER(ctypes.c_void_p), # d_coord1 (out)
    ctypes.POINTER(ctypes.c_void_p), # d_coord2 (out)
    ctypes.POINTER(ctypes.c_void_p), # d_simplices (out)
    ctypes.POINTER(ctypes.c_void_p), # d_distances (out)
]
_lib.allocate_and_copy_device_arrays.restype = None

_lib.compute_minimum_distance_device.argtypes = [
    ctypes.c_int,    # n
    ctypes.c_void_p, # d_bd1
    ctypes.c_void_p, # d_bd2
    ctypes.c_void_p, # d_simplices
    ctypes.c_void_p, # d_distances
]
_lib.compute_minimum_distance_device.restype = None

_lib.copy_results_from_device.argtypes = [
    ctypes.c_int,               # n
    ctypes.c_void_p,            # d_simplices
    ctypes.c_void_p,            # d_distances
    ctypes.POINTER(gkSimplex),  # simplices (host out)
    ctypes.POINTER(gkFloat),    # distances (host out)
]
_lib.copy_results_from_device.restype = None

_lib.free_device_arrays.argtypes = [
    ctypes.c_void_p, # d_bd1
    ctypes.c_void_p, # d_bd2
    ctypes.c_void_p, # d_coord1
    ctypes.c_void_p, # d_coord2
    ctypes.c_void_p, # d_simplices
    ctypes.c_void_p, # d_distances
]
_lib.free_device_arrays.restype = None

# --- Mid-level: EPA ---

_lib.allocate_epa_device_arrays.argtypes = [
    ctypes.c_int,                    # n
    ctypes.POINTER(ctypes.c_void_p), # d_witness1 (out)
    ctypes.POINTER(ctypes.c_void_p), # d_witness2 (out)
    ctypes.POINTER(ctypes.c_void_p), # d_contact_normals (out, nullable)
]
_lib.allocate_epa_device_arrays.restype = None

_lib.compute_epa_device.argtypes = [
    ctypes.c_int,    # n
    ctypes.c_void_p, # d_bd1
    ctypes.c_void_p, # d_bd2
    ctypes.c_void_p, # d_simplices (witnesses -> simplices[i].witnesses[0/1])
    ctypes.c_void_p, # d_distances
    ctypes.c_void_p, # d_contact_normals
]
_lib.compute_epa_device.restype = None

_lib.copy_epa_results_from_device.argtypes = [
    ctypes.c_int,            # n
    ctypes.c_void_p,         # d_witness1
    ctypes.c_void_p,         # d_witness2
    ctypes.c_void_p,         # d_contact_normals (nullable)
    ctypes.POINTER(gkFloat), # witness1 (host out)
    ctypes.POINTER(gkFloat), # witness2 (host out)
    ctypes.POINTER(gkFloat), # contact_normals (host out, nullable)
]
_lib.copy_epa_results_from_device.restype = None

_lib.free_epa_device_arrays.argtypes = [
    ctypes.c_void_p, # d_witness1
    ctypes.c_void_p, # d_witness2
    ctypes.c_void_p, # d_contact_normals (nullable)
]
_lib.free_epa_device_arrays.restype = None

# --- High-level: EPA ---

_lib.computeCollisionInformation.argtypes = [
    ctypes.c_int,               # n
    ctypes.POINTER(gkPolytope), # bd1
    ctypes.POINTER(gkPolytope), # bd2
    ctypes.POINTER(gkSimplex),  # simplices (in/out); witnesses -> simplices[i].witnesses[0/1]
    ctypes.POINTER(gkFloat),    # distances (in/out)
    ctypes.POINTER(gkFloat),    # contact_normals (n*3 floats)
]
_lib.computeCollisionInformation.restype = None

_lib.compute_gjk_epa.argtypes = [
    ctypes.c_int,               # n
    ctypes.POINTER(gkPolytope), # bd1
    ctypes.POINTER(gkPolytope), # bd2
    ctypes.POINTER(gkSimplex),  # simplices (out); witnesses -> simplices[i].witnesses[0/1]
    ctypes.POINTER(gkFloat),    # distances (out)
    ctypes.POINTER(gkFloat),    # contact_normals (n*3 floats)
]
_lib.compute_gjk_epa.restype = None

# --- Mid-level: indexed pool ---

_lib.allocate_and_copy_indexed_polytopes.argtypes = [
    ctypes.c_int,                    # num_polytopes
    ctypes.POINTER(gkPolytope),      # polytopes (host)
    ctypes.POINTER(ctypes.c_void_p), # d_polytopes (out)
    ctypes.POINTER(ctypes.c_void_p), # d_coords (out)
]
_lib.allocate_and_copy_indexed_polytopes.restype = None

_lib.upload_pairs_device.argtypes = [
    ctypes.c_int,                    # num_pairs
    ctypes.POINTER(gkCollisionPair), # pairs (host)
    ctypes.c_void_p,                 # d_pairs (pre-allocated device)
]
_lib.upload_pairs_device.restype = None

_lib.compute_minimum_distance_indexed_device.argtypes = [
    ctypes.c_int,    # num_pairs
    ctypes.c_void_p, # d_polytopes
    ctypes.c_void_p, # d_pairs
    ctypes.c_void_p, # d_simplices
    ctypes.c_void_p, # d_distances
]
_lib.compute_minimum_distance_indexed_device.restype = None

_lib.compute_epa_indexed_device.argtypes = [
    ctypes.c_int,    # num_pairs
    ctypes.c_void_p, # d_polytopes
    ctypes.c_void_p, # d_pairs
    ctypes.c_void_p, # d_simplices (witnesses -> simplices[i].witnesses[0/1])
    ctypes.c_void_p, # d_distances
    ctypes.c_void_p, # d_contact_normals
]
_lib.compute_epa_indexed_device.restype = None


# ============================================================================
# Python Objects
# ============================================================================

class PolytopeArray:
    """
    Batch of polytopes backed by a single contiguous coord buffer.

    Accepts a list of (n_i, 3) arrays — vertex counts may vary per polytope.
    Also accepts a 3D ndarray of shape (N, num_verts, 3) for uniform batches.
    """

    def __init__(self, vertices_list: List[np.ndarray]):
        n = len(vertices_list)
        self.n = n

        counts = [v.shape[0] for v in vertices_list]
        total = sum(counts) * 3
        self._all_coords = np.empty(total, dtype=DTYPE)
        self._array = (gkPolytope * n)()

        offset = 0
        for i, verts in enumerate(vertices_list):
            verts = np.asarray(verts, dtype=DTYPE)
            if verts.ndim != 2 or verts.shape[1] != 3:
                raise ValueError(f"Polytope {i}: expected shape (n, 3), got {verts.shape}")
            count = verts.shape[0] * 3
            self._all_coords[offset:offset + count] = verts.ravel()
            self._array[i].numpoints = verts.shape[0]
            self._array[i].s = (gkFloat * 3)(0, 0, 0)
            self._array[i].s_idx = 0
            self._array[i].coord = self._all_coords[offset:].ctypes.data_as(
                ctypes.POINTER(gkFloat)
            )
            offset += count

    def as_ptr(self):
        return ctypes.cast(self._array, ctypes.POINTER(gkPolytope))


class SimplexArray:
    """Batch of GJK simplex results."""

    def __init__(self, n: int):
        self.n = n
        self._array = (gkSimplex * n)()

    def as_ptr(self):
        return ctypes.cast(self._array, ctypes.POINTER(gkSimplex))

    def extract(self, n=None):
        """Return (witnesses1, witnesses2) as numpy arrays for the first n entries."""
        if n is None:
            n = self.n
        stride  = ctypes.sizeof(gkSimplex)
        w_off   = gkSimplex.witnesses.offset  # byte offset of witnesses field
        raw = np.frombuffer(
            (ctypes.c_byte * (n * stride)).from_address(ctypes.addressof(self._array[0])),
            dtype=DTYPE,
        ).reshape(n, stride // ctypes.sizeof(gkFloat))
        w_col = w_off // ctypes.sizeof(gkFloat)
        witnesses = raw[:, w_col : w_col + 6].reshape(n, 2, 3)
        return witnesses[:, 0, :].copy(), witnesses[:, 1, :].copy()


class GpuBatch:
    """
    Holds a pool of polytopes in GPU memory for repeated indexed collision queries.

    Polytope data is uploaded once at construction. Call compute() or compute_epa()
    with different index pair arrays to run GJK/EPA without re-uploading polytope data.

    Args:
        polytopes: PolytopeArray, (M, V, 3) ndarray, or list of (V_i, 3) arrays.
        max_pairs: Maximum number of collision pairs per compute call.
        with_epa:  Pre-allocate contact-normals buffer for EPA (default False).
    """

    def __init__(self, polytopes, max_pairs: int, with_epa: bool = False):
        self._bd       = _to_polytope_array(polytopes)
        self.max_pairs = max_pairs
        self._with_epa = with_epa

        # Upload pool to device once
        self._d_polytopes = ctypes.c_void_p()
        self._d_coords    = ctypes.c_void_p()
        _lib.allocate_and_copy_indexed_polytopes(
            self._bd.n,
            self._bd.as_ptr(),
            ctypes.byref(self._d_polytopes),
            ctypes.byref(self._d_coords),
        )

        # Pre-allocate device buffers sized for max_pairs
        self._d_pairs     = ctypes.c_void_p()
        self._d_simplices = ctypes.c_void_p()
        self._d_distances = ctypes.c_void_p()
        _cuda_malloc(ctypes.byref(self._d_pairs),     max_pairs * ctypes.sizeof(gkCollisionPair))
        _cuda_malloc(ctypes.byref(self._d_simplices), max_pairs * ctypes.sizeof(gkSimplex))
        _cuda_malloc(ctypes.byref(self._d_distances), max_pairs * ctypes.sizeof(gkFloat))

        # Host result buffers (reused across calls)
        self._simplices = SimplexArray(max_pairs)
        self._distances = np.zeros(max_pairs, dtype=DTYPE)

        # EPA device + host buffers
        self._d_contact_normals = ctypes.c_void_p()
        self._contact_normals   = np.empty((max_pairs, 3), dtype=DTYPE)
        if with_epa:
            _cuda_malloc(ctypes.byref(self._d_contact_normals), max_pairs * 3 * ctypes.sizeof(gkFloat))

    def _upload_pairs(self, pairs: np.ndarray) -> int:
        pairs = np.ascontiguousarray(pairs, dtype=np.int32)
        n = pairs.shape[0]
        if n > self.max_pairs:
            raise ValueError(f"pairs count {n} exceeds max_pairs {self.max_pairs}")
        _lib.upload_pairs_device(
            n,
            pairs.ctypes.data_as(ctypes.POINTER(gkCollisionPair)),
            self._d_pairs,
        )
        return n

    def compute(self, pairs: np.ndarray) -> dict:
        pairs = np.asarray(pairs, dtype=np.int32).reshape(-1, 2)
        n = self._upload_pairs(pairs)
        _lib.compute_minimum_distance_indexed_device(
            n,
            self._d_polytopes, self._d_pairs,
            self._d_simplices, self._d_distances,
        )
        _lib.copy_results_from_device(
            n,
            self._d_simplices, self._d_distances,
            self._simplices.as_ptr(),
            self._distances.ctypes.data_as(ctypes.POINTER(gkFloat)),
        )
        witnesses1, witnesses2 = self._simplices.extract(n)
        return {
            'distances':    self._distances[:n].copy(),
            'witnesses1':   witnesses1,
            'witnesses2':   witnesses2,
            'is_collision': np.abs(self._distances[:n]) < 1e-6,
        }

    def compute_epa(self, pairs: np.ndarray) -> dict:
        if not self._with_epa:
            raise RuntimeError("GpuBatch was created without with_epa=True")
        pairs = np.asarray(pairs, dtype=np.int32).reshape(-1, 2)
        n = self._upload_pairs(pairs)
        _lib.compute_minimum_distance_indexed_device(
            n,
            self._d_polytopes, self._d_pairs,
            self._d_simplices, self._d_distances,
        )
        _lib.compute_epa_indexed_device(
            n,
            self._d_polytopes, self._d_pairs,
            self._d_simplices, self._d_distances,
            self._d_contact_normals,
        )
        _lib.copy_results_from_device(
            n,
            self._d_simplices, self._d_distances,
            self._simplices.as_ptr(),
            self._distances.ctypes.data_as(ctypes.POINTER(gkFloat)),
        )
        _lib.copy_epa_results_from_device(
            n,
            ctypes.c_void_p(), ctypes.c_void_p(), self._d_contact_normals,
            ctypes.c_void_p(), ctypes.c_void_p(),
            self._contact_normals.ctypes.data_as(ctypes.POINTER(gkFloat)),
        )
        witnesses1, witnesses2 = self._simplices.extract(n)
        return {
            'penetration_depths': self._distances[:n].copy(),
            'witnesses1':         witnesses1,
            'witnesses2':         witnesses2,
            'contact_normals':    self._contact_normals[:n].copy(),
        }

    def __del__(self):
        for attr in ('_d_polytopes', '_d_coords', '_d_pairs', '_d_simplices', '_d_distances'):
            if hasattr(self, attr):
                _cuda_free(getattr(self, attr))
        if hasattr(self, '_with_epa') and self._with_epa:
            _cuda_free(self._d_contact_normals)


# ============================================================================
# Input Normalization
# ============================================================================

def _to_polytope_array(v) -> PolytopeArray:
    if isinstance(v, PolytopeArray):
        return v
    if isinstance(v, np.ndarray):
        if v.ndim == 3:
            return PolytopeArray([v[i] for i in range(v.shape[0])])
        if v.ndim == 2:
            return PolytopeArray([v])
    if isinstance(v, list):
        return PolytopeArray(v)
    raise ValueError(f"Expected PolytopeArray, ndarray, or list — got {type(v)}")


# ============================================================================
# Indexed Batch
# ============================================================================

class IndexedBatch:
    """
    Holds a pool of polytopes for repeated indexed GJK queries.

    Packs vertex data into a contiguous host buffer once at construction.
    Call compute(pairs) with different pair index arrays — of any size — to
    run GJK without re-packing vertex data each time.

    Args:
        polytopes: PolytopeArray, (M, D, 3) ndarray, or list of (V_i, 3) arrays
    """

    def __init__(self, polytopes: Union['PolytopeArray', np.ndarray, list]):
        self._bd = _to_polytope_array(polytopes)

    def compute(self, pairs: np.ndarray) -> dict:
        """
        Run GJK for the given index pairs.

        Args:
            pairs: (n_pairs, 2) int32 array of indices into the polytope pool

        Returns:
            Same keys as GpuBatch.compute_gjk().
        """
        pairs = np.ascontiguousarray(pairs, dtype=np.int32).reshape(-1, 2)
        num_pairs = pairs.shape[0]
        simplices = SimplexArray(num_pairs)
        distances = np.zeros(num_pairs, dtype=DTYPE)
        _lib.compute_minimum_distance_indexed(
            self._bd.n, num_pairs, self._bd.as_ptr(),
            pairs.ctypes.data_as(ctypes.POINTER(gkCollisionPair)),
            simplices.as_ptr(),
            distances.ctypes.data_as(ctypes.POINTER(gkFloat)),
        )
        witnesses1, witnesses2 = simplices.extract()
        return {
            'distances':    distances,
            'witnesses1':   witnesses1,
            'witnesses2':   witnesses2,
            'is_collision': np.abs(distances) < 1e-6,
        }


# ============================================================================
# Public API
# ============================================================================

def compute_minimum_distance(
    vertices1: Union['PolytopeArray', np.ndarray, list],
    vertices2: Union['PolytopeArray', np.ndarray, list],
) -> dict:
    """
    Compute minimum distance between polytope pairs using GPU GJK.

    Args:
        vertices1: PolytopeArray, (N, V, 3) ndarray, (V, 3) ndarray, or list of (V_i, 3) arrays
        vertices2: same

    Returns:
        'distances':     (N,) distances (0.0 = collision)
        'witnesses1':    (N, 3) closest points on first polytopes
        'witnesses2':    (N, 3) closest points on second polytopes
        'is_collision':  (N,) bool
        'simplex_nvrtx': (N,) simplex vertex counts
    """
    bd1 = _to_polytope_array(vertices1)
    bd2 = _to_polytope_array(vertices2)
    n = bd1.n
    simplices = SimplexArray(n)
    distances = np.zeros(n, dtype=DTYPE)
    _lib.compute_minimum_distance(
        n, bd1.as_ptr(), bd2.as_ptr(),
        simplices.as_ptr(),
        distances.ctypes.data_as(ctypes.POINTER(gkFloat)),
    )
    witnesses1, witnesses2 = simplices.extract()
    return {
        'distances':    distances,
        'witnesses1':   witnesses1,
        'witnesses2':   witnesses2,
        'is_collision': np.abs(distances) < 1e-6,
    }


def compute_minimum_distance_indexed(
    polytopes: Union['PolytopeArray', np.ndarray, list],
    pairs: np.ndarray,
) -> dict:
    """
    Compute distances for indexed polytope pairs.

    Args:
        polytopes: PolytopeArray, (M, V, 3) ndarray, or list of (V_i, 3) arrays
        pairs:     (num_pairs, 2) int32 array of polytope index pairs

    Returns:
        Same keys as compute_minimum_distance.
    """
    bd = _to_polytope_array(polytopes)
    pairs = np.ascontiguousarray(pairs, dtype=np.int32).reshape(-1, 2)
    num_pairs = pairs.shape[0]
    simplices = SimplexArray(num_pairs)
    distances = np.zeros(num_pairs, dtype=DTYPE)
    _lib.compute_minimum_distance_indexed(
        bd.n, num_pairs, bd.as_ptr(),
        pairs.ctypes.data_as(ctypes.POINTER(gkCollisionPair)),
        simplices.as_ptr(),
        distances.ctypes.data_as(ctypes.POINTER(gkFloat)),
    )
    witnesses1, witnesses2 = simplices.extract()
    return {
        'distances':    distances,
        'witnesses1':   witnesses1,
        'witnesses2':   witnesses2,
        'is_collision': np.abs(distances) < 1e-6,
    }


def compute_epa(
    vertices1: Union['PolytopeArray', np.ndarray, list],
    vertices2: Union['PolytopeArray', np.ndarray, list],
) -> dict:
    """
    Compute penetration depth and witness points using GPU EPA.

    Args:
        vertices1: PolytopeArray, (N, V, 3) ndarray, or list of (V_i, 3) arrays
        vertices2: same

    Returns:
        'penetration_depths': (N,) penetration distances
        'witnesses1':         (N, 3) contact points on first polytopes
        'witnesses2':         (N, 3) contact points on second polytopes
        'contact_normals':    (N, 3) contact normals
    """
    bd1 = _to_polytope_array(vertices1)
    bd2 = _to_polytope_array(vertices2)
    n = bd1.n
    simplices = SimplexArray(n)
    distances = np.zeros(n, dtype=DTYPE)
    contact_normals = np.empty((n, 3), dtype=DTYPE)
    _lib.computeCollisionInformation(
        n, bd1.as_ptr(), bd2.as_ptr(),
        simplices.as_ptr(),
        distances.ctypes.data_as(ctypes.POINTER(gkFloat)),
        contact_normals.ctypes.data_as(ctypes.POINTER(gkFloat)),
    )
    witnesses1, witnesses2 = simplices.extract()
    return {
        'penetration_depths': distances,
        'witnesses1':         witnesses1,
        'witnesses2':         witnesses2,
        'contact_normals':    contact_normals,
    }


def compute_gjk_epa(
    vertices1: Union['PolytopeArray', np.ndarray, list],
    vertices2: Union['PolytopeArray', np.ndarray, list],
) -> dict:
    """
    Combined GJK + EPA: GJK for separation distance, EPA for penetration depth.

    Returns GJK witnesses for separated pairs and EPA contact points for colliding pairs.

    Args:
        vertices1: PolytopeArray, (N, V, 3) ndarray, or list of (V_i, 3) arrays
        vertices2: same

    Returns:
        'distances':     (N,) distances (negative = penetration depth)
        'is_collision':  (N,) bool
        'witnesses1':    (N, 3) witness/contact points on first polytopes
        'witnesses2':    (N, 3) witness/contact points on second polytopes
        'simplex_nvrtx': (N,) simplex vertex counts
    """
    bd1 = _to_polytope_array(vertices1)
    bd2 = _to_polytope_array(vertices2)
    n = bd1.n
    simplices = SimplexArray(n)
    distances = np.zeros(n, dtype=DTYPE)
    contact_normals = np.empty((n, 3), dtype=DTYPE)
    _lib.compute_gjk_epa(
        n, bd1.as_ptr(), bd2.as_ptr(),
        simplices.as_ptr(),
        distances.ctypes.data_as(ctypes.POINTER(gkFloat)),
        contact_normals.ctypes.data_as(ctypes.POINTER(gkFloat)),
    )
    witnesses1, witnesses2 = simplices.extract()
    return {
        'distances':       distances,
        'is_collision':    np.abs(distances) < 1e-6,
        'witnesses1':      witnesses1,
        'witnesses2':      witnesses2,
        'contact_normals': contact_normals,
    }


# ============================================================================
# Module Info
# ============================================================================

__version__ = "3.1.0"
__all__ = [
    "PolytopeArray",
    "SimplexArray",
    "GpuBatch",
    "IndexedBatch",
    "compute_minimum_distance",
    "compute_minimum_distance_indexed",
    "compute_epa",
    "compute_gjk_epa",
]
