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

USE_32BITS = False  # Must match USE_32BITS compile flag

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
    ctypes.c_void_p, # d_simplices
    ctypes.c_void_p, # d_distances
    ctypes.c_void_p, # d_witness1
    ctypes.c_void_p, # d_witness2
    ctypes.c_void_p, # d_contact_normals (nullable)
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

    def extract(self):
        """Return (witnesses1, witnesses2, nvrtx) as numpy arrays."""
        witnesses1 = np.empty((self.n, 3), dtype=DTYPE)
        witnesses2 = np.empty((self.n, 3), dtype=DTYPE)
        nvrtx = np.empty(self.n, dtype=np.int32)
        for i in range(self.n):
            s = self._array[i]
            witnesses1[i] = s.witnesses[0]
            witnesses2[i] = s.witnesses[1]
            nvrtx[i] = s.nvrtx
        return witnesses1, witnesses2, nvrtx


class GpuBatch:
    """
    Owns GPU memory for a fixed set of polytope pairs.

    Upload happens once at construction. GJK and EPA can be called
    repeatedly without re-allocating or re-copying polytope data.

    Args:
        bd1:      First polytope array.
        bd2:      Second polytope array.
        with_epa: Pre-allocate GPU witness/normals buffers for EPA (default False).
    """

    def __init__(
        self,
        bd1: PolytopeArray,
        bd2: PolytopeArray,
        with_epa: bool = False,
    ):
        if bd1.n != bd2.n:
            raise ValueError(f"bd1 has {bd1.n} polytopes but bd2 has {bd2.n}")
        self.n = bd1.n
        self._bd1 = bd1
        self._bd2 = bd2
        self._with_epa = with_epa

        # Host output buffers (pre-allocated, reused across calls)
        self._simplices  = SimplexArray(self.n)
        self._distances  = np.zeros(self.n, dtype=DTYPE)

        # GJK device pointers
        self._d_bd1       = ctypes.c_void_p()
        self._d_bd2       = ctypes.c_void_p()
        self._d_coord1    = ctypes.c_void_p()
        self._d_coord2    = ctypes.c_void_p()
        self._d_simplices = ctypes.c_void_p()
        self._d_distances = ctypes.c_void_p()

        _lib.allocate_and_copy_device_arrays(
            self.n,
            bd1.as_ptr(), bd2.as_ptr(),
            ctypes.byref(self._d_bd1),
            ctypes.byref(self._d_bd2),
            ctypes.byref(self._d_coord1),
            ctypes.byref(self._d_coord2),
            ctypes.byref(self._d_simplices),
            ctypes.byref(self._d_distances),
        )

        # EPA device pointers and host buffers (only GPU memory allocated when requested)
        self._d_witness1        = ctypes.c_void_p()
        self._d_witness2        = ctypes.c_void_p()
        self._d_contact_normals = ctypes.c_void_p()
        self._witnesses1        = np.empty((self.n, 3), dtype=DTYPE)
        self._witnesses2        = np.empty((self.n, 3), dtype=DTYPE)
        self._contact_normals   = np.empty((self.n, 3), dtype=DTYPE)

        if self._with_epa:
            # Always allocate d_contact_normals — the EPA kernel writes to it
            # unconditionally in its non-collision branch, so passing null
            # causes a GPU memory error that corrupts all subsequent results.
            _lib.allocate_epa_device_arrays(
                self.n,
                ctypes.byref(self._d_witness1),
                ctypes.byref(self._d_witness2),
                ctypes.byref(self._d_contact_normals),
            )

    def compute_gjk(self) -> dict:
        _lib.compute_minimum_distance_device(
            self.n,
            self._d_bd1, self._d_bd2,
            self._d_simplices, self._d_distances,
        )
        _lib.copy_results_from_device(
            self.n,
            self._d_simplices, self._d_distances,
            self._simplices.as_ptr(),
            self._distances.ctypes.data_as(ctypes.POINTER(gkFloat)),
        )
        witnesses1, witnesses2, nvrtx = self._simplices.extract()
        return {
            'distances':     self._distances.copy(),
            'witnesses1':    witnesses1,
            'witnesses2':    witnesses2,
            'is_collision':  np.abs(self._distances) < 1e-6,
            'simplex_nvrtx': nvrtx,
        }

    def compute_epa(self) -> dict:
        if not self._with_epa:
            raise RuntimeError("GpuBatch was created without with_epa=True")

        # GJK must run first — EPA expands the GJK simplex
        _lib.compute_minimum_distance_device(
            self.n,
            self._d_bd1, self._d_bd2,
            self._d_simplices, self._d_distances,
        )
        _lib.compute_epa_device(
            self.n,
            self._d_bd1, self._d_bd2,
            self._d_simplices, self._d_distances,
            self._d_witness1, self._d_witness2,
            self._d_contact_normals,  # null c_void_p if not with_normals
        )
        _lib.copy_epa_results_from_device(
            self.n,
            self._d_witness1, self._d_witness2, self._d_contact_normals,
            self._witnesses1.ctypes.data_as(ctypes.POINTER(gkFloat)),
            self._witnesses2.ctypes.data_as(ctypes.POINTER(gkFloat)),
            self._contact_normals.ctypes.data_as(ctypes.POINTER(gkFloat)),
        )
        # Copy distances back too (EPA overwrites them with penetration depths)
        _lib.copy_results_from_device(
            self.n,
            self._d_simplices, self._d_distances,
            self._simplices.as_ptr(),
            self._distances.ctypes.data_as(ctypes.POINTER(gkFloat)),
        )
        return {
            'penetration_depths': self._distances.copy(),
            'witnesses1':         self._witnesses1.copy(),
            'witnesses2':         self._witnesses2.copy(),
            'contact_normals':    self._contact_normals.copy(),
        }

    def compute_gjk_epa(self) -> dict:
        if not self._with_epa:
            raise RuntimeError("GpuBatch was created without with_epa=True")

        gjk = self.compute_gjk()

        _lib.compute_epa_device(
            self.n,
            self._d_bd1, self._d_bd2,
            self._d_simplices, self._d_distances,
            self._d_witness1, self._d_witness2,
            self._d_contact_normals,
        )
        _lib.copy_epa_results_from_device(
            self.n,
            self._d_witness1, self._d_witness2, self._d_contact_normals,
            self._witnesses1.ctypes.data_as(ctypes.POINTER(gkFloat)),
            self._witnesses2.ctypes.data_as(ctypes.POINTER(gkFloat)),
            self._contact_normals.ctypes.data_as(ctypes.POINTER(gkFloat)),
        )
        _lib.copy_results_from_device(
            self.n,
            self._d_simplices, self._d_distances,
            self._simplices.as_ptr(),
            self._distances.ctypes.data_as(ctypes.POINTER(gkFloat)),
        )

        # Colliding pairs: use EPA witnesses + penetration depth
        # Separated pairs: use GJK witnesses + distance
        is_col = gjk['is_collision']
        witnesses1 = np.where(is_col[:, None], self._witnesses1, gjk['witnesses1'])
        witnesses2 = np.where(is_col[:, None], self._witnesses2, gjk['witnesses2'])

        return {
            'distances':        self._distances.copy(),
            'is_collision':     is_col,
            'witnesses1':       witnesses1,
            'witnesses2':       witnesses2,
            'contact_normals':  self._contact_normals.copy(),
            'simplex_nvrtx':    gjk['simplex_nvrtx'],
        }

    def __del__(self):
        if hasattr(self, '_d_bd1') and self._d_bd1.value:
            _lib.free_device_arrays(
                self._d_bd1, self._d_bd2,
                self._d_coord1, self._d_coord2,
                self._d_simplices, self._d_distances,
            )
        if hasattr(self, '_with_epa') and self._with_epa and self._d_witness1.value:
            _lib.free_epa_device_arrays(
                self._d_witness1,
                self._d_witness2,
                self._d_contact_normals,
            )


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
    return GpuBatch(bd1, bd2).compute_gjk()


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

    pairs = np.asarray(pairs, dtype=np.int32)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"pairs must have shape (num_pairs, 2), got {pairs.shape}")
    num_pairs = pairs.shape[0]

    pairs_array = (gkCollisionPair * num_pairs)()
    for i in range(num_pairs):
        pairs_array[i].idx1 = int(pairs[i, 0])
        pairs_array[i].idx2 = int(pairs[i, 1])

    simplices = SimplexArray(num_pairs)
    distances = np.zeros(num_pairs, dtype=DTYPE)

    _lib.compute_minimum_distance_indexed(
        bd.n,
        num_pairs,
        bd.as_ptr(),
        ctypes.cast(pairs_array, ctypes.POINTER(gkCollisionPair)),
        simplices.as_ptr(),
        distances.ctypes.data_as(ctypes.POINTER(gkFloat)),
    )

    witnesses1, witnesses2, nvrtx = simplices.extract()
    return {
        'distances':     distances,
        'witnesses1':    witnesses1,
        'witnesses2':    witnesses2,
        'is_collision':  np.abs(distances) < 1e-6,
        'simplex_nvrtx': nvrtx,
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
    return GpuBatch(bd1, bd2, with_epa=True).compute_epa()


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
    return GpuBatch(bd1, bd2, with_epa=True).compute_gjk_epa()


# ============================================================================
# Module Info
# ============================================================================

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
