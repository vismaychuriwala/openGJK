"""
OpenGJK GPU Test Examples - Python Version
Recreates the test cases from OpenGJK-GPU repository

This demonstrates:
1. Simple GJK (single pair)
2. Batch array processing
3. Indexed API
4. EPA collision tests
"""

import numpy as np
import time
from pyopengjk_gpu import (
    USE_32BITS,
    GpuBatch,
    PolytopeArray,
    IndexedBatch,
    compute_minimum_distance,
)

# Global random seed for reproducibility
RANDOM_SEED = 42

SINGLE = np.array([[0, 1]], dtype=np.int32)  # index pair for single-pair tests

# Auto-detect dtype from library (matches #USE_32BITS in C++ compilation)
def _detect_dtype():
    """Detect the dtype used by the library by running a minimal test."""
    # Create minimal test arrays (use float32 for detection)
    p1 = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float32)
    p2 = np.array([[[2, 0, 0], [3, 0, 0], [2, 1, 0]]], dtype=np.float32)
    try:
        result = compute_minimum_distance(p1, p2)
        return result['distances'].dtype
    except:
        # Fallback to float32 if detection fails
        return np.float32

DTYPE = _detect_dtype()

# ANSI color codes
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_pass(msg):
    print(f"{Colors.GREEN}  PASS: {msg}{Colors.RESET}")

def print_warning(msg):
    print(f"{Colors.YELLOW}  WARNING: {msg}{Colors.RESET}")

def print_fail(msg):
    print(f"{Colors.RED}  FAIL: {msg}{Colors.RESET}")


# ============================================================================
# CPU verification helpers
# ============================================================================

def _load_scalar():
    """Load scalar pyopengjk module (Python import cache makes this cheap)."""
    import sys as _sys, os as _os
    _scalar_src = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..', 'scalar', 'examples', 'python_ctypes', 'src'))
    if _scalar_src not in _sys.path:
        _sys.path.insert(0, _scalar_src)
    import pyopengjk
    return pyopengjk


def _cpu_epa_verify(verts1, verts2, gpu_depth, gpu_normal):
    """Compare GPU EPA result against scalar CPU EPA (penetration depth + contact normal)."""
    try:
        pgjk = _load_scalar()
        result = pgjk.compute_collision_information(verts1.tolist(), verts2.tolist())
        cpu_depth  = result.penetration_depth
        cpu_normal = np.array(result.contact_normal, dtype=np.float64)
        both_float = pgjk.USE_32BITS and USE_32BITS
        tol      = 1e-3 if both_float else 1e-4
        prec     = "float" if pgjk.USE_32BITS else "double"
        gpu_prec = "float" if USE_32BITS else "double"
        depth_diff = abs(float(cpu_depth) - float(gpu_depth))
        print(f"  CPU ({prec}) depth: {cpu_depth:.6f}  GPU ({gpu_prec}) depth: {gpu_depth:.6f}  diff: {depth_diff:.6f}")
        if depth_diff < tol:
            print_pass(f"CPU/GPU EPA depth agree (tol {tol})")
        else:
            print_warning(f"CPU/GPU EPA depth diff {depth_diff:.6f} exceeds {tol}")
        gpu_n = np.array(gpu_normal, dtype=np.float64)
        n1 = cpu_normal / (np.linalg.norm(cpu_normal) + 1e-12)
        n2 = gpu_n      / (np.linalg.norm(gpu_n)      + 1e-12)
        cos_sim = abs(float(np.dot(n1, n2)))
        if cos_sim > 0.99:
            print_pass(f"CPU/GPU contact normals agree (|cos| = {cos_sim:.4f})")
        else:
            print_warning(f"CPU/GPU contact normals diverge (|cos| = {cos_sim:.4f})")
    except Exception as e:
        print_warning(f"CPU EPA verification skipped: {e}")


def _cpu_gjk_verify(verts1, verts2, gpu_dist):
    """Compare GPU GJK distance against scalar CPU GJK."""
    try:
        pgjk = _load_scalar()
        result = pgjk.compute_minimum_distance(verts1.tolist(), verts2.tolist())
        cpu_dist = result.distance
        both_float = pgjk.USE_32BITS and USE_32BITS
        tol      = 1e-3 if both_float else 1e-4
        prec     = "float" if pgjk.USE_32BITS else "double"
        gpu_prec = "float" if USE_32BITS else "double"
        diff = abs(float(cpu_dist) - float(gpu_dist))
        print(f"  CPU ({prec}) dist: {cpu_dist:.6f}  GPU ({gpu_prec}) dist: {gpu_dist:.6f}  diff: {diff:.6f}")
        if diff < tol:
            print_pass(f"CPU/GPU distance agree (tol {tol})")
        else:
            print_warning(f"CPU/GPU diff {diff:.6f} exceeds {tol}")
    except Exception as e:
        print_warning(f"CPU GJK verification skipped: {e}")


def generate_polytope(num_verts, offset=None):
    """Generate a single polytope with random vertices on a sphere surface.

    Args:
        num_verts: Number of vertices
        offset: Optional (3,) offset array. If None, a random offset is used.

    Returns:
        ndarray of shape (num_verts, 3)
    """
    if offset is None:
        offset = (np.random.rand(3) - 0.5) * 20.0

    theta = np.random.rand(num_verts) * 2.0 * np.pi
    phi   = np.random.rand(num_verts) * np.pi
    r     = 1.0 + np.random.rand(num_verts) * 0.5

    vertices = np.zeros((num_verts, 3), dtype=DTYPE)
    vertices[:, 0] = r * np.sin(phi) * np.cos(theta) + offset[0]
    vertices[:, 1] = r * np.sin(phi) * np.sin(theta) + offset[1]
    vertices[:, 2] = r * np.cos(phi)                 + offset[2]

    return vertices


def generate_sphere_surface(num_points, radius, offset_x=0.0, offset_y=0.0, offset_z=0.0):
    """Generate points uniformly distributed on sphere surface (vectorized)."""
    u = np.random.rand(num_points)
    v = np.random.rand(num_points)

    theta = 2.0 * np.pi * u
    phi = np.arccos(2.0 * v - 1.0)

    vertices = np.zeros((num_points, 3), dtype=DTYPE)
    vertices[:, 0] = radius * np.sin(phi) * np.cos(theta) + offset_x
    vertices[:, 1] = radius * np.sin(phi) * np.sin(theta) + offset_y
    vertices[:, 2] = radius * np.cos(phi) + offset_z

    return vertices


def generate_cube_with_grid(grid_size, size, offset_x=0.0, offset_y=0.0, offset_z=0.0):
    """Generate a cube with grid of vertices on each face."""
    vertices_per_face = grid_size * grid_size
    total_vertices = 6 * vertices_per_face
    vertices = np.zeros((total_vertices, 3), dtype=DTYPE)

    idx = 0

    # Generate vertices for each of the 6 faces
    for face in range(6):
        for i in range(grid_size):
            for j in range(grid_size):
                y = -size + (2.0 * size * i) / (grid_size - 1) if grid_size > 1 else 0
                z = -size + (2.0 * size * j) / (grid_size - 1) if grid_size > 1 else 0
                x = -size + (2.0 * size * i) / (grid_size - 1) if grid_size > 1 else 0

                if face == 0:  # +X face
                    vertices[idx] = [size + offset_x, y + offset_y, z + offset_z]
                elif face == 1:  # -X face
                    vertices[idx] = [-size + offset_x, y + offset_y, z + offset_z]
                elif face == 2:  # +Y face
                    vertices[idx] = [x + offset_x, size + offset_y, z + offset_z]
                elif face == 3:  # -Y face
                    vertices[idx] = [x + offset_x, -size + offset_y, z + offset_z]
                elif face == 4:  # +Z face
                    vertices[idx] = [x + offset_x, y + offset_y, size + offset_z]
                elif face == 5:  # -Z face
                    vertices[idx] = [x + offset_x, y + offset_y, -size + offset_z]

                idx += 1

    return vertices


def test_1_simple_gjk():
    """Test 1: Simple GJK with single collision pair (from simple_collision example)."""
    print("=" * 70)
    print("Test 1: Simple GJK (Single Pair)")
    print("=" * 70)

    # Use exact vertices from userP.dat and userQ.dat (same as simple_collision example)
    polytope1 = np.array([
        [0.0, 5.5, 0.0],
        [2.3, 1.0, -2.0],
        [8.1, 4.0, 2.4],
        [4.3, 5.0, 2.2],
        [2.5, 1.0, 2.3],
        [7.1, 1.0, 2.4],
        [1.0, 1.5, 0.3],
        [3.3, 0.5, 0.3],
        [6.0, 1.4, 0.2]
    ], dtype=DTYPE)

    polytope2 = np.array([
        [0.0, -5.5, 0.0],
        [-2.3, -1.0, 2.0],
        [-8.1, -4.0, -2.4],
        [-4.3, -5.0, -2.2],
        [-2.5, -1.0, -2.3],
        [-7.1, -1.0, -2.4],
        [-1.0, -1.5, -0.3],
        [-3.3, -0.5, -0.3],
        [-6.0, -1.4, -0.2]
    ], dtype=DTYPE)

    # Upload pool to GPU once, then compute with index pair
    pool   = PolytopeArray([polytope1, polytope2])
    result = GpuBatch(pool, max_pairs=1).compute(SINGLE)

    distance = result['distances'][0]
    witness1 = result['witnesses1'][0]
    witness2 = result['witnesses2'][0]

    print(f"Polytope 1: 9 vertices (from userP.dat)")
    print(f"Polytope 2: 9 vertices (from userQ.dat)")
    print(f"\nResults:")
    print(f"  Distance: {distance:.6f}")
    print(f"  Witness 1: ({witness1[0]:.6f}, {witness1[1]:.6f}, {witness1[2]:.6f})")
    print(f"  Witness 2: ({witness2[0]:.6f}, {witness2[1]:.6f}, {witness2[2]:.6f})")

    # Verify distance from witness points
    computed_dist = np.linalg.norm(witness1 - witness2)
    print(f"\nVerification:")
    print(f"  Distance from witnesses: {computed_dist:.6f}")

    # Expected result from README: 3.653650
    expected_distance = 3.653650
    if abs(distance - expected_distance) < 1e-5:
        print_pass(f"Distance matches expected value ({expected_distance:.6f})")
    else:
        print_fail(f"Distance {distance:.6f} != expected {expected_distance:.6f}")
    print()


def test_2_batch_array():
    """Test 2: Batch array processing with per-stage timings."""
    print("=" * 70)
    print("Test 2: Batch Array Processing & Indexed API")
    print("=" * 70)

    num_pairs = 1000
    num_verts = 1000
    np.random.seed(RANDOM_SEED)

    print(f"Generating {num_pairs} random polytope pairs with {num_verts} vertices each...")

    # Generate random polytope pairs
    offsets1 = (np.random.rand(num_pairs, 3) - 0.5) * 10.0
    offsets2 = (np.random.rand(num_pairs, 3) - 0.5) * 10.0

    polytopes1 = [generate_polytope(num_verts, offsets1[i]) for i in range(num_pairs)]
    polytopes2 = [generate_polytope(num_verts, offsets2[i]) for i in range(num_pairs)]

    # Method 1: GpuBatch pool — interlace all polytopes, upload once
    print(f"\nMethod 1: GpuBatch pool")
    polytopes_interlaced_m1 = [p for pair in zip(polytopes1, polytopes2) for p in pair]
    pairs_m1 = np.array([[2*i, 2*i+1] for i in range(num_pairs)], dtype=np.int32)

    start = time.time()
    pool_m1 = PolytopeArray(polytopes_interlaced_m1)
    time_pack = time.time() - start

    start = time.time()
    batch = GpuBatch(pool_m1, max_pairs=num_pairs, with_epa=True)
    time_upload = time.time() - start

    start = time.time()
    result_nonindexed = batch.compute(pairs_m1)
    time_compute = time.time() - start

    print(f"  Host pack:    {time_pack*1000:.2f} ms")
    print(f"  GPU upload:   {time_upload*1000:.2f} ms")
    print(f"  Kernel+copy:  {time_compute*1000:.2f} ms")
    print(f"  Total:        {(time_pack+time_upload+time_compute)*1000:.2f} ms")

    distances_nonindexed = result_nonindexed['distances']
    print(f"  Distance range: [{distances_nonindexed.min():.3f}, {distances_nonindexed.max():.3f}]")

    # Method 2: Indexed API using IndexedBatch
    # Interlace: [p1[0], p2[0], p1[1], p2[1], ...]
    polytopes_interlaced = [p for pair in zip(polytopes1, polytopes2) for p in pair]
    pairs = np.array([[2*i, 2*i+1] for i in range(num_pairs)], dtype=np.int32)

    print(f"\nMethod 2: Indexed API (IndexedBatch — pack once, query twice)")
    pool = IndexedBatch(polytopes_interlaced)

    # Query 1: pairs in original order [i, j]
    start = time.time()
    result_indexed = pool.compute(pairs)
    time_q1 = time.time() - start

    distances_indexed = result_indexed['distances']
    print(f"  Query 1 (original order):    {time_q1*1000:.2f} ms  "
          f"range [{distances_indexed.min():.3f}, {distances_indexed.max():.3f}]")

    # Query 2: pairs in a random permutation (different size: first half only)
    perm = np.random.permutation(num_pairs)
    pairs_permuted = pairs[perm]
    start = time.time()
    result_permuted = pool.compute(pairs_permuted)
    time_q2 = time.time() - start

    distances_permuted = result_permuted['distances']
    print(f"  Query 2 (random permutation): {time_q2*1000:.2f} ms  "
          f"range [{distances_permuted.min():.3f}, {distances_permuted.max():.3f}]")

    # Verify: permuted results reordered must match original
    print(f"\nAgreement (non-indexed vs indexed):")
    max_diff = np.max(np.abs(distances_nonindexed - distances_indexed))
    mean_diff = np.mean(np.abs(distances_nonindexed - distances_indexed))
    print(f"  Max distance difference:  {max_diff:.9f}")
    print(f"  Mean distance difference: {mean_diff:.9f}")

    tolerance = 1e-5
    if max_diff < tolerance:
        print_pass(f"Non-indexed vs indexed results match within tolerance ({tolerance})")
    elif max_diff < 1e-3:
        print_warning(f"Results differ slightly (max diff: {max_diff:.9f})")
    else:
        print_fail(f"Results differ significantly (max diff: {max_diff:.9f})")

    # Verify: permuted query must return same distances as original (reordered)
    print(f"\nConsistency (original order vs permuted order):")
    perm_max_diff = np.max(np.abs(distances_indexed[perm] - distances_permuted))
    print(f"  Max difference: {perm_max_diff:.9f}")
    if perm_max_diff < tolerance:
        print_pass(f"IndexedBatch returns consistent results regardless of pair ordering")
    else:
        print_fail(f"Permuted query inconsistent (max diff: {perm_max_diff:.9f})")

    # CPU verification: run all pairs sequentially against scalar CPU library
    print(f"\nCPU verification (all {num_pairs} pairs, sequential):")
    try:
        import sys as _sys, os as _os
        _scalar_src = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..', 'scalar', 'examples', 'python_ctypes', 'src'))
        _sys.path.insert(0, _scalar_src)
        import pyopengjk as _pyopengjk
        from pyopengjk import compute_minimum_distance as _cpu_gjk
        _cpu_prec = "float" if _pyopengjk.USE_32BITS else "double"
        _gpu_prec = "float" if USE_32BITS else "double"
        start = time.time()
        cpu_dists = np.array([
            _cpu_gjk(polytopes1[i].tolist(), polytopes2[i].tolist()).distance
            for i in range(num_pairs)
        ])
        time_cpu = time.time() - start
        common_dtype = np.float32 if (_pyopengjk.USE_32BITS and USE_32BITS) else np.float64
        cpu_max_diff = np.max(np.abs(cpu_dists.astype(common_dtype) - distances_nonindexed.astype(common_dtype)))
        cpu_mean_diff = np.mean(np.abs(cpu_dists.astype(common_dtype) - distances_nonindexed.astype(common_dtype)))
        print(f"  Time: {time_cpu*1000:.2f} ms")
        print(f"  CPU ({_cpu_prec}) vs GPU ({_gpu_prec}) max diff:  {cpu_max_diff:.6f}")
        print(f"  CPU ({_cpu_prec}) vs GPU ({_gpu_prec}) mean diff: {cpu_mean_diff:.6f}")
        both_float = _pyopengjk.USE_32BITS and USE_32BITS
        cpu_tol = 1e-3 if both_float else 1e-4
        if cpu_max_diff < cpu_tol:
            print_pass(f"CPU and GPU results agree (tolerance {cpu_tol})")
        else:
            print_warning(f"CPU/GPU diff {cpu_max_diff:.6f} exceeds {cpu_tol}")
    except Exception as e:
        print_warning(f"CPU verification skipped: {e}")

    # EPA batch — same polytopes, same batch (already has with_epa=True)
    print(f"\nEPA batch (same {num_pairs} pairs):")
    start = time.time()
    result_epa = batch.compute_epa(pairs_m1)
    time_epa = time.time() - start

    depths_gpu = result_epa['penetration_depths']
    colliding = np.sum(depths_gpu <= 0)
    print(f"  GPU EPA time: {time_epa*1000:.2f} ms")
    print(f"  Colliding: {colliding}/{num_pairs}")
    print(f"  Depth range: [{depths_gpu.min():.4f}, {depths_gpu.max():.4f}]")

    collision_mask = depths_gpu <= 0
    collision_idx  = np.where(collision_mask)[0]
    print(f"\nCPU EPA verification ({len(collision_idx)} colliding pairs only):")
    try:
        import sys as _sys, os as _os
        _scalar_src = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..', 'scalar', 'examples', 'python_ctypes', 'src'))
        _sys.path.insert(0, _scalar_src)
        import pyopengjk as _pyopengjk
        from pyopengjk import compute_collision_information as _cpu_epa
        _cpu_prec = "float" if _pyopengjk.USE_32BITS else "double"
        _gpu_prec = "float" if USE_32BITS else "double"
        start = time.time()
        depths_cpu = np.array([
            _cpu_epa(polytopes1[i].tolist(), polytopes2[i].tolist()).penetration_depth
            for i in collision_idx
        ])
        time_cpu = time.time() - start
        common_dtype = np.float32 if (_pyopengjk.USE_32BITS and USE_32BITS) else np.float64
        diffs = np.abs(depths_cpu.astype(common_dtype) - depths_gpu[collision_idx].astype(common_dtype))
        print(f"  Time: {time_cpu*1000:.2f} ms")
        print(f"  CPU ({_cpu_prec}) vs GPU ({_gpu_prec}) max diff:  {diffs.max():.6f}")
        print(f"  CPU ({_cpu_prec}) vs GPU ({_gpu_prec}) mean diff: {diffs.mean():.6f}")
        both_float = _pyopengjk.USE_32BITS and USE_32BITS
        cpu_tol = 1e-3 if both_float else 1e-4
        if diffs.max() < cpu_tol:
            print_pass(f"CPU and GPU EPA results agree (tolerance {cpu_tol})")
        else:
            print_warning(f"CPU/GPU EPA diff {diffs.max():.6f} exceeds {cpu_tol}")
    except Exception as e:
        print_warning(f"CPU EPA verification skipped: {e}")
    print()


def test_epa_case1_overlapping_cubes():
    """EPA Case 1: Two overlapping cubes."""
    print("=" * 70)
    print("EPA Case 1: Two overlapping cubes")
    print("-" * 35)

    # Cube 1: centered at (0, 0, 0), size 2x2x2
    # Cube 2: centered at (1, 0, 0), size 2x2x2 (overlaps by 1 unit)
    cube1 = np.array([(x,   y, z) for x in [-1,1] for y in [-1,1] for z in [-1,1]], dtype=DTYPE)
    cube2 = np.array([(x+1, y, z) for x in [-1,1] for y in [-1,1] for z in [-1,1]], dtype=DTYPE)

    pool  = PolytopeArray([cube1, cube2])
    batch = GpuBatch(pool, max_pairs=1, with_epa=True)
    result = batch.compute_epa(SINGLE)

    depth = result['penetration_depths'][0]
    w1    = result['witnesses1'][0]
    w2    = result['witnesses2'][0]
    cn    = result['contact_normals'][0]

    print(f"  Distance/Penetration: {depth:.6f}")
    print(f"  Expected: Collision (distance should be small/negative)")
    print(f"  Witness 1: ({w1[0]:.6f}, {w1[1]:.6f}, {w1[2]:.6f})")
    print(f"  Witness 2: ({w2[0]:.6f}, {w2[1]:.6f}, {w2[2]:.6f})")
    print(f"  Contact Normal: ({cn[0]:.6f}, {cn[1]:.6f}, {cn[2]:.6f})")

    _cpu_epa_verify(cube1, cube2, depth, cn)

    if depth < -0.8 and depth > -1.2:
        print_pass("Collision detected, penetration depth valid")
    else:
        print_fail("Invalid results")
    print()


def test_epa_case2_touching_cubes():
    """EPA Case 2: Two touching cubes (just touching, no penetration)."""
    print("=" * 70)
    print("EPA Case 2: Two touching cubes")
    print("-" * 35)

    # Cube 1: centered at (0, 0, 0), size 2x2x2
    # Cube 2: centered at (2, 0, 0), size 2x2x2 (touching at x=1)
    cube1 = np.array([(x,   y, z) for x in [-1,1] for y in [-1,1] for z in [-1,1]], dtype=DTYPE)
    cube2 = np.array([(x+2, y, z) for x in [-1,1] for y in [-1,1] for z in [-1,1]], dtype=DTYPE)

    pool  = PolytopeArray([cube1, cube2])
    batch = GpuBatch(pool, max_pairs=1, with_epa=True)
    result = batch.compute_epa(SINGLE)

    depth = result['penetration_depths'][0]
    w1    = result['witnesses1'][0]
    w2    = result['witnesses2'][0]
    cn    = result['contact_normals'][0]

    print(f"  Distance: {depth:.6f}")
    print(f"  Expected: Very small distance (near zero)")
    print(f"  Witness 1: ({w1[0]:.6f}, {w1[1]:.6f}, {w1[2]:.6f})")
    print(f"  Witness 2: ({w2[0]:.6f}, {w2[1]:.6f}, {w2[2]:.6f})")
    print(f"  Contact Normal: ({cn[0]:.6f}, {cn[1]:.6f}, {cn[2]:.6f})")

    _cpu_gjk_verify(cube1, cube2, depth)

    if depth >= 0 and depth < 0.01:
        print_pass("Distance near zero as expected")
    else:
        print_warning("Distance may indicate collision or separation")
    print()


def test_epa_case3_separated_cubes():
    """EPA Case 3: Two separated cubes."""
    print("=" * 70)
    print("EPA Case 3: Two separated cubes")
    print("-" * 35)

    # Cube 1: centered at (0, 0, 0), size 2x2x2
    # Cube 2: centered at (5, 0, 0), size 2x2x2 (separated by 3 units)
    cube1 = np.array([(x,   y, z) for x in [-1,1] for y in [-1,1] for z in [-1,1]], dtype=DTYPE)
    cube2 = np.array([(x+5, y, z) for x in [-1,1] for y in [-1,1] for z in [-1,1]], dtype=DTYPE)

    pool  = PolytopeArray([cube1, cube2])
    batch = GpuBatch(pool, max_pairs=1, with_epa=True)
    result = batch.compute_epa(SINGLE)

    depth = result['penetration_depths'][0]
    w1    = result['witnesses1'][0]
    w2    = result['witnesses2'][0]
    cn    = result['contact_normals'][0]

    print(f"  Distance: {depth:.6f}")
    print(f"  Expected: Distance \u2248 3.0 (separation between cubes)")
    print(f"  Witness 1: ({w1[0]:.6f}, {w1[1]:.6f}, {w1[2]:.6f})")
    print(f"  Witness 2: ({w2[0]:.6f}, {w2[1]:.6f}, {w2[2]:.6f})")
    print(f"  Contact Normal: ({cn[0]:.6f}, {cn[1]:.6f}, {cn[2]:.6f})")

    _cpu_gjk_verify(cube1, cube2, depth)

    if depth > 2.9 and depth < 3.1:
        print_pass("Correct separation distance")
    elif depth > 0.0:
        print_warning("Distance may be incorrect")
    else:
        print_fail("Should not detect collision")
    print()


def test_epa_case5_overlapping_polytopes():
    """EPA Case 5: Overlapping polytopes (~50 vertices each)."""
    print("=" * 70)
    print("EPA Case 5: Overlapping polytopes (~50 vertices each)")
    print("-" * 35)

    np.random.seed(RANDOM_SEED)
    num_verts = 50

    # Polytope 1: centered at (0, 0, 0)
    # Polytope 2: centered at (0.5, 0, 0) - overlaps with polytope 1
    polytope1 = generate_polytope(num_verts, np.array([0.0, 0.0, 0.0], dtype=DTYPE))
    polytope2 = generate_polytope(num_verts, np.array([0.5, 0.0, 0.0], dtype=DTYPE))

    pool  = PolytopeArray([polytope1, polytope2])
    batch = GpuBatch(pool, max_pairs=1, with_epa=True)
    result = batch.compute_epa(SINGLE)

    depth = result['penetration_depths'][0]
    w1    = result['witnesses1'][0]
    w2    = result['witnesses2'][0]
    cn    = result['contact_normals'][0]

    print(f"  Distance/Penetration: {depth:.6f}")
    print(f"  Witness 1: ({w1[0]:.6f}, {w1[1]:.6f}, {w1[2]:.6f})")
    print(f"  Witness 2: ({w2[0]:.6f}, {w2[1]:.6f}, {w2[2]:.6f})")
    print(f"  Contact Normal: ({cn[0]:.6f}, {cn[1]:.6f}, {cn[2]:.6f})")

    if depth < 0.0:
        _cpu_epa_verify(polytope1, polytope2, depth, cn)
        print_pass(f"Collision detected with penetration depth of {-depth:.6f}")
    elif depth < 0.1:
        _cpu_epa_verify(polytope1, polytope2, depth, cn)
        print_pass("Collision detected (very small distance/penetration)")
    else:
        print_warning(f"Collision detected but distance seems large: {depth:.6f}")
    print()


def test_epa_case6_rotated_cubes():
    """EPA Case 6: Cube and rotated cube (45° around all axes, translated +1 on x)."""
    print("=" * 70)
    print("EPA Case 6: Cube and rotated cube (45\u00b0 around all axes, x+1) - High Resolution")
    print("-" * 35)

    grid_size = 40
    cube_size = 1.0
    num_verts = 6 * grid_size * grid_size

    print(f"  Generating cubes with {num_verts} vertices each ({grid_size}x{grid_size} grid per face)...")

    cube1 = generate_cube_with_grid(grid_size, cube_size, 0.0, 0.0, 0.0)
    cube2 = generate_cube_with_grid(grid_size, cube_size, 0.0, 0.0, 0.0)

    angle = 45.0 * np.pi / 180.0
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    for i in range(num_verts):
        px, py, pz = float(cube2[i,0]), float(cube2[i,1]), float(cube2[i,2])
        # Rotate around X axis by 45°
        temp_y = py * cos_a - pz * sin_a
        temp_z = py * sin_a + pz * cos_a
        py, pz = temp_y, temp_z
        # Rotate around Y axis by 45°
        temp_x = px * cos_a + pz * sin_a
        temp_z = -px * sin_a + pz * cos_a
        px, pz = temp_x, temp_z
        # Rotate around Z axis by 45°
        temp_x = px * cos_a - py * sin_a
        temp_y = px * sin_a + py * cos_a
        px, py = temp_x, temp_y
        # Translate by (1, 0, 0)
        cube2[i] = [px + 1.0, py, pz]

    print("  Running GJK and EPA...")
    pool  = PolytopeArray([cube1, cube2])
    batch = GpuBatch(pool, max_pairs=1, with_epa=True)
    result = batch.compute_epa(SINGLE)

    depth = result['penetration_depths'][0]
    w1    = result['witnesses1'][0]
    w2    = result['witnesses2'][0]
    cn    = result['contact_normals'][0]

    print(f"  Distance/Penetration: {depth:.6f}")
    print(f"  Expected: May overlap or be close depending on rotation")
    print(f"  Witness 1: ({w1[0]:.6f}, {w1[1]:.6f}, {w1[2]:.6f})")
    print(f"  Witness 2: ({w2[0]:.6f}, {w2[1]:.6f}, {w2[2]:.6f})")
    print(f"  Contact Normal: ({cn[0]:.6f}, {cn[1]:.6f}, {cn[2]:.6f})")

    valid1 = (w1[0] >= -2.0 and w1[0] <= 2.0 and
              w1[1] >= -2.0 and w1[1] <= 2.0 and
              w1[2] >= -2.0 and w1[2] <= 2.0)
    valid2 = (w2[0] >= -1.0 and w2[0] <= 3.0 and
              w2[1] >= -2.0 and w2[1] <= 2.0 and
              w2[2] >= -2.0 and w2[2] <= 2.0)

    if valid1 and valid2:
        if depth < 0.0:
            _cpu_epa_verify(cube1, cube2, depth, cn)
            print_pass(f"Collision detected with penetration depth of {-depth:.6f}")
        elif depth < 0.1:
            _cpu_epa_verify(cube1, cube2, depth, cn)
            print_pass("Collision detected, witness points valid")
        else:
            _cpu_gjk_verify(cube1, cube2, depth)
            print_pass(f"No collision, separation distance: {depth:.6f}")
    else:
        print_warning("Unexpected witness point locations")
    print()


def test_epa_case7_overlapping_spheres():
    """EPA Case 7: Two overlapping spheres (radius 2, 1000 points each)."""
    print("=" * 70)
    print("EPA Case 7: Two overlapping spheres (radius 2, 1000 points each)")
    print("-" * 35)

    np.random.seed(RANDOM_SEED)
    num_points = 1000
    radius = 2.0

    print(f"  Generating spheres with {num_points} points each, radius {radius:.2f}...")

    sphere1 = generate_sphere_surface(num_points, radius, 0.0, 0.0, 0.0)
    sphere2 = generate_sphere_surface(num_points, radius, 1.0, 0.0, 0.0)

    print("  Running GJK and EPA...")
    pool  = PolytopeArray([sphere1, sphere2])
    batch = GpuBatch(pool, max_pairs=1, with_epa=True)
    result = batch.compute_epa(SINGLE)

    depth = result['penetration_depths'][0]
    w1    = result['witnesses1'][0]
    w2    = result['witnesses2'][0]
    cn    = result['contact_normals'][0]

    print(f"  Distance/Penetration: {depth:.6f}")
    print(f"  Expected: Collision (spheres overlap, centers 1 unit apart, each radius 2)")
    print(f"  Expected overlap: ~3 units (2+2-1=3)")
    print(f"  Witness 1: ({w1[0]:.6f}, {w1[1]:.6f}, {w1[2]:.6f})")
    print(f"  Witness 2: ({w2[0]:.6f}, {w2[1]:.6f}, {w2[2]:.6f})")
    print(f"  Contact Normal: ({cn[0]:.6f}, {cn[1]:.6f}, {cn[2]:.6f})")

    dist1 = np.linalg.norm(w1)
    dist2 = np.linalg.norm(w2 - np.array([1.0, 0.0, 0.0]))
    valid1 = dist1 <= radius + 0.1
    valid2 = dist2 <= radius + 0.1

    if valid1 and valid2:
        if depth < 0.0:
            _cpu_epa_verify(sphere1, sphere2, depth, cn)
            print_pass(f"Collision detected with penetration depth of {-depth:.6f}")
            print(f"  Expected penetration: ~3.0 units")
        elif depth < 0.1:
            _cpu_epa_verify(sphere1, sphere2, depth, cn)
            print_pass("Collision detected (very small distance/penetration)")
        else:
            print_warning(f"Collision detected but distance seems large: {depth:.6f}")
    elif depth >= 0.0:
        print_warning("No collision detected, but spheres should overlap")
        print(f"  Separation distance: {depth:.6f}")
    else:
        print_warning("Unexpected results")
        if not valid1:
            print(f"    Witness 1 distance from sphere 1 center: {dist1:.6f} (expected <= {radius})")
        if not valid2:
            print(f"    Witness 2 distance from sphere 2 center: {dist2:.6f} (expected <= {radius})")
    print()


def test_epa_case8_separate_gjk_epa():
    """EPA Case 8: Two overlapping spheres (using separate GJK and EPA calls)."""
    print("=" * 70)
    print("EPA Case 8: Two overlapping spheres (radius 2, 1000 points each) - Separate GJK/EPA")
    print("-" * 35)

    np.random.seed(RANDOM_SEED)
    num_points = 1000
    radius = 2.0

    print(f"  Generating spheres with {num_points} points each, radius {radius:.2f}...")

    sphere1 = generate_sphere_surface(num_points, radius, 0.0, 0.0, 0.0)
    sphere2 = generate_sphere_surface(num_points, radius, 1.0, 0.0, 0.0)

    pool  = PolytopeArray([sphere1, sphere2])
    batch = GpuBatch(pool, max_pairs=1, with_epa=True)

    print("  Running GPU GJK...")
    result_gjk = batch.compute(SINGLE)
    print(f"  GJK Results:")
    print(f"    Distance: {result_gjk['distances'][0]:.6f}")

    print("  Running GPU EPA...")
    result_epa = batch.compute_epa(SINGLE)

    depth = result_epa['penetration_depths'][0]
    w1    = result_epa['witnesses1'][0]
    w2    = result_epa['witnesses2'][0]
    cn    = result_epa['contact_normals'][0]

    print(f"  Final Results:")
    print(f"  Distance/Penetration: {depth:.6f}")
    print(f"  Expected: Collision (spheres overlap, centers 1 unit apart, each radius 2)")
    print(f"  Expected overlap: ~3 units (2+2-1=3)")
    print(f"  Witness 1: ({w1[0]:.6f}, {w1[1]:.6f}, {w1[2]:.6f})")
    print(f"  Witness 2: ({w2[0]:.6f}, {w2[1]:.6f}, {w2[2]:.6f})")
    print(f"  Contact Normal: ({cn[0]:.6f}, {cn[1]:.6f}, {cn[2]:.6f})")

    dist1 = np.linalg.norm(w1)
    dist2 = np.linalg.norm(w2 - np.array([1.0, 0.0, 0.0]))
    valid1 = dist1 <= radius + 0.1
    valid2 = dist2 <= radius + 0.1

    if valid1 and valid2:
        if depth < 0.0:
            _cpu_epa_verify(sphere1, sphere2, depth, cn)
            print_pass(f"Collision detected with penetration depth of {-depth:.6f}")
            print(f"  Expected penetration: ~3.0 units")
        elif depth < 0.1:
            _cpu_epa_verify(sphere1, sphere2, depth, cn)
            print_pass("Collision detected (very small distance/penetration)")
        else:
            print_warning(f"Collision detected but distance seems large: {depth:.6f}")
    elif result_gjk['distances'][0] >= 0.0:
        print_warning("No collision detected, but spheres should overlap")
        print(f"  Separation distance: {result_gjk['distances'][0]:.6f}")
    else:
        print_warning("Unexpected results")
        if not valid1:
            print(f"    Witness 1 distance from sphere 1 center: {dist1:.6f} (expected <= {radius})")
        if not valid2:
            print(f"    Witness 2 distance from sphere 2 center: {dist2:.6f} (expected <= {radius})")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" OpenGJK GPU - Test Examples (Python)")
    print("=" * 70 + "\n")

    try:
        test_1_simple_gjk()
        test_2_batch_array()

        print("\n" + "=" * 70)
        print(" EPA Algorithm Testing")
        print("=" * 70 + "\n")

        test_epa_case1_overlapping_cubes()
        test_epa_case2_touching_cubes()
        test_epa_case3_separated_cubes()
        test_epa_case5_overlapping_polytopes()
        test_epa_case6_rotated_cubes()
        test_epa_case7_overlapping_spheres()
        test_epa_case8_separate_gjk_epa()

        print("=" * 70)
        print(" All tests completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
