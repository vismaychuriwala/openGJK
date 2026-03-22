# OpenGJK GPU - Python Wrapper

GPU-accelerated GJK (Gilbert-Johnson-Keerthi) and EPA (Expanding Polytope Algorithm) for batch collision detection in Python.

## Features

- **Batch Processing**: Process thousands of collision pairs in a single GPU call
- **Warp-Level Parallelism**: 8 threads per GJK for efficient GPU utilization
- **Two API levels**: Convenience functions for one-shot calls; `GpuBatch` for amortized upload cost
- **GJK**: Minimum distance and witness points for separated polytopes
- **EPA**: Penetration depth, contact points, and contact normals for colliding polytopes
- **Combined GJK+EPA**: Single call that handles both separated and colliding pairs
- **Indexed API**: Test many pairs from a shared polytope pool without duplicating data

## Requirements

- **NVIDIA GPU** with CUDA support (compute capability 6.0+)
- **CUDA Toolkit** 11.0+
- **Python** 3.7+
- **NumPy**
- **CMake** 3.18+

## Installation

### 1. Build the Shared Library

From the repository root:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_GPU=ON -DBUILD_SCALAR=OFF -DBUILD_SIMD=OFF -DUSE_32BITS=ON
cmake --build build --config Release
```

The library will be at:
- Windows: `build/gpu/Release/openGJK_GPU.dll`
- Linux:   `build/gpu/libopenGJK_GPU.so`
- macOS:   `build/gpu/libopenGJK_GPU.dylib`

The wrapper searches these locations automatically. For 64-bit precision, build with
`-DUSE_32BITS=OFF` and set `USE_32BITS = False` at the top of `opengjk_gpu.py`.

### 2. Set Up Python

```bash
cd gpu/examples/python
pip install numpy
python example.py        # simple single-pair example
python test_examples.py  # comprehensive test suite
```

## Usage

### Convenience functions (one-shot)

```python
import numpy as np
from pyopengjk_gpu import compute_minimum_distance

# (N, D, 3) array — N pairs with D vertices each
polytopes1 = np.random.rand(100, 8, 3).astype(np.float32)
polytopes2 = np.random.rand(100, 8, 3).astype(np.float32)

result = compute_minimum_distance(polytopes1, polytopes2)
print(result['distances'])   # shape (100,)
print(result['witnesses1'])  # shape (100, 3)
```

For variable vertex counts, pass a list of `(V_i, 3)` arrays instead of a 3D ndarray.

### `GpuBatch` (amortized upload)

When running GJK and/or EPA multiple times on the same polytopes, use `GpuBatch` to
upload data to the GPU once and compute repeatedly:

```python
from pyopengjk_gpu import GpuBatch, PolytopeArray

bd1 = PolytopeArray(polytopes1)
bd2 = PolytopeArray(polytopes2)

# GJK only
batch = GpuBatch(bd1, bd2)
result = batch.compute_gjk()

# GJK + EPA (must set with_epa=True at construction)
batch = GpuBatch(bd1, bd2, with_epa=True)
gjk_result = batch.compute_gjk()
epa_result = batch.compute_epa()
combined   = batch.compute_gjk_epa()
```

### Indexed API

Test pairs from a shared polytope pool. Use `IndexedBatch` to pack the pool once and query with different pairs each frame:

```python
from pyopengjk_gpu import IndexedBatch

pool = IndexedBatch(np.random.rand(20, 8, 3).astype(np.float32))
result = pool.compute(np.array([[0, 1], [2, 5], [3, 7]], dtype=np.int32))
result = pool.compute(np.array([[0, 5], [1, 3]], dtype=np.int32))  # different size
```

## API Reference

### `PolytopeArray(vertices_list)`

Packs a list of polytopes into a contiguous host buffer and builds the `gkPolytope`
struct array required by the GPU kernels.

**Parameters:**
- `vertices_list`: `(N, D, 3)` ndarray (uniform vertex count), list of `(V_i, 3)` arrays (variable counts), or a single `(D, 3)` ndarray

---

### `GpuBatch(bd1, bd2, with_epa=False)`

Uploads polytope data to the GPU once and owns the device memory for its lifetime.
Re-use the same batch to avoid repeated upload overhead.

**Parameters:**
- `bd1`, `bd2`: `PolytopeArray` — must have equal length
- `with_epa`: bool — pre-allocate GPU witness/normal buffers for EPA (required before calling `compute_epa` or `compute_gjk_epa`)

**Methods:**

`compute_gjk()` — Returns:
- `'distances'`: `(N,)` — minimum distances; 0.0 for collisions
- `'witnesses1'`, `'witnesses2'`: `(N, 3)` — closest points
- `'is_collision'`: `(N,)` bool
- `'simplex_nvrtx'`: `(N,)` — vertices in final simplex

`compute_epa()` — Runs GJK then EPA. Returns:
- `'penetration_depths'`: `(N,)` — penetration distances (positive)
- `'witnesses1'`, `'witnesses2'`: `(N, 3)` — contact points
- `'contact_normals'`: `(N, 3)` — contact normals

`compute_gjk_epa()` — Runs GJK+EPA together; uses GJK witnesses for separated pairs
and EPA contact points for colliding pairs. Returns all keys from both above, plus
`'simplex_nvrtx'`.

---

### `compute_minimum_distance(vertices1, vertices2)`

One-shot GJK. Equivalent to `GpuBatch(...).compute_gjk()`.

**Parameters:** same flexible input as `PolytopeArray` — `(N, V, 3)` ndarray, list of `(V_i, 3)` arrays, etc.

**Returns:** same as `GpuBatch.compute_gjk()`

---

### `compute_epa(vertices1, vertices2)`

One-shot EPA. Equivalent to `GpuBatch(..., with_epa=True).compute_epa()`.

**Returns:** same as `GpuBatch.compute_epa()`

---

### `compute_gjk_epa(vertices1, vertices2)`

One-shot combined GJK+EPA. Equivalent to `GpuBatch(..., with_epa=True).compute_gjk_epa()`.

**Returns:** same as `GpuBatch.compute_gjk_epa()`

---

### `IndexedBatch(polytopes)`

Packs a polytope pool into a host buffer once, then accepts different pair index arrays on each call. Use this when the set of polytopes is fixed but the pairs to test change between queries.

**Parameters:**
- `polytopes`: `PolytopeArray`, `(M, D, 3)` ndarray, or list of `(V_i, 3)` arrays — the polytope pool

**Method:** `compute(pairs)` — `pairs` is an `(n_pairs, 2)` int32 array of indices; size may vary between calls. Returns same keys as `GpuBatch.compute_gjk()`.

```python
from pyopengjk_gpu import IndexedBatch

pool = IndexedBatch(polytopes)           # pack once

result_a = pool.compute(pairs_frame_0)  # (n0, 2) pairs
result_b = pool.compute(pairs_frame_1)  # (n1, 2) pairs — different size ok
```

---

### `compute_minimum_distance_indexed(polytopes, pairs)`

One-shot indexed GJK. Equivalent to `IndexedBatch(polytopes).compute(pairs)`.

**Parameters:**
- `polytopes`: `PolytopeArray`, `(M, D, 3)` ndarray, or list of `(V_i, 3)` arrays — the polytope pool
- `pairs`: `(n_pairs, 2)` int32 array — indices into `polytopes`

**Returns:** same as `GpuBatch.compute_gjk()`

---

## Examples

- [example.py](example.py) — single polytope pair, `GpuBatch` usage
- [test_examples.py](test_examples.py) — nine tests covering GJK, EPA, batch processing, indexed API, touching/overlapping/separated geometry, and CPU cross-validation

## License

GPL-3.0-only (same as OpenGJK)

Copyright 2022-2026 Mattia Montanari, University of Oxford
Copyright 2025-2026 Vismay Churiwala, Marcus Hedlund

## See Also

- [OpenGJK-GPU](https://github.com/vismaychuriwala/OpenGJK-GPU) - GPU implementation
- [OpenGJK](https://www.mattiamontanari.com/opengjk/) - Main project
- [GPU API Documentation](../../README.md) - C/CUDA API reference
