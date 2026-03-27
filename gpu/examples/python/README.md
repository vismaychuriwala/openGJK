# OpenGJK GPU - Python Wrapper

GPU-accelerated GJK (Gilbert-Johnson-Keerthi) and EPA (Expanding Polytope Algorithm) for batch collision detection in Python.

## Features

- **Batch Processing**: Process thousands of collision pairs in a single GPU call
- **Warp-Level Parallelism**: 8 threads per GJK, 32 threads per EPA
- **GJK**: Minimum distance and witness points for separated polytopes
- **EPA**: Penetration depth, contact points, and contact normals for colliding polytopes
- **Combined GJK+EPA**: Single call for mixed separated/colliding batches
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
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_GPU=ON -DBUILD_SCALAR=OFF -DBUILD_SIMD=OFF -DBUILD_TESTS=OFF
cmake --build build --config Release
```

The library will be at:
- Windows: `build/gpu/Release/openGJK_GPU.dll`
- Linux:   `build/gpu/libopenGJK_GPU.so`
- macOS:   `build/gpu/libopenGJK_GPU.dylib`

The wrapper searches these locations automatically. The Python wrapper defaults to double precision (`USE_32BITS=False`), so build with `-DUSE_32BITS=OFF` to match (the global CMake default is float32).
To use float32 instead, build without `-DUSE_32BITS=OFF` and set `USE_32BITS = True` at the top of `opengjk_gpu.py`.

> **Using the scalar CPU wrapper too?** Both libraries must be built with the same `USE_32BITS` value, and the flag must match in both `opengjk_gpu.py` and `opengjk.py`.

### 2. Set Up Python

```bash
cd gpu/examples/python
pip install -e .
python example.py        # single-pair example
python test_examples.py  # comprehensive test suite
```

## Usage

### One-shot functions

The simplest way to use the library — pack your data and get results back in one call.

#### `compute_minimum_distance` — GJK for separated polytopes

```python
import numpy as np
from pyopengjk_gpu import compute_minimum_distance

# (N, V, 3) array — N pairs, each polytope has V vertices
polytopes1 = np.random.rand(100, 8, 3)
polytopes2 = np.random.rand(100, 8, 3)

result = compute_minimum_distance(polytopes1, polytopes2)
print(result['distances'])   # (100,) — 0.0 means collision
print(result['witnesses1'])  # (100, 3) — closest point on each polytope
print(result['witnesses2'])  # (100, 3)
print(result['is_collision']) # (100,) bool
```

#### `compute_epa` — EPA for colliding polytopes

```python
from pyopengjk_gpu import compute_epa

result = compute_epa(polytopes1, polytopes2)
print(result['penetration_depths'])  # (N,) — how far they overlap
print(result['contact_normals'])     # (N, 3) — unit normal from body2 to body1
print(result['witnesses1'])          # (N, 3) — contact point on body1
print(result['witnesses2'])          # (N, 3) — contact point on body2
```

#### `compute_gjk_epa` — combined for mixed batches

Runs GJK+EPA together; returns GJK witnesses for separated pairs and EPA contact
points for colliding pairs.

```python
from pyopengjk_gpu import compute_gjk_epa

result = compute_gjk_epa(polytopes1, polytopes2)
# result['distances']:      (N,) — positive = separation, 0 = collision
# result['is_collision']:   (N,) bool
# result['witnesses1/2']:   (N, 3) — GJK witnesses or EPA contact points
# result['contact_normals']: (N, 3)
```

---

### `GpuBatch` — recommended for repeated queries against a fixed pool

Use `GpuBatch` when you have a fixed set of polytopes and want to query different
pairs repeatedly (e.g. every physics frame). Polytope data is uploaded to the GPU
once at construction and stays resident — only the pair indices are re-uploaded
each call.

```python
from pyopengjk_gpu import GpuBatch
import numpy as np

pool = np.random.rand(50, 8, 3)
max_pairs = 200

# Upload pool to GPU once
batch = GpuBatch(pool, max_pairs)

# Query with different pairs each call — polytope data stays on GPU
pairs_a = np.array([[0, 1], [2, 5], [3, 7]], dtype=np.int32)
result = batch.compute(pairs_a)
# result['distances']:    (3,)
# result['witnesses1/2']: (3, 3)
# result['is_collision']: (3,) bool

pairs_b = np.array([[1, 4], [0, 3]], dtype=np.int32)
result = batch.compute(pairs_b)  # polytope data not re-uploaded
```

For EPA, pass `with_epa=True` at construction:

```python
batch = GpuBatch(pool, max_pairs, with_epa=True)

epa_result = batch.compute_epa(pairs_a)
# epa_result['penetration_depths']: (n,)
# epa_result['witnesses1/2']:       (n, 3)
# epa_result['contact_normals']:    (n, 3)
```

---

## API Reference

### `PolytopeArray(vertices_list)`

Packs a list of polytopes into a contiguous host buffer.

**Parameters:**
- `vertices_list`: `(N, V, 3)` ndarray (uniform vertex count), list of `(V_i, 3)` arrays (variable counts), or a single `(V, 3)` ndarray

---

### `GpuBatch(polytopes, max_pairs, with_epa=False)`

Uploads a polytope pool to the GPU once and owns the device memory for its lifetime.

**Parameters:**
- `polytopes`: `PolytopeArray`, `(M, V, 3)` ndarray, or list of `(V_i, 3)` arrays — the pool
- `max_pairs`: maximum number of pairs per `compute` call (pre-allocates result buffers)
- `with_epa`: pre-allocate contact-normals buffer (required before calling `compute_epa`)

**Methods:**

`compute(pairs)` — GJK for the given pairs. Returns:
- `'distances'`: `(n,)` — 0.0 for collisions
- `'witnesses1'`, `'witnesses2'`: `(n, 3)` — closest points
- `'is_collision'`: `(n,)` bool

`compute_epa(pairs)` — GJK then EPA for the given pairs. Returns:
- `'penetration_depths'`: `(n,)` — positive penetration distance
- `'witnesses1'`, `'witnesses2'`: `(n, 3)` — contact points
- `'contact_normals'`: `(n, 3)` — unit normal from body2 to body1

---

### `compute_minimum_distance(vertices1, vertices2)`

One-shot GJK for two matched batches of polytopes.

**Returns:** `distances`, `witnesses1`, `witnesses2`, `is_collision`

---

### `compute_epa(vertices1, vertices2)`

One-shot EPA for two matched batches of polytopes.

**Returns:** `penetration_depths`, `witnesses1`, `witnesses2`, `contact_normals`

---

### `compute_gjk_epa(vertices1, vertices2)`

One-shot combined GJK+EPA.

**Returns:** `distances`, `is_collision`, `witnesses1`, `witnesses2`, `contact_normals`

---

## Examples

- [example.py](example.py) — single polytope pair
- [test_examples.py](test_examples.py) — GJK, EPA, batch, indexed API, touching/overlapping/separated geometry, CPU cross-validation

## License

GPL-3.0-only (same as OpenGJK)

Copyright 2022-2026 Mattia Montanari, University of Oxford
Copyright 2025-2026 Vismay Churiwala, Marcus Hedlund

## See Also

- [OpenGJK-GPU](https://github.com/vismaychuriwala/OpenGJK-GPU) - GPU implementation
- [OpenGJK](https://www.mattiamontanari.com/opengjk/) - Main project
- [GPU API Documentation](../../README.md) - C/CUDA API reference
