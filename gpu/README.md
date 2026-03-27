# OpenGJK GPU Implementation

CUDA-accelerated GJK and EPA for batch collision detection on NVIDIA GPUs.

- **GJK**: minimum distance and witness points for separated polytopes
- **EPA**: penetration depth, contact points, and contact normals for colliding polytopes
- **Warp-level parallelism**: 8 threads per GJK collision, 32 threads per EPA collision
- **Indexed API**: query pairs from a shared polytope pool without duplicating data

## Build

From the repository root:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_GPU=ON -DBUILD_SCALAR=OFF -DBUILD_SIMD=OFF -DBUILD_TESTS=OFF
cmake --build build --config Release
```

Default precision is float32 (`USE_32BITS=ON`). For double use `-DUSE_32BITS=OFF`.

For the latest updates, benchmarks, and additional details see the [OpenGJK-GPU repository](https://github.com/vismaychuriwala/OpenGJK-GPU).

## Data Structures

```c
// Coordinates are flattened: [x0,y0,z0, x1,y1,z1, ...]
typedef struct { int numpoints; gkFloat s[3]; int s_idx; gkFloat* coord; } gkPolytope;

// Witness points written to witnesses[0] and witnesses[1]
typedef struct { int nvrtx; gkFloat vrtx[4][3]; int vrtx_idx[4][2]; gkFloat witnesses[2][3]; } gkSimplex;

// Index pair into a polytope pool
typedef struct { int idx1; int idx2; } gkCollisionPair;
```

## API

### Indexed — recommended for repeated queries

Upload a polytope pool once; send only pair indices each call. Pool stays on GPU between calls.

```c
// Allocate once — pass non-NULL d_contact_normals to enable EPA
void allocate_indexed_device(
    int num_polytopes, int max_pairs, const gkPolytope* polytopes,
    gkPolytope** d_polytopes, gkFloat** d_coords, gkCollisionPair** d_pairs,
    gkSimplex** d_simplices, gkFloat** d_distances, gkFloat** d_contact_normals);

// Upload new pair indices each call
void upload_pairs_device(int num_pairs, const gkCollisionPair* pairs, gkCollisionPair* d_pairs);

// Run GJK / EPA (pool stays on GPU)
void compute_minimum_distance_indexed_device(
    int num_pairs, const gkPolytope* d_polytopes, const gkCollisionPair* d_pairs,
    gkSimplex* d_simplices, gkFloat* d_distances);

void compute_epa_indexed_device(
    int num_pairs, const gkPolytope* d_polytopes, const gkCollisionPair* d_pairs,
    gkSimplex* d_simplices, gkFloat* d_distances, gkFloat* d_contact_normals);

// Copy results to host, then free when done
void copy_results_from_device(int n, const gkSimplex* d_simplices, const gkFloat* d_distances,
    gkSimplex* simplices, gkFloat* distances);

void free_indexed_device(gkPolytope* d_polytopes, gkFloat* d_coords, gkCollisionPair* d_pairs,
    gkSimplex* d_simplices, gkFloat* d_distances, gkFloat* d_contact_normals);
```

---

### High-level — one-shot, automatic memory management

```c
// GJK: distances[i] == 0.0 indicates collision
void compute_minimum_distance(int n, const gkPolytope* bd1, const gkPolytope* bd2,
    gkSimplex* simplices, gkFloat* distances);

// EPA: witnesses written to simplices[i].witnesses[0/1]
void computeCollisionInformation(int n, const gkPolytope* bd1, const gkPolytope* bd2,
    gkSimplex* simplices, gkFloat* distances, gkFloat* contact_normals);

// Combined GJK + EPA
void compute_gjk_epa(int n, const gkPolytope* bd1, const gkPolytope* bd2,
    gkSimplex* simplices, gkFloat* distances, gkFloat* contact_normals);

// Indexed one-shot variants
void compute_minimum_distance_indexed(int num_polytopes, int num_pairs,
    const gkPolytope* polytopes, const gkCollisionPair* pairs,
    gkSimplex* simplices, gkFloat* distances);

void compute_gjk_epa_indexed(int num_polytopes, int num_pairs,
    const gkPolytope* polytopes, const gkCollisionPair* pairs,
    gkSimplex* simplices, gkFloat* distances, gkFloat* contact_normals);
```

---

### Low-level — direct kernel invocation

```c
__global__ void compute_minimum_distance_kernel(
    const gkPolytope* polytopes1, const gkPolytope* polytopes2,
    gkSimplex* simplices, gkFloat* distances, int n);

__global__ void compute_epa_kernel(
    const gkPolytope* polytopes1, const gkPolytope* polytopes2,
    gkSimplex* simplices, gkFloat* distances, gkFloat* contact_normals, int n);

__global__ void compute_minimum_distance_indexed_kernel(
    const gkPolytope* polytopes, const gkCollisionPair* pairs,
    gkSimplex* simplices, gkFloat* distances, int n);

__global__ void compute_epa_kernel_indexed_kernel(
    const gkPolytope* polytopes, const gkCollisionPair* pairs,
    gkSimplex* simplices, gkFloat* distances, gkFloat* contact_normals, int n);
```

## Files

- `openGJK_GPU.cu` — complete CUDA implementation
- `include/openGJK_GPU.h` — public API header
- `examples/` — C++ and Python usage examples

## Python Wrapper

See [examples/python/README.md](examples/python/README.md).

## Credits

Based on [OpenGJK-GPU](https://github.com/vismaychuriwala/OpenGJK-GPU) by Vismay Churiwala and Marcus Hedlund.
Original GJK implementation by Mattia Montanari, University of Oxford.
