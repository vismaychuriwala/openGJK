# Python `ctypes` wrapper

This wrapper uses `ctypes` to wrap the full C API for use from Python.

## Getting Started

Build the scalar shared library from the project root:

    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SCALAR=ON -DBUILD_SIMD=OFF -DUSE_32BITS=OFF
    cmake --build . --config Release

The shared library will be placed at:

- Windows: `build/scalar/Release/opengjk_scalar.dll`
- Linux:   `build/scalar/libopengjk_scalar.so`
- macOS:   `build/scalar/libopengjk_scalar.dylib`

The Python wrapper searches these locations automatically. The Python wrapper defaults to double precision (`USE_32BITS=False`), so pass `-DUSE_32BITS=OFF` at configure time to match (the global CMake default is float32).
To use float32 instead, omit `-DUSE_32BITS=OFF` and set `USE_32BITS = True` at the top of `opengjk.py`.

Then set up the Python environment:

    cd scalar/examples/python_ctypes
    python -m venv .env
    . .env/bin/activate        # or .env\Scripts\activate on Windows
    (.env) pip install --upgrade pip
    (.env) pip install -e .[test]
    (.env) pytest test/

## Usage

The API exposes two functions. Both accept any "list of lists" as vertices —
a `Point3` named tuple is provided for convenience, but a `(N, 3)` numpy
array works just as well.

### `compute_minimum_distance` — separated polytopes

Runs GJK and returns the minimum distance and closest witness points for
polytopes that are **not** colliding:

```python
from pyopengjk import compute_minimum_distance, Point3

vertices0 = [
    Point3(0.0, 5.5, 0.0),
    Point3(2.3, 1.0, -2.0),
    Point3(8.1, 4.0, 2.4),
    Point3(4.3, 5.0, 2.2),
    Point3(2.5, 1.0, 2.3),
    Point3(7.1, 1.0, 2.4),
    Point3(1.0, 1.5, 0.3),
    Point3(3.3, 0.5, 0.3),
    Point3(6.0, 1.4, 0.2)
]

vertices1 = [
    Point3(0.0, -5.5, 0.0),
    Point3(-2.3, -1.0, 2.0),
    Point3(-8.1, -4.0, -2.4),
    Point3(-4.3, -5.0, -2.2),
    Point3(-2.5, -1.0, -2.3),
    Point3(-7.1, -1.0, -2.4),
    Point3(-1.0, -1.5, -0.3),
    Point3(-3.3, -0.5, -0.3),
    Point3(-6.0, -1.4, -0.2)
]

result = compute_minimum_distance(vertices0, vertices1)
print(f"Minimum distance: {result.distance}")
print("Witness points:")
print(result.simplex.witnesses[0])
print(result.simplex.witnesses[1])
```

will produce the output:

    Minimum distance: 3.653649722294501
    Witness points:
    Point3(x=1.0251728907330566, y=1.4903181189488242, z=0.2554633471645919)
    Point3(x=-1.0251728907330566, y=-1.4903181189488242, z=-0.2554633471645919)

Returns a `DistanceResult(distance, simplex)` where `simplex` contains the
final simplex vertices, their source indices, and the witness points.

### `compute_collision_information` — colliding polytopes

Runs GJK then EPA to compute penetration depth, contact points, and contact
normal for polytopes that **are** colliding:

```python
from pyopengjk import compute_collision_information, Point3

# Two overlapping cubes
cube = [
    Point3(-1, -1, -1), Point3( 1, -1, -1),
    Point3(-1,  1, -1), Point3( 1,  1, -1),
    Point3(-1, -1,  1), Point3( 1, -1,  1),
    Point3(-1,  1,  1), Point3( 1,  1,  1),
]
shifted = [Point3(x + 0.5, y, z) for x, y, z in cube]

result = compute_collision_information(cube, shifted)
print(f"Penetration depth: {result.penetration_depth}")
print(f"Contact normal:    {result.contact_normal}")
print(f"Witness on body 1: {result.simplex.witnesses[0]}")
print(f"Witness on body 2: {result.simplex.witnesses[1]}")
```

Returns a `CollisionResult(penetration_depth, simplex, contact_normal)`:
- `penetration_depth`: positive scalar — how far the shapes overlap
- `contact_normal`: `(float, float, float)` — unit normal pointing from body 2 to body 1
- `simplex.witnesses`: the two contact points, one on each body's surface
