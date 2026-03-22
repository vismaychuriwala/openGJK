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

The Python wrapper searches these locations automatically. For 32-bit precision, build
with `-DUSE_32BITS=ON` and set `USE_32BITS = True` at the top of `opengjk.py`.

Then set up the Python environment:

    cd scalar/examples/python_ctypes
    python -m venv .env
    . .env/bin/activate        # or .env\Scripts\activate on Windows
    (.env) pip install --upgrade pip
    (.env) pip install -e .[test]
    (.env) pytest test/

## Usage

The API exposes the `compute_minimum_distance` function, taking as an
argument two lists of vertices. A `Point3` type is provided for
convenience, for example:

```python
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

However, any "list of lists" will suffice, *i.e.* a (N, 3) `numpy` array
will work just as well. The simplex that is returned contains
the final simplex vertices and their source indices in addition to
the witness points.
