"""Python wrapper for openGJK.

openGJK is a fast and robust C implementation of the
Gilbert-Johnson-Keerthi (GJK) algorithm.
"""

from .opengjk import (compute_minimum_distance, compute_collision_information,
                      Point3, Simplex, DistanceResult, CollisionResult, USE_32BITS)

__all__ = ["compute_minimum_distance", "compute_collision_information",
           "Point3", "Simplex", "DistanceResult", "CollisionResult", "USE_32BITS"]
