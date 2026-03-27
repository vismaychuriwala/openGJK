/*
 *                          _____      _ _  __
 *                         / ____|    | | |/ /
 *   ___  _ __   ___ _ __ | |  __     | | ' /
 *  / _ \| '_ \ / _ \ '_ \| | |_ |_   | |  <
 * | (_) | |_) |  __/ | | | |__| | |__| | . \
 *  \___/| .__/ \___|_| |_|\_____|\____/|_|\_\
 *       | |
 *       |_|
 *
 * Copyright 2022-2026 Mattia Montanari, University of Oxford
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3. See https://www.gnu.org/licenses/
 */

/**
 * @file openGJK.h
 * @author Mattia Montanari
 * @date 1 Jan 2023
 * @brief Main interface of OpenGJK containing quick reference and API
 * documentation.
 *
 * @see https://www.mattiamontanari.com/opengjk/
 */

#ifndef OPENGJK_H__
#define OPENGJK_H__

#include <float.h>
#include <stdbool.h>

#ifdef __cplusplus
#define restrict
#endif

#ifdef INCLUDE_CMAKE_HEADER
#include "opengjk_export.h" /* CMake-generated export header for shared library symbols */
#else
#define OPENGJK_EXPORT /* Builds that don't use CMake (cython, zig, ...) don't
                           need a definition here */
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*! @brief Precision of floating-point numbers.
 *
 * Default is set to 64-bit (Double). Change this to quickly play around with
 * 16- and 32-bit. */
#ifdef USE_32BITS
#define gkFloat   float
#define gkEpsilon FLT_EPSILON
#define gkSqrt sqrtf
#define gkFmax fmaxf
#define gkFmin fminf
#define gkFabs fabsf
#else
#define gkFloat   double
#define gkEpsilon DBL_EPSILON
#define gkSqrt sqrt
#define gkFmax fmax
#define gkFmin fmin
#define gkFabs fabs
#endif

/*! @brief Data structure for convex polytopes.
 *
 * Polytopes are three-dimensional shapes and the GJK algorithm works directly
 * on their convex-hull. However the convex-hull is never computed explicitly,
 * instead each GJK-iteration employs a support function that has a cost
 * linearly dependent on the number of points defining the polytope. */
typedef struct gkPolytope_ {
  int numpoints;   /*!< Number of points defining the polytope. */
  gkFloat s[3];    /*!< Furthest point returned by the support function and updated
                   at each GJK-iteration. For the first iteration this value is
                   a guess - and this guess not irrelevant. */
  int s_idx;       /*!< Index of the furthest point returned by the support function.
              */
  gkFloat** coord; /*!< Coordinates of the points of the polytope. This is owned
                      by user who manages and garbage-collects the memory for
                      these coordinates. */
} gkPolytope;

/*! @brief Data structure for simplex.
 *
 * The simplex is updated at each GJK-iteration. For the first iteration this
 * value is a guess - and this guess not irrelevant. */
typedef struct gkSimplex_ {
  int nvrtx;               /*!< Number of points defining the simplex. */
  gkFloat vrtx[4][3];      /*!< Coordinates of the points of the simplex. */
  int vrtx_idx[4][2];      /*!< Indices of the points of the simplex. */
  gkFloat witnesses[2][3]; /*!< Witness points (closest points on each body).
                              After calling compute_minimum_distance():
                              - witnesses[0] contains the closest point on bd1
                              - witnesses[1] contains the closest point on bd2
                              These are computed using barycentric coordinates
                              from the final simplex vertices. */
} gkSimplex;


#define MAX_EPA_FACES 128

#define MAX_EPA_VERTICES (MAX_EPA_FACES + 4)

// Face structure for EPA polytope
// Each face is a triangle with 3 vertex indices
typedef struct {
  int v[3];           // Vertex indices in the polytope
  int v_idx[3][2];    // Original vertex indices from original polytopes for witness computation [vertex][body]
  gkFloat normal[3];  // Face normal (pointing outward from origin)
  gkFloat distance;   // Distance from origin to face plane
  bool valid;         // Whether this face is still valid (not removed)
} EPAFace;

// Polytope structure for EPA
typedef struct {
  gkFloat vertices[MAX_EPA_FACES + 4][3];  // Vertex coordinates in the Minkowski difference
  int vertex_indices[MAX_EPA_FACES + 4][2]; // Original vertex indices [vertex][body]
  int num_vertices;
  EPAFace faces[MAX_EPA_FACES];
  int max_face_index; // Highest face index in use (for iteration bounds)
} EPAPolytope;


// Structure for horizon edge collection
typedef struct {
  int v1, v2;  // Vertex indices in polytope
  int v_idx1[2], v_idx2[2];  // Original vertex indices for witness computation
  bool valid;
} EPAEdge;

/*! @brief Invoke the GJK algorithm to compute the minimum distance between two
 * polytopes.
 *
 * @param[in]     bd1  First polytope (passed by value, modified internally).
 * @param[in]     bd2  Second polytope (passed by value, modified internally).
 * @param[in,out] s    Simplex structure. Must be initialized (set nvrtx = 0)
 *                     before first call. After return, contains the final
 *                     simplex and witness points in s->witnesses.
 * @return The minimum Euclidean distance between the two polytopes.
 *         Returns 0 if the polytopes are intersecting or touching.
 *
 * @note The simplex has to be initialised prior the call to this function.
 *       Witness points are automatically computed and stored in s->witnesses.
 */
OPENGJK_EXPORT gkFloat compute_minimum_distance(gkPolytope bd1, gkPolytope bd2, gkSimplex* s);

/*! @brief Invoke the EPA algorithm to compute the collision information between two colliding
 * polytopes.
 *
 * @param[in]     bd1                 First polytope (passed by value, modified internally).
 * @param[in]     bd2                 Second polytope (passed by value, modified internally).
 * @param[in,out] s                   Simplex structure. Must be initialized. 
 *                                    After return, contains the final
 *                                    simplex and witness points in s->witnesses.
 * @param[in, out] distance           Takes in the distance computed by the GJK algorithm 
 *                                    (0 -> collision detected, run EPA), otherwise return early.
 *                                    After return, stores the penetration distance computed between 
 *                                    the two polytopes. 
 * @param[out] contact_normal[3]      Contact normal computed by the EPA algorithm
 
 * Witness points are automatically computed and stored in s->witnesses. */
OPENGJK_EXPORT void computeCollisionInformation(
  const gkPolytope* bd1,
  const gkPolytope* bd2,
  gkSimplex* simplex,
  gkFloat* distance,
  gkFloat contact_normal[3]);

/*! @brief Testing wrappers - expose internal functions for cross-validation.
 *
 * These functions allow testing the internal simplex sub-algorithms (S1D, S2D, S3D)
 * against SIMD implementations. Vertex ordering convention:
 *   - S1D: vrtx[1] = newest (p), vrtx[0] = oldest (q)
 *   - S2D: vrtx[2] = newest (p), vrtx[1] = q, vrtx[0] = oldest (r)
 *   - S3D: vrtx[3] = newest (p), vrtx[2] = q, vrtx[1] = r, vrtx[0] = oldest (t)
 */
OPENGJK_EXPORT void opengjk_test_S1D(gkSimplex* s, gkFloat* v);
OPENGJK_EXPORT void opengjk_test_S2D(gkSimplex* s, gkFloat* v);
OPENGJK_EXPORT void opengjk_test_S3D(gkSimplex* s, gkFloat* v);

#ifdef __cplusplus
}
#endif

#endif /* OPENGJK_H__ */
