# mojo-qhull

`mojo-qhull` is a standalone Mojo port of the headline geometry algorithms in
[Qhull](http://www.qhull.org/), with a NumPy-friendly Python API. It computes
simplicial convex hulls in 2D, 3D, and higher dimensions, Delaunay
triangulations, Voronoi diagrams, and intersections of halfspaces.

This is a source port, not a wrapper around the installed Qhull library. The
hot hull construction, facet planes, paraboloid projection, polar transform,
and Voronoi-center solves execute in the compiled Mojo shared library. Tests
use SciPy's bindings to the real upstream Qhull as the parity reference.

## Upstream and license

The port was derived from Qhull's reentrant C implementation at upstream
commit `d1c2fc0caa5f644f3a0f220290d4a868c68ed4f6`. In particular, it follows
`libqhull_r.c` (`qh_buildhull`, `qh_partitionall`), `poly2_r.c`
(`qh_initialhull`), `geom_r.c` (`qh_distplane`, `qh_setfacetplane`), and
`geom2_r.c` (`qh_distround`, `qh_maxsimplex`, `qh_setdelaunay`,
`qh_sethalfspace`, and `qh_voronoi_center`).

The new repository code is MIT licensed. Qhull has its own permissive
redistribution terms; the required upstream text is included in
[`QHULL-LICENSE.txt`](QHULL-LICENSE.txt), and modification attribution is in
[`NOTICE`](NOTICE). The original source remains available from
[qhull.org](http://www.qhull.org/).

## Coverage

Included:

- float64 simplicial Quickhull for dimensions 2 and higher;
- outward facet equations, hull vertices, surface measure, and volume;
- ordinary Delaunay triangulation through Qhull's Qbb-scaled paraboloid lift
  and Qz-style point at infinity;
- finite Voronoi vertices, site ridges, unbounded ridge markers, and regions;
- bounded halfspace intersections from a strictly interior feasible point;
- duplicate and interior point handling, full-dimensionality checks, and
  Qhull's distance-roundoff formula.

Not included are Qhull's command-line option language, incremental insertion,
furthest-site Delaunay/Voronoi, joggled input (`QJ`), facet merging,
non-simplicial facet output, coplanar-point metadata, or Qhull's reporting and
statistics system. Exactly cospherical inputs may select a different,
equally-valid diagonal when a non-simplicial lifted facet is triangulated.

## Install

```bash
pixi install
pixi run build
```

The build produces `dist/libmojo-qhull.so`. The pixi environment adds
`python/` to `PYTHONPATH`.

## Usage

```python
import numpy as np
from mojo_qhull import ConvexHull, Delaunay, HalfspaceIntersection, Voronoi

points = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
    [0.4, 0.3],
])

hull = ConvexHull(points)
print(hull.vertices)       # [0 1 2 3]
print(hull.volume)         # 1.0 (area in 2D)

triangles = Delaunay(points).simplices
diagram = Voronoi(points)

halfspaces = np.array([
    [-1.0, 0.0, 0.0],
    [ 1.0, 0.0, -1.0],
    [0.0, -1.0, 0.0],
    [0.0,  1.0, -1.0],
])
vertices = HalfspaceIntersection(
    halfspaces, np.array([0.5, 0.5])
).intersections
```

Run the parity suite with `pixi run test`.

## How it works

Python validates NumPy inputs and allocates all result and scratch buffers.
Their addresses cross the C ABI as 64-bit integers; Mojo reconstructs mutable
raw pointers and never owns Python memory. One shared library contains all
kernels, avoiding per-operation compilation.

The hull uses flat structure-of-arrays storage: `int64` simplex indices,
`float64` normalized facet equations, an integer facet-state array, and one
outside-facet assignment per point. Horizon ridges are matched by indices
instead of Qhull's pointer-linked facet graph. Dead facet slots are reused and
facet adjacency is maintained incrementally. Visible facets are traversed from
the eye point's assigned facet, and point reassignment selects the next eye in
the same pass. Delaunay reuses the maintained adjacency and compacts lower
facets in place instead of rebuilding topology or copying result buffers in
NumPy.

Plane dot products, general-dimensional plane normalization, paraboloid
lifting, polar transforms, and Voronoi-center assembly use native-width
float64 SIMD with scalar remainder loops; 2D and 3D planes use direct closed
forms. Large independent batches of Voronoi-center solves run in parallel once
`count * dim * dim >= 32768`; benchmark-sized batches stay serial to avoid
thread-launch overhead. Halfspaces are mapped to polar points, and polar hull
planes map back to primal intersection vertices with native tolerance-based
deduplication for ordinary result sizes.

There is no GPU path. Plane distance, paraboloid lifting, and polar transforms
are far below the roughly 2 flop/byte threshold, while horizon updates are
control-heavy. High-dimensional center solves can have more arithmetic per
byte, but normal batches are too small to repay device transfers and launch
costs. NumPy float64 C-contiguous inputs remain zero-copy across the FFI
boundary. Other real NumPy dtypes and layouts are converted to C-contiguous
float64; complex-valued inputs are rejected instead of narrowed.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux 6.8.0-136, x86-64. Each entry is the best of three complete Python API
calls. SciPy uses its mature upstream Qhull binding. Mojo is faster for all
measured cases.

| case | mojo-qhull | SciPy/Qhull | ratio |
|---|---:|---:|---:|
| Convex hull (20k x 2) | 1.051 ms | 3.083 ms | 2.93x faster |
| Convex hull (5k x 3) | 0.809 ms | 1.332 ms | 1.65x faster |
| Delaunay (300 x 2) | 0.846 ms | 1.025 ms | 1.21x faster |
| Voronoi (300 x 2) | 1.413 ms | 1.596 ms | 1.13x faster |
| Halfspace intersection (120 x 3) | 0.379 ms | 0.541 ms | 1.43x faster |
