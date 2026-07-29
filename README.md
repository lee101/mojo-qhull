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
facet adjacency is maintained incrementally, so horizon discovery does not
rescan every historical facet. Plane dot products and the general-dimensional
plane normalization use native-width float64 SIMD with scalar remainder loops;
2D and 3D planes use direct closed forms. Delaunay runs the same hull on the
scaled paraboloid lift and keeps lower facets. Voronoi centers are dense small
solves over Delaunay simplices. Halfspaces are mapped to polar points; polar
hull planes map back to primal intersection vertices.

CPU execution remains serial at the measured sizes. The independent point
passes do only `O(dim)` work per item, while hull updates are irregular and
dependent, so there is not enough coarse independent work to justify thread
launches here. There is no GPU path: plane distance and paraboloid lifting are
about 0.125 flop/byte, well below the roughly 2 flop/byte threshold, while
horizon updates are control-heavy. Device transfers would therefore make
these kernels slower. NumPy float64 C-contiguous inputs remain zero-copy across
the FFI boundary. Other real NumPy dtypes and layouts are converted to
C-contiguous float64; complex-valued inputs are rejected instead of narrowed.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux 6.8.0-136, x86-64. Each entry is the best of three complete Python API
calls. SciPy uses its mature upstream Qhull binding. Mojo is faster for the
large 2D hull case and slower for the other measured cases.

| case | mojo-qhull | SciPy/Qhull | ratio |
|---|---:|---:|---:|
| Convex hull (20k x 2) | 3.483 ms | 6.758 ms | 1.94x faster |
| Convex hull (5k x 3) | 2.385 ms | 2.379 ms | 1.00x slower |
| Delaunay (300 x 2) | 4.079 ms | 2.021 ms | 0.50x slower |
| Voronoi (300 x 2) | 5.798 ms | 3.589 ms | 0.62x slower |
| Halfspace intersection (120 x 3) | 1.399 ms | 0.910 ms | 0.65x slower |
