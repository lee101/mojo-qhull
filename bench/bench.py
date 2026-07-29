"""Benchmark mojo-qhull against SciPy's bindings to upstream Qhull."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np
from scipy.spatial import ConvexHull as SciPyConvexHull
from scipy.spatial import Delaunay as SciPyDelaunay
from scipy.spatial import HalfspaceIntersection as SciPyHalfspaceIntersection
from scipy.spatial import Voronoi as SciPyVoronoi

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)

from mojo_qhull import ConvexHull, Delaunay, HalfspaceIntersection, Voronoi  # noqa: E402


def timeit(function, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def machine_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def cases():
    rng = np.random.default_rng(42)
    points_2d = rng.normal(size=(20_000, 2))
    points_3d = rng.normal(size=(5_000, 3))
    sites = rng.random((300, 2))
    normals = rng.normal(size=(120, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    halfspaces = np.hstack((normals, -np.ones((len(normals), 1))))
    feasible = np.zeros(3)
    return [
        (
            "Convex hull (20k x 2)",
            lambda: ConvexHull(points_2d),
            lambda: SciPyConvexHull(points_2d),
        ),
        (
            "Convex hull (5k x 3)",
            lambda: ConvexHull(points_3d),
            lambda: SciPyConvexHull(points_3d),
        ),
        (
            "Delaunay (300 x 2)",
            lambda: Delaunay(sites),
            lambda: SciPyDelaunay(sites),
        ),
        (
            "Voronoi (300 x 2)",
            lambda: Voronoi(sites),
            lambda: SciPyVoronoi(sites),
        ),
        (
            "Halfspace intersection (120 x 3)",
            lambda: HalfspaceIntersection(halfspaces, feasible),
            lambda: SciPyHalfspaceIntersection(halfspaces, feasible),
        ),
    ]


def main():
    benchmark_cases = cases()
    benchmark_cases[0][1]()
    print(f"Machine: {machine_name()} ({platform.platform()})")
    print("| case | mojo-qhull | SciPy/Qhull | ratio |")
    print("|---|---:|---:|---:|")
    for name, ours, reference in benchmark_cases:
        mojo_time = timeit(ours)
        scipy_time = timeit(reference)
        ratio = scipy_time / mojo_time
        label = "faster" if ratio >= 1 else "slower"
        print(
            f"| {name} | {mojo_time * 1e3:.3f} ms | "
            f"{scipy_time * 1e3:.3f} ms | {ratio:.2f}x {label} |"
        )


if __name__ == "__main__":
    main()
