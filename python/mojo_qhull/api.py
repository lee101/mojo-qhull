"""SciPy-shaped public geometry API backed by Mojo kernels."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from ._core import (
    QhullError,
    circumcenters,
    delaunay_arrays,
    hull_measures,
    points_array,
    raw_hull,
    topology_from_neighbors,
)
from ._lib import addr, f64, lib


class ConvexHull:
    """Convex hull of 2-d or higher-dimensional points."""

    def __init__(self, points):
        self.points = points_array(points)
        self.simplices, self.equations, _ = raw_hull(self.points)
        self.vertices = np.unique(self.simplices)
        if self.points.shape[1] == 2:
            center = self.points[self.vertices].mean(axis=0)
            vectors = self.points[self.vertices] - center
            order = np.argsort(np.arctan2(vectors[:, 1], vectors[:, 0]))
            self.vertices = self.vertices[order]
        self.area, self.volume = hull_measures(
            self.points, self.simplices, self.vertices
        )
        self.nsimplex = len(self.simplices)
        self.ndim = self.points.shape[1]


class Delaunay:
    """Simplicial Delaunay triangulation via Qhull's paraboloid transform."""

    def __init__(self, points):
        self.points = points_array(points)
        self.simplices, lifted_equations, hull_neighbors = delaunay_arrays(self.points)
        (
            self.neighbors,
            self._ridge_points,
            self._ridge_first,
            self._ridge_second,
        ) = topology_from_neighbors(self.simplices, hull_neighbors)
        self.convex_hull = self._ridge_points[self._ridge_second < 0]
        self.ndim = self.points.shape[1]
        self.nsimplex = len(self.simplices)
        normal_xy = lifted_equations[:, : self.ndim]
        normal_z = lifted_equations[:, self.ndim, None]
        offset = lifted_equations[:, self.ndim + 1, None]
        self.equations = np.hstack(
            (normal_xy / -normal_z, offset / -normal_z)
        )


class Voronoi:
    """Voronoi diagram represented by vertices, ridges, and point regions."""

    def __init__(self, points):
        self.points = points_array(points)
        triangulation = Delaunay(self.points)
        self.vertices = circumcenters(self.points, triangulation.simplices)
        if self.points.shape[1] == 2:
            self.ridge_points = triangulation._ridge_points
            self.ridge_vertices = np.column_stack(
                (triangulation._ridge_first, triangulation._ridge_second)
            ).tolist()
        else:
            hull_pairs: set[tuple[int, int]] = set()
            for facet in triangulation.convex_hull:
                for a, b in combinations(sorted(facet.tolist()), 2):
                    hull_pairs.add((a, b))
            pair_centers: dict[tuple[int, int], list[int]] = {}
            for simplex_i, simplex in enumerate(triangulation.simplices):
                for a, b in combinations(sorted(simplex.tolist()), 2):
                    pair_centers.setdefault((a, b), []).append(simplex_i)
            # A dual ridge has dimension ndim - 1 and needs at least that many
            # incident simplex centers/directions.  Qhull omits lower-rank
            # hull-edge candidates.
            pair_centers = {
                pair: centers
                for pair, centers in pair_centers.items()
                if len(centers) >= self.points.shape[1] - 1
            }
            self.ridge_points = np.array(list(pair_centers), dtype=np.int64)
            self.ridge_vertices = []
            for pair, incident_centers in pair_centers.items():
                incident = incident_centers.copy()
                if pair in hull_pairs:
                    incident.append(-1)
                self.ridge_vertices.append(incident)

        flat_points = triangulation.simplices.ravel()
        flat_centers = np.repeat(
            np.arange(len(triangulation.simplices), dtype=np.int64),
            triangulation.simplices.shape[1],
        )
        order = np.lexsort((flat_centers, flat_points))
        counts = np.bincount(flat_points, minlength=len(self.points))
        ordered_centers = flat_centers[order]
        offsets = np.concatenate(([0], np.cumsum(counts)))
        self.regions = [
            ordered_centers[offsets[i] : offsets[i + 1]].tolist()
            for i in range(len(self.points))
        ]
        self.point_region = np.arange(len(self.points), dtype=np.int64)
        for point_i in np.unique(triangulation.convex_hull):
            self.regions[int(point_i)].append(-1)
        self.ndim = self.points.shape[1]


class HalfspaceIntersection:
    """Intersection vertices for halfspaces ``A @ x + b <= 0``."""

    def __init__(self, halfspaces, interior_point):
        self.halfspaces = f64(halfspaces)
        self.interior_point = f64(interior_point)
        if self.halfspaces.ndim != 2:
            raise ValueError("halfspaces must be a two-dimensional array")
        count, width = self.halfspaces.shape
        dim = width - 1
        if dim < 2 or self.interior_point.shape != (dim,):
            raise ValueError("interior_point has the wrong shape")
        if count < dim + 1:
            raise QhullError("not enough halfspaces for a bounded intersection")
        if not np.isfinite(self.halfspaces).all():
            raise ValueError("halfspaces must contain only finite coefficients")
        if not np.isfinite(self.interior_point).all():
            raise ValueError("interior_point must contain only finite coordinates")
        dual = np.empty((count, dim), dtype=np.float64)
        status = lib().mq_halfspace_dual(
            addr(self.halfspaces),
            addr(self.interior_point),
            addr(dual),
            count,
            dim,
        )
        if status < 0:
            raise QhullError(
                f"interior point is not strictly inside halfspace {-status - 1}"
            )
        dual_facets, dual_equations, _ = raw_hull(dual)
        intersections = np.empty((len(dual_facets), dim), dtype=np.float64)
        status = lib().mq_polar_vertices(
            addr(dual_equations),
            addr(self.interior_point),
            addr(intersections),
            len(dual_facets),
            dim,
        )
        if status < 0:
            raise QhullError("polar hull does not define a bounded intersection")
        scale = max(float(np.abs(intersections).max()), 1.0)
        quantum = 64 * np.finfo(float).eps * scale
        if len(intersections) <= 1024:
            quantized = np.empty_like(intersections)
            unique_count = lib().mq_compact_unique_rows(
                addr(intersections),
                addr(quantized),
                len(intersections),
                dim,
                quantum,
            )
            self.intersections = intersections[:unique_count]
        else:
            rounded = np.round(intersections / quantum)
            _, unique = np.unique(rounded, axis=0, return_index=True)
            self.intersections = intersections[np.sort(unique)]
        self.dual_points = dual
        self.dual_facets = dual_facets
        present = np.zeros(count, dtype=bool)
        present[dual_facets] = True
        self.dual_vertices = np.flatnonzero(present)
        self.ndim = dim


def convex_hull(points) -> ConvexHull:
    return ConvexHull(points)


def delaunay(points) -> Delaunay:
    return Delaunay(points)


def voronoi(points) -> Voronoi:
    return Voronoi(points)


def halfspace_intersection(halfspaces, interior_point) -> HalfspaceIntersection:
    return HalfspaceIntersection(halfspaces, interior_point)
