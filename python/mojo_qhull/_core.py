"""Internal buffer management and topology assembly."""

from __future__ import annotations

import math

import numpy as np

from ._lib import addr, f64, lib


class QhullError(ValueError):
    """The input is geometrically invalid for the requested operation."""


def points_array(points) -> np.ndarray:
    result = f64(points)
    if result.ndim != 2:
        raise ValueError("points must be a two-dimensional array")
    if not np.isfinite(result).all():
        raise ValueError("points must contain only finite coordinates")
    if result.shape[1] < 2:
        raise ValueError("points must have at least two coordinates")
    return result


def raw_hull(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, dim = points.shape
    if n < dim + 1:
        raise QhullError(f"not enough points ({n}) to construct a {dim}-d hull")
    capacity = max(dim + 1, 8 * n)
    for _ in range(9):
        facets = np.empty((capacity, dim), dtype=np.int64)
        planes = np.empty((capacity, dim + 1), dtype=np.float64)
        alive = np.empty(capacity, dtype=np.int64)
        neighbors = np.empty((capacity, dim), dtype=np.int64)
        new_facets = np.empty(capacity, dtype=np.int64)
        simplex = np.empty(dim + 1, dtype=np.int64)
        interior = np.empty(dim, dtype=np.float64)
        basis = np.empty((dim, dim), dtype=np.float64)
        residual = np.empty(dim, dtype=np.float64)
        matrix = np.empty((dim, dim), dtype=np.float64)
        point_facets = np.empty(n, dtype=np.int64)
        point_distances = np.empty(n, dtype=np.float64)
        count = lib().mq_convex_hull(
            addr(points),
            n,
            dim,
            addr(facets),
            addr(planes),
            addr(alive),
            addr(neighbors),
            addr(new_facets),
            capacity,
            addr(simplex),
            addr(interior),
            addr(basis),
            addr(residual),
            addr(matrix),
            addr(point_facets),
            addr(point_distances),
        )
        if count >= 0:
            return facets[:count].copy(), planes[:count].copy()
        if count == -2:
            raise QhullError("input is less than full dimensional")
        if count != -3:
            raise QhullError("unable to construct initial hull")
        capacity *= 2
    raise QhullError("facet capacity exceeded")


def simplex_measure(vertices: np.ndarray) -> float:
    dim = vertices.shape[0] - 1
    matrix = vertices[1:] - vertices[0]
    return abs(float(np.linalg.det(matrix))) / math.factorial(dim)


def hull_measures(
    points: np.ndarray, simplices: np.ndarray, vertices: np.ndarray
) -> tuple[float, float]:
    dim = points.shape[1]
    center = points[vertices].mean(axis=0)
    faces = points[simplices]
    edges = faces[:, 1:] - faces[:, :1]
    gram = edges @ np.swapaxes(edges, 1, 2)
    area = np.sqrt(np.maximum(np.linalg.det(gram), 0.0)).sum()
    area /= math.factorial(dim - 1)
    volume_matrices = faces - center
    volume = np.abs(np.linalg.det(volume_matrices)).sum() / math.factorial(dim)
    return float(area), float(volume)


def delaunay_arrays(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, dim = points.shape
    if n < dim + 1:
        raise QhullError(f"not enough points ({n}) for a {dim}-d triangulation")
    lifted = np.empty((n + 1, dim + 1), dtype=np.float64)
    lib().mq_lift_delaunay(addr(points), addr(lifted), n, dim)
    facets, equations = raw_hull(lifted)
    cutoff = -2.0 * 1.01 * (dim + 1) * np.finfo(np.float64).eps
    keep = (equations[:, dim] < cutoff) & np.all(facets != n, axis=1)
    simplices = facets[keep].copy()
    if len(simplices) == 0:
        raise QhullError("no lower facets in lifted hull")
    return simplices, equations[keep].copy()


def simplex_topology(
    simplices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count, width = simplices.shape
    ridges = np.empty((count, width, width - 1), dtype=np.int64)
    for opposite in range(width):
        ridges[:, opposite, :opposite] = simplices[:, :opposite]
        ridges[:, opposite, opposite:] = simplices[:, opposite + 1 :]
    ridges.sort(axis=2)
    flat_ridges = ridges.reshape(count * width, width - 1)
    order = np.lexsort(flat_ridges[:, ::-1].T)
    ordered_ridges = flat_ridges[order]
    matches = np.all(ordered_ridges[1:] == ordered_ridges[:-1], axis=1)
    left = order[:-1][matches]
    right = order[1:][matches]
    neighbors = np.full(count * width, -1, dtype=np.int64)
    neighbors[left] = right // width
    neighbors[right] = left // width
    owners = np.repeat(np.arange(count, dtype=np.int64), width)
    keep = (neighbors < 0) | (owners < neighbors)
    return (
        neighbors.reshape(count, width),
        flat_ridges[keep].copy(),
        owners[keep],
        neighbors[keep],
    )


def simplex_neighbors(simplices: np.ndarray) -> np.ndarray:
    return simplex_topology(simplices)[0]


def circumcenters(points: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    dim = points.shape[1]
    centers = np.empty((len(simplices), dim), dtype=np.float64)
    matrix = np.empty((dim, dim), dtype=np.float64)
    rhs = np.empty(dim, dtype=np.float64)
    status = lib().mq_circumcenters(
        addr(points),
        addr(simplices),
        addr(centers),
        len(simplices),
        dim,
        addr(matrix),
        addr(rhs),
    )
    if status < 0:
        raise QhullError("singular Delaunay simplex")
    return centers
