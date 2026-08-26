from itertools import combinations

import numpy as np
import pytest
from scipy.spatial import Delaunay as SciPyDelaunay
from scipy.spatial import Voronoi as SciPyVoronoi

from mojo_qhull import Delaunay, QhullError, Voronoi, delaunay, voronoi
from mojo_qhull._core import circumcenters


def simplex_set(array):
    return {tuple(sorted(row.tolist())) for row in array}


@pytest.mark.parametrize(("dim", "count"), [(2, 45), (3, 32), (4, 24)])
def test_delaunay_simplices_match_qhull(dim, count):
    points = np.random.default_rng(200 + dim).normal(size=(count, dim))
    actual = Delaunay(points)
    expected = SciPyDelaunay(points)
    assert simplex_set(actual.simplices) == simplex_set(expected.simplices)
    assert actual.nsimplex == len(expected.simplices)


def test_delaunay_neighbors_are_reciprocal_and_match_ridges():
    points = np.random.default_rng(8).random((40, 3))
    triangulation = Delaunay(points)
    for simplex_i, row in enumerate(triangulation.neighbors):
        for opposite, neighbor in enumerate(row):
            if neighbor >= 0:
                assert simplex_i in triangulation.neighbors[neighbor]
                ridge = set(np.delete(triangulation.simplices[simplex_i], opposite))
                assert len(ridge & set(triangulation.simplices[neighbor])) == 3


def test_cocircular_square_uses_qhull_qz_point_at_infinity():
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    actual = Delaunay(points)
    expected = SciPyDelaunay(points)
    assert actual.simplices.shape == expected.simplices.shape == (2, 3)
    assert set(actual.simplices.ravel()) == set(expected.simplices.ravel()) == set(range(4))
    actual_area = sum(
        abs(
            (points[s[1], 0] - points[s[0], 0])
            * (points[s[2], 1] - points[s[0], 1])
            - (points[s[1], 1] - points[s[0], 1])
            * (points[s[2], 0] - points[s[0], 0])
        )
        / 2.0
        for s in actual.simplices
    )
    np.testing.assert_allclose(actual_area, 1.0)


def test_duplicate_site_is_not_used_as_a_new_simplex_vertex():
    points = np.array(
        [[0, 0], [1, 0], [1, 1], [0, 1], [0.2, 0.3], [0.2, 0.3]],
        dtype=float,
    )
    actual = Delaunay(points)
    expected = SciPyDelaunay(points)
    assert len(actual.simplices) == len(expected.simplices)
    for simplex in actual.simplices:
        assert not ({4, 5} <= set(simplex))


def test_voronoi_vertices_match_qhull():
    points = np.random.default_rng(17).random((35, 2))
    actual = Voronoi(points)
    expected = SciPyVoronoi(points)
    actual_vertices = np.array(
        sorted(map(tuple, np.round(actual.vertices, decimals=11)))
    )
    expected_vertices = np.array(
        sorted(map(tuple, np.round(expected.vertices, decimals=11)))
    )
    np.testing.assert_allclose(actual_vertices, expected_vertices, atol=2e-11)
    assert {tuple(sorted(pair)) for pair in actual.ridge_points} == {
        tuple(sorted(pair)) for pair in expected.ridge_points
    }


def test_three_dimensional_voronoi_vertices_and_site_ridges_match_qhull():
    points = np.random.default_rng(18).random((24, 3))
    actual = Voronoi(points)
    expected = SciPyVoronoi(points)
    np.testing.assert_allclose(
        np.array(sorted(map(tuple, np.round(actual.vertices, decimals=10)))),
        np.array(sorted(map(tuple, np.round(expected.vertices, decimals=10)))),
        atol=2e-10,
    )
    assert {tuple(sorted(pair)) for pair in actual.ridge_points} == {
        tuple(sorted(pair)) for pair in expected.ridge_points
    }


def test_four_dimensional_voronoi_site_ridges_match_qhull():
    points = np.random.default_rng(1041).random((18, 4))
    actual = Voronoi(points)
    expected = SciPyVoronoi(points)
    assert {tuple(sorted(pair)) for pair in actual.ridge_points} == {
        tuple(sorted(pair)) for pair in expected.ridge_points
    }


def test_voronoi_ridges_reference_incident_delaunay_centers():
    points = np.random.default_rng(19).random((25, 2))
    diagram = Voronoi(points)
    triangulation = Delaunay(points)
    for pair, vertex_ids in zip(diagram.ridge_points, diagram.ridge_vertices):
        finite_ids = [value for value in vertex_ids if value >= 0]
        expected = [
            index
            for index, simplex in enumerate(triangulation.simplices)
            if set(pair) <= set(simplex)
        ]
        assert finite_ids == expected


def test_collinear_delaunay_and_voronoi_raise():
    points = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
    with pytest.raises(QhullError):
        Delaunay(points)
    with pytest.raises(QhullError):
        Voronoi(points)


def test_benchmark_scale_delaunay_adjacency_matches_qhull():
    points = np.random.default_rng(42).random((300, 2))
    actual = Delaunay(points)
    expected = SciPyDelaunay(points)
    assert simplex_set(actual.simplices) == simplex_set(expected.simplices)
    assert np.count_nonzero(actual.neighbors < 0) == len(expected.convex_hull)


def test_delaunay_simd_tail_dimension_matches_qhull():
    points = np.random.default_rng(607).normal(size=(14, 6))
    actual = Delaunay(points)
    expected = SciPyDelaunay(points)
    assert simplex_set(actual.simplices) == simplex_set(expected.simplices)


@pytest.mark.parametrize("count", [32, 8192])
def test_circumcenter_serial_and_parallel_thresholds(count):
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    simplices = np.tile(np.array([[0, 1, 2]], dtype=np.int64), (count, 1))
    centers = circumcenters(points, simplices)
    np.testing.assert_allclose(centers, 0.5)


def test_delaunay_and_voronoi_convenience_functions():
    points = np.random.default_rng(43).random((20, 2))
    assert isinstance(delaunay(points), Delaunay)
    assert isinstance(voronoi(points), Voronoi)
