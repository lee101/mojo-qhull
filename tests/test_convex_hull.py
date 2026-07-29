import numpy as np
import pytest
from scipy.spatial import ConvexHull as SciPyConvexHull

from mojo_qhull import ConvexHull, QhullError, convex_hull


def simplex_set(array):
    return {tuple(sorted(row.tolist())) for row in array}


@pytest.mark.parametrize(("dim", "count"), [(2, 50), (3, 50), (4, 35), (5, 25)])
def test_random_hull_facets_and_measures_match_qhull(dim, count):
    points = np.random.default_rng(100 + dim).normal(size=(count, dim))
    actual = ConvexHull(points)
    expected = SciPyConvexHull(points)
    assert simplex_set(actual.simplices) == simplex_set(expected.simplices)
    np.testing.assert_allclose(actual.area, expected.area, rtol=2e-12)
    np.testing.assert_allclose(actual.volume, expected.volume, rtol=2e-12)
    assert set(actual.vertices) == set(expected.vertices)


def test_equations_are_outward_supporting_planes():
    points = np.random.default_rng(4).normal(size=(60, 3))
    hull = ConvexHull(points)
    distances = points @ hull.equations[:, :-1].T + hull.equations[:, -1]
    assert distances.max() < 2e-14
    np.testing.assert_allclose(
        np.linalg.norm(hull.equations[:, :-1], axis=1), 1.0, rtol=2e-15
    )
    for facet, equation in zip(hull.simplices, hull.equations):
        np.testing.assert_allclose(
            points[facet] @ equation[:-1] + equation[-1], 0.0, atol=2e-14
        )


def test_duplicate_and_interior_points_match_qhull():
    points = np.array(
        [[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5], [0, 0], [1, 1]],
        dtype=float,
    )
    actual = ConvexHull(points)
    expected = SciPyConvexHull(points)
    assert simplex_set(actual.simplices) == simplex_set(expected.simplices)
    np.testing.assert_array_equal(actual.vertices, expected.vertices)


@pytest.mark.parametrize(
    "points",
    [
        np.empty((0, 2)),
        np.array([[0.0, 0.0]]),
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
        np.zeros((8, 3)),
    ],
)
def test_degenerate_inputs_raise(points):
    with pytest.raises(QhullError):
        ConvexHull(points)


def test_nonfinite_input_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        ConvexHull([[0, 0], [1, 0], [0, np.nan]])


def test_simd_tail_dimension_matches_qhull():
    points = np.random.default_rng(606).normal(size=(18, 6))
    actual = ConvexHull(points)
    expected = SciPyConvexHull(points)
    assert simplex_set(actual.simplices) == simplex_set(expected.simplices)
    np.testing.assert_allclose(actual.area, expected.area, rtol=2e-12)
    np.testing.assert_allclose(actual.volume, expected.volume, rtol=2e-12)


def test_contiguous_float64_input_stays_zero_copy():
    points = np.random.default_rng(77).normal(size=(40, 3))
    hull = ConvexHull(points)
    assert hull.points is points


@pytest.mark.parametrize(
    "points",
    [
        np.asfortranarray(np.random.default_rng(78).normal(size=(40, 3))),
        np.random.default_rng(79).normal(size=(40, 6))[:, ::2],
        np.random.default_rng(80).normal(size=(40, 3)).astype(np.float32),
    ],
)
def test_layout_and_real_dtype_are_converted_to_ffi_format(points):
    hull = ConvexHull(points)
    expected = SciPyConvexHull(np.asarray(points, dtype=np.float64))
    assert hull.points.dtype == np.float64
    assert hull.points.flags.c_contiguous
    assert simplex_set(hull.simplices) == simplex_set(expected.simplices)


def test_complex_input_is_rejected_instead_of_silently_narrowed():
    points = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.complex128)
    with pytest.raises(TypeError, match="complex"):
        ConvexHull(points)


def test_convex_hull_convenience_function():
    points = np.random.default_rng(81).normal(size=(20, 2))
    assert isinstance(convex_hull(points), ConvexHull)
