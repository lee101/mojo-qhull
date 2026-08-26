import numpy as np
import pytest
from scipy.spatial import HalfspaceIntersection as SciPyHalfspaceIntersection

from mojo_qhull import HalfspaceIntersection, QhullError, halfspace_intersection
from mojo_qhull._lib import addr, lib


def sorted_rows(array, decimals=12):
    return np.array(sorted(map(tuple, np.round(array, decimals=decimals))))


def test_axis_aligned_box_matches_qhull():
    halfspaces = np.array(
        [
            [-1, 0, 0],
            [1, 0, -1],
            [0, -1, 0],
            [0, 1, -1],
        ],
        dtype=float,
    )
    feasible = np.array([0.5, 0.5])
    actual = HalfspaceIntersection(halfspaces, feasible)
    expected = SciPyHalfspaceIntersection(halfspaces, feasible)
    np.testing.assert_allclose(
        sorted_rows(actual.intersections), sorted_rows(expected.intersections)
    )


def test_three_dimensional_polytope_matches_qhull():
    halfspaces = np.array(
        [
            [-1, 0, 0, 0],
            [1, 0, 0, -2],
            [0, -1, 0, 0],
            [0, 1, 0, -3],
            [0, 0, -1, 0],
            [0, 0, 1, -4],
            [1, 1, 1, -7],
        ],
        dtype=float,
    )
    feasible = np.array([0.5, 0.5, 0.5])
    actual = HalfspaceIntersection(halfspaces, feasible)
    expected = SciPyHalfspaceIntersection(halfspaces, feasible)
    np.testing.assert_allclose(
        sorted_rows(actual.intersections), sorted_rows(expected.intersections), atol=2e-12
    )
    assert np.max(
        actual.intersections @ halfspaces[:, :-1].T + halfspaces[:, -1]
    ) < 2e-12


def test_redundant_halfspace_does_not_change_vertices():
    base = np.array([[-1, 0, 0], [1, 0, -1], [0, -1, 0], [0, 1, -1]], float)
    redundant = np.vstack((base, [1, 0, -2], [0, 1, -3]))
    feasible = np.array([0.25, 0.25])
    np.testing.assert_allclose(
        sorted_rows(HalfspaceIntersection(base, feasible).intersections),
        sorted_rows(HalfspaceIntersection(redundant, feasible).intersections),
    )


def test_boundary_or_outside_feasible_point_is_rejected():
    halfspaces = np.array([[-1, 0, 0], [1, 0, -1], [0, -1, 0], [0, 1, -1]], float)
    with pytest.raises(QhullError, match="strictly inside"):
        HalfspaceIntersection(halfspaces, [0.0, 0.5])
    with pytest.raises(QhullError, match="strictly inside"):
        HalfspaceIntersection(halfspaces, [2.0, 0.5])


@pytest.mark.parametrize(
    ("halfspaces", "feasible", "message"),
    [
        (
            [[-1, 0, 0], [1, 0, -1], [0, -1, 0], [0, 1, np.nan]],
            [0.5, 0.5],
            "halfspaces",
        ),
        (
            [[-1, 0, 0], [1, 0, -1], [0, -1, 0], [0, 1, -1]],
            [np.inf, 0.5],
            "interior_point",
        ),
    ],
)
def test_nonfinite_halfspace_inputs_are_rejected_before_ffi(
    halfspaces, feasible, message
):
    with pytest.raises(ValueError, match=message):
        HalfspaceIntersection(halfspaces, feasible)


def test_halfspace_simd_tail_dimension_matches_qhull():
    dim = 6
    halfspaces = np.zeros((2 * dim, dim + 1))
    for axis in range(dim):
        halfspaces[2 * axis, axis] = -1.0
        halfspaces[2 * axis + 1, axis] = 1.0
        halfspaces[2 * axis + 1, -1] = -1.0
    feasible = np.full(dim, 0.5)
    actual = HalfspaceIntersection(halfspaces, feasible)
    expected = SciPyHalfspaceIntersection(halfspaces, feasible)
    np.testing.assert_allclose(
        sorted_rows(actual.intersections),
        sorted_rows(expected.intersections),
        atol=2e-12,
    )


def test_native_row_compaction_preserves_first_occurrences():
    quantum = 1e-6
    values = np.array(
        [[1.0, 2.0], [1.0 + 0.1 * quantum, 2.0], [3.0, 4.0], [1.0, 2.0]]
    )
    quantized = np.empty_like(values)
    count = lib().mq_compact_unique_rows(
        addr(values), addr(quantized), len(values), values.shape[1], quantum
    )
    assert count == 2
    np.testing.assert_array_equal(values[:count], [[1.0, 2.0], [3.0, 4.0]])


def test_halfspace_convenience_function():
    halfspaces = np.array(
        [[-1, 0, 0], [1, 0, -1], [0, -1, 0], [0, 1, -1]], dtype=float
    )
    assert isinstance(
        halfspace_intersection(halfspaces, [0.5, 0.5]), HalfspaceIntersection
    )
