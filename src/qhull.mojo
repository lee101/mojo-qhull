"""Simplicial Quickhull kernels and the C ABI used by Python.

All storage is caller-owned and row-major.  Facets are flat Int64 indices,
planes are [normal..., offset], and state arrays replace Qhull's pointer graph.
"""

# Derived from Qhull, copyright (c) 1993-2020 C.B. Barber and The Geometry
# Center. See QHULL-LICENSE.txt and NOTICE in the repository root.

from std.math import sqrt
from std.sys import simd_width_of

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime REAL_EPSILON = 2.220446049250313e-16


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


# qhull: src/libqhull_r/geom_r.c qh_distplane
def dist_plane(points: FPtr, point_i: Int, dim: Int, planes: FPtr, facet_i: Int) -> Float64:
    var point_base = point_i * dim
    var plane_base = facet_i * (dim + 1)
    if dim == 2:
        return (
            planes[plane_base + 2]
            + points[point_base] * planes[plane_base]
            + points[point_base + 1] * planes[plane_base + 1]
        )
    if dim == 3:
        return (
            planes[plane_base + 3]
            + points[point_base] * planes[plane_base]
            + points[point_base + 1] * planes[plane_base + 1]
            + points[point_base + 2] * planes[plane_base + 2]
        )
    var value = planes[plane_base + dim]
    comptime W = simd_width_of[DType.float64]()
    var k = 0
    while k + W <= dim:
        var dot = (
            points.load[width=W](point_i * dim + k)
            * planes.load[width=W](plane_base + k)
        ).reduce_add()
        value += dot
        k += W
    while k < dim:
        value += points[point_base + k] * planes[plane_base + k]
        k += 1
    return value


# qhull: src/libqhull_r/geom2_r.c qh_distround
def distance_roundoff(points: FPtr, n: Int, dim: Int) -> Float64:
    var maxabs = 0.0
    var maxsumabs = 0.0
    for k in range(dim):
        var coordinate_max = 0.0
        for i in range(n):
            var value = abs(points[i * dim + k])
            maxabs = max(maxabs, value)
            coordinate_max = max(coordinate_max, value)
        maxsumabs += coordinate_max
    var maxdistsum = min(sqrt(Float64(dim)) * maxabs, maxsumabs)
    return REAL_EPSILON * (Float64(dim) * maxdistsum * 1.01 + maxabs)


# qhull: src/libqhull_r/geom2_r.c qh_determinant
def determinant(a: FPtr, n: Int) -> Float64:
    var sign = 1.0
    var value = 1.0
    for col in range(n):
        var pivot = col
        var best = abs(a[col * n + col])
        for row in range(col + 1, n):
            var candidate = abs(a[row * n + col])
            if candidate > best:
                best = candidate
                pivot = row
        if best == 0.0:
            return 0.0
        if pivot != col:
            for j in range(n):
                var tmp = a[col * n + j]
                a[col * n + j] = a[pivot * n + j]
                a[pivot * n + j] = tmp
            sign = -sign
        var diagonal = a[col * n + col]
        value *= diagonal
        for row in range(col + 1, n):
            var scale = a[row * n + col] / diagonal
            for j in range(col + 1, n):
                a[row * n + j] -= scale * a[col * n + j]
    return sign * value


# qhull: src/libqhull_r/geom_r.c qh_setfacetplane/qh_sethyperplane_det
def set_facet_plane(
    points: FPtr,
    dim: Int,
    facets: IPtr,
    planes: FPtr,
    facet_i: Int,
    interior: FPtr,
    matrix: FPtr,
) -> Bool:
    var base = facets[facet_i * dim]
    if dim == 2:
        var other = facets[facet_i * dim + 1]
        var nx = points[Int(other) * 2 + 1] - points[Int(base) * 2 + 1]
        var ny = points[Int(base) * 2] - points[Int(other) * 2]
        var norm2 = nx * nx + ny * ny
        if norm2 == 0.0:
            return False
        var invnorm = 1.0 / sqrt(norm2)
        nx *= invnorm
        ny *= invnorm
        var offset = -(points[Int(base) * 2] * nx + points[Int(base) * 2 + 1] * ny)
        if offset + interior[0] * nx + interior[1] * ny > 0.0:
            nx = -nx
            ny = -ny
            offset = -offset
        planes[facet_i * 3] = nx
        planes[facet_i * 3 + 1] = ny
        planes[facet_i * 3 + 2] = offset
        return True
    if dim == 3:
        var b = facets[facet_i * 3 + 1]
        var c = facets[facet_i * 3 + 2]
        var ux = points[Int(b) * 3] - points[Int(base) * 3]
        var uy = points[Int(b) * 3 + 1] - points[Int(base) * 3 + 1]
        var uz = points[Int(b) * 3 + 2] - points[Int(base) * 3 + 2]
        var vx = points[Int(c) * 3] - points[Int(base) * 3]
        var vy = points[Int(c) * 3 + 1] - points[Int(base) * 3 + 1]
        var vz = points[Int(c) * 3 + 2] - points[Int(base) * 3 + 2]
        var nx = uy * vz - uz * vy
        var ny = uz * vx - ux * vz
        var nz = ux * vy - uy * vx
        var norm2 = nx * nx + ny * ny + nz * nz
        if norm2 == 0.0:
            return False
        var invnorm = 1.0 / sqrt(norm2)
        nx *= invnorm
        ny *= invnorm
        nz *= invnorm
        var offset = -(
            points[Int(base) * 3] * nx
            + points[Int(base) * 3 + 1] * ny
            + points[Int(base) * 3 + 2] * nz
        )
        if (
            offset + interior[0] * nx + interior[1] * ny + interior[2] * nz
            > 0.0
        ):
            nx = -nx
            ny = -ny
            nz = -nz
            offset = -offset
        planes[facet_i * 4] = nx
        planes[facet_i * 4 + 1] = ny
        planes[facet_i * 4 + 2] = nz
        planes[facet_i * 4 + 3] = offset
        return True
    var norm2 = 0.0
    for omitted in range(dim):
        var column_out = 0
        for row in range(dim - 1):
            var vertex = facets[facet_i * dim + row + 1]
            column_out = 0
            for column in range(dim):
                if column != omitted:
                    matrix[row * (dim - 1) + column_out] = (
                        points[Int(vertex) * dim + column] - points[Int(base) * dim + column]
                    )
                    column_out += 1
        var coefficient = determinant(matrix, dim - 1)
        if omitted % 2 == 1:
            coefficient = -coefficient
        planes[facet_i * (dim + 1) + omitted] = coefficient
        norm2 += coefficient * coefficient
    if norm2 == 0.0:
        return False
    var invnorm = 1.0 / sqrt(norm2)
    var offset = 0.0
    comptime W = simd_width_of[DType.float64]()
    var k = 0
    while k + W <= dim:
        var normal = planes.load[width=W](facet_i * (dim + 1) + k) * invnorm
        planes.store(facet_i * (dim + 1) + k, normal)
        var base_dot = (
            points.load[width=W](Int(base) * dim + k) * normal
        ).reduce_add()
        offset -= base_dot
        k += W
    while k < dim:
        planes[facet_i * (dim + 1) + k] *= invnorm
        offset -= (
            points[Int(base) * dim + k] * planes[facet_i * (dim + 1) + k]
        )
        k += 1
    planes[facet_i * (dim + 1) + dim] = offset
    var interior_dist = offset
    k = 0
    while k + W <= dim:
        var interior_dot = (
            interior.load[width=W](k)
            * planes.load[width=W](facet_i * (dim + 1) + k)
        ).reduce_add()
        interior_dist += interior_dot
        k += W
    while k < dim:
        interior_dist += interior[k] * planes[facet_i * (dim + 1) + k]
        k += 1
    if interior_dist > 0.0:
        k = 0
        while k + W <= dim + 1:
            planes.store(
                facet_i * (dim + 1) + k,
                -planes.load[width=W](facet_i * (dim + 1) + k),
            )
            k += W
        while k < dim + 1:
            planes[facet_i * (dim + 1) + k] = -planes[facet_i * (dim + 1) + k]
            k += 1
    return True


def selected(simplex: IPtr, count: Int, point_i: Int) -> Bool:
    for j in range(count):
        if simplex[j] == Int64(point_i):
            return True
    return False


# qhull: src/libqhull_r/geom2_r.c qh_maxsimplex
def initial_simplex(
    points: FPtr,
    n: Int,
    dim: Int,
    simplex: IPtr,
    basis: FPtr,
    residual: FPtr,
    roundoff: Float64,
) -> Bool:
    var widest = -1.0
    var min_i = 0
    var max_i = 0
    for k in range(dim):
        var lo = points[k]
        var hi = points[k]
        var lo_i = 0
        var hi_i = 0
        for i in range(1, n):
            var value = points[i * dim + k]
            if value < lo:
                lo = value
                lo_i = i
            if value > hi:
                hi = value
                hi_i = i
        if hi - lo > widest:
            widest = hi - lo
            min_i = lo_i
            max_i = hi_i
    if widest <= 2.0 * roundoff:
        return False
    simplex[0] = Int64(min_i)
    simplex[1] = Int64(max_i)
    var length2 = 0.0
    for j in range(dim):
        var value = points[max_i * dim + j] - points[min_i * dim + j]
        basis[j] = value
        length2 += value * value
    var invlength = 1.0 / sqrt(length2)
    for j in range(dim):
        basis[j] *= invlength

    for count in range(2, dim + 1):
        var best_i = -1
        var best2 = -1.0
        for i in range(n):
            if selected(simplex, count, i):
                continue
            for j in range(dim):
                residual[j] = points[i * dim + j] - points[min_i * dim + j]
            for b in range(count - 1):
                var projection = 0.0
                for j in range(dim):
                    projection += residual[j] * basis[b * dim + j]
                for j in range(dim):
                    residual[j] -= projection * basis[b * dim + j]
            var candidate2 = 0.0
            for j in range(dim):
                candidate2 += residual[j] * residual[j]
            if candidate2 > best2:
                best2 = candidate2
                best_i = i
        if best_i < 0 or sqrt(max(best2, 0.0)) <= 2.0 * roundoff:
            return False
        simplex[count] = Int64(best_i)
        for j in range(dim):
            residual[j] = points[best_i * dim + j] - points[min_i * dim + j]
        for b in range(count - 1):
            var projection = 0.0
            for j in range(dim):
                projection += residual[j] * basis[b * dim + j]
            for j in range(dim):
                residual[j] -= projection * basis[b * dim + j]
        var inverse = 1.0 / sqrt(best2)
        for j in range(dim):
            basis[(count - 1) * dim + j] = residual[j] * inverse
    return True


def contains_vertex(facets: IPtr, facet_i: Int, dim: Int, vertex: Int64) -> Bool:
    for j in range(dim):
        if facets[facet_i * dim + j] == vertex:
            return True
    return False


def shares_ridge(
    facets: IPtr, visible_facet: Int, omitted: Int, other_facet: Int, dim: Int
) -> Bool:
    for j in range(dim):
        if j != omitted:
            if not contains_vertex(
                facets, other_facet, dim, facets[visible_facet * dim + j]
            ):
                return False
    return True


# qhull: src/libqhull_r/libqhull_r.c qh_buildhull/qh_partitionall
def quickhull(
    points: FPtr,
    n: Int,
    dim: Int,
    facets: IPtr,
    planes: FPtr,
    alive: IPtr,
    neighbors: IPtr,
    new_facets: IPtr,
    capacity: Int,
    simplex: IPtr,
    interior: FPtr,
    basis: FPtr,
    residual: FPtr,
    matrix: FPtr,
    point_facets: IPtr,
    point_distances: FPtr,
) -> Int:
    if dim < 2 or n < dim + 1 or capacity < dim + 1:
        return -1
    var roundoff = distance_roundoff(points, n, dim)
    if not initial_simplex(points, n, dim, simplex, basis, residual, roundoff):
        return -2
    for k in range(dim):
        interior[k] = 0.0
        for j in range(dim + 1):
            interior[k] += points[Int(simplex[j]) * dim + k]
        interior[k] /= Float64(dim + 1)

    var facet_count = dim + 1
    for omitted in range(dim + 1):
        var dst = 0
        for j in range(dim + 1):
            if j != omitted:
                facets[omitted * dim + dst] = simplex[j]
                dst += 1
        alive[omitted] = 1
        if not set_facet_plane(points, dim, facets, planes, omitted, interior, matrix):
            return -2
    for f in range(facet_count):
        for omitted in range(dim):
            neighbors[f * dim + omitted] = -1
            for other in range(facet_count):
                if other != f and shares_ridge(facets, f, omitted, other, dim):
                    neighbors[f * dim + omitted] = Int64(other)
                    break

    # qhull: src/libqhull_r/libqhull_r.c qh_partitionall
    for i in range(n):
        point_facets[i] = -1
        point_distances[i] = 2.0 * roundoff
        for f in range(facet_count):
            var distance = dist_plane(points, i, dim, planes, f)
            if distance > point_distances[i]:
                point_distances[i] = distance
                point_facets[i] = Int64(f)

    var iterations = 0
    while iterations < n:
        var eye = -1
        var furthest = 2.0 * roundoff
        for i in range(n):
            if point_facets[i] >= 0 and point_distances[i] > furthest:
                furthest = point_distances[i]
                eye = i
        if eye < 0:
            break

        var visible_count = 0
        for f in range(facet_count):
            if alive[f] == 1 and dist_plane(points, eye, dim, planes, f) > roundoff:
                alive[f] = 2
                visible_count += 1
        if visible_count == 0:
            break

        var before_new = facet_count
        var next_free = 0
        var new_count = 0
        for f in range(before_new):
            if alive[f] != 2:
                continue
            for omitted in range(dim):
                var neighbor = Int(neighbors[f * dim + omitted])
                if neighbor >= 0:
                    if alive[neighbor] != 1:
                        continue
                    while next_free < before_new and alive[next_free] != 0:
                        next_free += 1
                    var new_facet = facet_count
                    if next_free < before_new:
                        new_facet = next_free
                        next_free += 1
                    else:
                        if facet_count >= capacity:
                            return -3
                        facet_count += 1
                    var dst = 0
                    for j in range(dim):
                        if j != omitted:
                            facets[new_facet * dim + dst] = facets[f * dim + j]
                            dst += 1
                    facets[new_facet * dim + dim - 1] = Int64(eye)
                    alive[new_facet] = 3
                    for j in range(dim):
                        neighbors[new_facet * dim + j] = -1
                    if not set_facet_plane(
                        points, dim, facets, planes, new_facet, interior, matrix
                    ):
                        alive[new_facet] = 0
                    else:
                        neighbors[new_facet * dim + dim - 1] = Int64(neighbor)
                        for neighbor_omitted in range(dim):
                            if neighbors[neighbor * dim + neighbor_omitted] == Int64(f):
                                neighbors[neighbor * dim + neighbor_omitted] = Int64(new_facet)
                                break
                        new_facets[new_count] = Int64(new_facet)
                        new_count += 1

        for new_i in range(new_count):
            var new_facet = Int(new_facets[new_i])
            for omitted in range(dim - 1):
                if neighbors[new_facet * dim + omitted] >= 0:
                    continue
                for other_i in range(new_i + 1, new_count):
                    var other = Int(new_facets[other_i])
                    if shares_ridge(facets, new_facet, omitted, other, dim):
                        neighbors[new_facet * dim + omitted] = Int64(other)
                        for other_omitted in range(dim - 1):
                            if shares_ridge(
                                facets, other, other_omitted, new_facet, dim
                            ):
                                neighbors[other * dim + other_omitted] = Int64(new_facet)
                                break
                        break

        # qhull: src/libqhull_r/poly_r.c qh_partitionvisible
        for i in range(n):
            var assigned = point_facets[i]
            if assigned >= 0 and alive[Int(assigned)] == 2:
                point_facets[i] = -1
                point_distances[i] = 2.0 * roundoff
                for f in range(facet_count):
                    if alive[f] == 3:
                        var distance = dist_plane(points, i, dim, planes, f)
                        if distance > point_distances[i]:
                            point_distances[i] = distance
                            point_facets[i] = Int64(f)
        for f in range(before_new):
            if alive[f] == 2:
                alive[f] = 0
        for f in range(facet_count):
            if alive[f] == 3:
                alive[f] = 1
        iterations += 1

    var output_count = 0
    for f in range(facet_count):
        if alive[f] == 1:
            if output_count != f:
                for j in range(dim):
                    facets[output_count * dim + j] = facets[f * dim + j]
                for j in range(dim + 1):
                    planes[output_count * (dim + 1) + j] = planes[f * (dim + 1) + j]
            output_count += 1
    return output_count


@export("mq_convex_hull")
def mq_convex_hull(
    points_addr: Int,
    n: Int,
    dim: Int,
    facets_addr: Int,
    planes_addr: Int,
    alive_addr: Int,
    neighbors_addr: Int,
    new_facets_addr: Int,
    capacity: Int,
    simplex_addr: Int,
    interior_addr: Int,
    basis_addr: Int,
    residual_addr: Int,
    matrix_addr: Int,
    point_facets_addr: Int,
    point_distances_addr: Int,
) abi("C") -> Int:
    return quickhull(
        fp(points_addr),
        n,
        dim,
        ip(facets_addr),
        fp(planes_addr),
        ip(alive_addr),
        ip(neighbors_addr),
        ip(new_facets_addr),
        capacity,
        ip(simplex_addr),
        fp(interior_addr),
        fp(basis_addr),
        fp(residual_addr),
        fp(matrix_addr),
        ip(point_facets_addr),
        fp(point_distances_addr),
    )


# qhull: src/libqhull_r/geom2_r.c qh_setdelaunay/qh_scalelast and io_r.c qh_readpoints
@export("mq_lift_delaunay")
def mq_lift_delaunay(points_addr: Int, lifted_addr: Int, n: Int, dim: Int) abi("C"):
    var points = fp(points_addr)
    var lifted = fp(lifted_addr)
    var maxboloid = 0.0
    var maxcoord = 0.0
    for k in range(dim):
        lifted[n * (dim + 1) + k] = 0.0
    for i in range(n):
        var paraboloid = 0.0
        for k in range(dim):
            var value = points[i * dim + k]
            lifted[i * (dim + 1) + k] = value
            lifted[n * (dim + 1) + k] += value / Float64(n)
            paraboloid += value * value
            maxcoord = max(maxcoord, abs(value))
        lifted[i * (dim + 1) + dim] = paraboloid
        maxboloid = max(maxboloid, paraboloid)
    lifted[n * (dim + 1) + dim] = max(maxboloid * 1.1, 1.0)

    var low = lifted[dim]
    var high = low
    for i in range(1, n + 1):
        low = min(low, lifted[i * (dim + 1) + dim])
        high = max(high, lifted[i * (dim + 1) + dim])
    if high > low:
        var scale = maxcoord / (high - low)
        var shift = -low * scale
        for i in range(n + 1):
            lifted[i * (dim + 1) + dim] = (
                lifted[i * (dim + 1) + dim] * scale + shift
            )


# qhull: src/libqhull_r/geom2_r.c qh_sethalfspace_all/qh_sethalfspace
@export("mq_halfspace_dual")
def mq_halfspace_dual(
    halfspaces_addr: Int,
    feasible_addr: Int,
    dual_addr: Int,
    count: Int,
    dim: Int,
) abi("C") -> Int:
    var halfspaces = fp(halfspaces_addr)
    var feasible = fp(feasible_addr)
    var dual = fp(dual_addr)
    for i in range(count):
        var distance = halfspaces[i * (dim + 1) + dim]
        for k in range(dim):
            distance += halfspaces[i * (dim + 1) + k] * feasible[k]
        if distance >= 0.0:
            return -(i + 1)
        for k in range(dim):
            dual[i * dim + k] = halfspaces[i * (dim + 1) + k] / -distance
    return 0


# qhull: src/libqhull_r/geom2_r.c qh_sethalfspace_all (polar reconstruction)
@export("mq_polar_vertices")
def mq_polar_vertices(
    planes_addr: Int,
    feasible_addr: Int,
    vertices_addr: Int,
    count: Int,
    dim: Int,
) abi("C") -> Int:
    var planes = fp(planes_addr)
    var feasible = fp(feasible_addr)
    var vertices = fp(vertices_addr)
    for i in range(count):
        var offset = planes[i * (dim + 1) + dim]
        if offset >= 0.0:
            return -1
        for k in range(dim):
            vertices[i * dim + k] = (
                feasible[k] - planes[i * (dim + 1) + k] / offset
            )
    return count


# qhull: src/libqhull_r/geom2_r.c qh_gausselim/qh_backnormal
def solve_linear(a: FPtr, b: FPtr, n: Int) -> Bool:
    for col in range(n):
        var pivot = col
        var best = abs(a[col * n + col])
        for row in range(col + 1, n):
            var candidate = abs(a[row * n + col])
            if candidate > best:
                best = candidate
                pivot = row
        if best <= REAL_EPSILON:
            return False
        if pivot != col:
            for j in range(n):
                var tmp = a[col * n + j]
                a[col * n + j] = a[pivot * n + j]
                a[pivot * n + j] = tmp
            var rhs = b[col]
            b[col] = b[pivot]
            b[pivot] = rhs
        var diagonal = a[col * n + col]
        for row in range(col + 1, n):
            var scale = a[row * n + col] / diagonal
            for j in range(col + 1, n):
                a[row * n + j] -= scale * a[col * n + j]
            b[row] -= scale * b[col]
    for ii in range(n):
        var row = n - 1 - ii
        var value = b[row]
        for j in range(row + 1, n):
            value -= a[row * n + j] * b[j]
        b[row] = value / a[row * n + row]
    return True


# qhull: src/libqhull_r/geom2_r.c qh_voronoi_center
@export("mq_circumcenters")
def mq_circumcenters(
    points_addr: Int,
    simplices_addr: Int,
    centers_addr: Int,
    count: Int,
    dim: Int,
    matrix_addr: Int,
    rhs_addr: Int,
) abi("C") -> Int:
    var points = fp(points_addr)
    var simplices = ip(simplices_addr)
    var centers = fp(centers_addr)
    var matrix = fp(matrix_addr)
    var rhs = fp(rhs_addr)
    for s in range(count):
        var base = Int(simplices[s * (dim + 1)])
        var base_norm2 = 0.0
        for k in range(dim):
            var value = points[base * dim + k]
            base_norm2 += value * value
        for row in range(dim):
            var vertex = Int(simplices[s * (dim + 1) + row + 1])
            var vertex_norm2 = 0.0
            for k in range(dim):
                var value = points[vertex * dim + k]
                vertex_norm2 += value * value
                matrix[row * dim + k] = 2.0 * (
                    value - points[base * dim + k]
                )
            rhs[row] = vertex_norm2 - base_norm2
        if not solve_linear(matrix, rhs, dim):
            return -(s + 1)
        for k in range(dim):
            centers[s * dim + k] = rhs[k]
    return count
