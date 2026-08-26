"""ctypes loader for the Mojo geometry kernels."""

from __future__ import annotations

import ctypes
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_QHULL_LIB", os.path.join(ROOT, "dist", "libmojo-qhull.so"))

I = ctypes.c_int64
D = ctypes.c_double

_SIGNATURES = {
    "mq_convex_hull": ([I] * 16, I),
    "mq_lift_delaunay": ([I] * 4, None),
    "mq_halfspace_dual": ([I] * 5, I),
    "mq_polar_vertices": ([I] * 5, I),
    "mq_compact_unique_rows": ([I] * 4 + [D], I),
    "mq_topology_ridges": ([I] * 7, I),
    "mq_filter_lower_facets": ([I] * 7 + [D], I),
    "mq_circumcenters": ([I] * 8, I),
}

_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        if not os.path.exists(LIB):
            raise RuntimeError("Mojo library is missing; run `pixi run build`")
        _library = ctypes.CDLL(LIB)
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def addr(array: np.ndarray) -> int:
    """Return an address only for an ABI-compatible, nonempty NumPy buffer."""
    if not isinstance(array, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if array.dtype not in (np.dtype(np.float64), np.dtype(np.int64)):
        raise TypeError("FFI buffers must have dtype float64 or int64")
    if not array.flags.c_contiguous or not array.flags.aligned:
        raise ValueError("FFI buffers must be C-contiguous and aligned")
    address = int(array.ctypes.data)
    if array.size == 0 or address == 0:
        raise ValueError("FFI buffers must be nonempty")
    return address


def f64(array) -> np.ndarray:
    """Make the API's float64 conversion explicit without discarding complex data."""
    source = np.asarray(array)
    if np.issubdtype(source.dtype, np.complexfloating):
        raise TypeError("complex-valued input is not supported")
    return np.ascontiguousarray(source, dtype=np.float64)
