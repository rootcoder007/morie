"""Tests for otsktol.ot_sinkhorn_tol."""

import math

from morie.fn import _array_core as np

from morie.fn.otsktol import ot_sinkhorn_tol


def test_otsktol_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    nr, nc = 5, 4
    # Coupling matrix T must be a 2-D object with len(a) rows and len(b) columns
    T = rng.uniform(0, 1, (nr, nc))
    # Target marginals (the function internally closes them to unit mass)
    a = rng.uniform(0, 1, nr)
    b = rng.uniform(0, 1, nc)

    result = ot_sinkhorn_tol(T, a, b)
    assert isinstance(result, dict)
    for key in ("estimate", "row_error", "col_error", "nrow", "ncol", "method"):
        assert key in result
    assert result["nrow"] == nr
    assert result["ncol"] == nc
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["row_error"])
    assert math.isfinite(result["col_error"])
    assert result["estimate"] >= 0
    assert result["row_error"] >= 0
    assert result["col_error"] >= 0
    assert result["estimate"] >= result["row_error"]
    assert result["estimate"] >= result["col_error"]


def test_otsktol_edge():
    """Test edge cases: matching marginals yield zero violation."""
    nr, nc = 3, 4
    # Use values exactly representable in binary floating point
    a = [0.5, 0.25, 0.25]
    b = [0.5, 0.25, 0.125, 0.125]
    # Outer product: row sums equal a and column sums equal b
    T = [[a[i] * b[j] for j in range(nc)] for i in range(nr)]

    result = ot_sinkhorn_tol(T, a, b)
    assert isinstance(result, dict)
    for key in ("estimate", "row_error", "col_error", "nrow", "ncol", "method"):
        assert key in result
    assert result["nrow"] == nr
    assert result["ncol"] == nc
    assert math.isfinite(result["estimate"])
    assert math.isclose(result["estimate"], 0, abs_tol=1e-12)
    assert math.isclose(result["row_error"], 0, abs_tol=1e-12)
    assert math.isclose(result["col_error"], 0, abs_tol=1e-12)
