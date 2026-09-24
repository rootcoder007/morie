"""Tests for otfgw.ot_fused_gromov_wasserstein."""

import math

from morie.fn import _array_core as np

from morie.fn.otfgw import ot_fused_gromov_wasserstein


def test_otfgw_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, m = 10, 10
    M = rng.normal(0, 1, (n, m))

    Cx_raw = rng.normal(0, 1, (n, n))
    Cx = [[(Cx_raw[i][j] + Cx_raw[j][i]) / 2.0 for j in range(n)]
          for i in range(n)]

    Cy_raw = rng.normal(0, 1, (m, m))
    Cy = [[(Cy_raw[i][j] + Cy_raw[j][i]) / 2.0 for j in range(m)]
          for i in range(m)]

    a_raw = rng.uniform(0, 1, n)
    sa = sum(a_raw)
    a = [x / sa for x in a_raw]

    b_raw = rng.uniform(0, 1, m)
    sb = sum(b_raw)
    b = [x / sb for x in b_raw]

    result = ot_fused_gromov_wasserstein(M, Cx, Cy, a, b, alpha=0.5, max_iter=20)

    assert isinstance(result, dict)
    assert result["n"] == n
    assert result["m"] == m
    assert result["iters"] == 20
    assert "T" in result
    assert "cost" in result
    assert "wass_part" in result
    assert "gromov_part" in result
    assert "method" in result
    assert math.isfinite(result["cost"])
    assert len(result["T"]) == n
    assert all(len(row) == m for row in result["T"])


def test_otfgw_edge():
    """Test edge case: rectangular input with alpha at boundary."""
    rng = np.random.default_rng(7)
    n, m = 6, 8

    M = rng.normal(0, 1, (n, m))

    Cx_raw = rng.normal(0, 1, (n, n))
    Cx = [[(Cx_raw[i][j] + Cx_raw[j][i]) / 2.0 for j in range(n)]
          for i in range(n)]

    Cy_raw = rng.normal(0, 1, (m, m))
    Cy = [[(Cy_raw[i][j] + Cy_raw[j][i]) / 2.0 for j in range(m)]
          for i in range(m)]

    a_raw = rng.uniform(0, 1, n)
    sa = sum(a_raw)
    a = [x / sa for x in a_raw]

    b_raw = rng.uniform(0, 1, m)
    sb = sum(b_raw)
    b = [x / sb for x in b_raw]

    # alpha = 0 reduces to pure Wasserstein transport
    result = ot_fused_gromov_wasserstein(M, Cx, Cy, a, b, alpha=0.0, max_iter=5)

    assert isinstance(result, dict)
    assert result["n"] == n
    assert result["m"] == m
    assert result["iters"] == 5
    assert math.isfinite(result["cost"])
    assert len(result["T"]) == n
    assert all(len(row) == m for row in result["T"])
