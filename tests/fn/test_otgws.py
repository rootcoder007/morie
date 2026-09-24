"""Tests for otgws.ot_gromov_sinkhorn."""

import math

from morie.fn import _array_core as np
from morie.fn.otgws import ot_gromov_sinkhorn


def test_otgws_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 10
    m = 12
    pts_x = rng.normal(0, 1, (n, 2))
    Cx = [[math.sqrt(sum((pts_x[i][k] - pts_x[j][k]) ** 2 for k in range(2)))
           for j in range(n)] for i in range(n)]
    pts_y = rng.normal(0, 1, (m, 2))
    Cy = [[math.sqrt(sum((pts_y[i][k] - pts_y[j][k]) ** 2 for k in range(2)))
           for j in range(m)] for i in range(m)]

    a = np.abs(rng.normal(0, 1, n))
    b = np.abs(rng.normal(0, 1, m))

    result = ot_gromov_sinkhorn(Cx, Cy, a, b, epsilon=0.01, max_iter=5)
    assert isinstance(result, dict)
    assert "T" in result
    assert "cost" in result
    assert "GW" in result
    assert "n" in result
    assert "m" in result
    assert "iters" in result
    assert result["n"] == n
    assert result["m"] == m
    assert math.isfinite(result["cost"])
    assert math.isfinite(result["GW"])
    assert result["GW"] >= 0.0
    assert result["iters"] == 5
    assert len(result["T"]) == n
    assert len(result["T"][0]) == m


def test_otgws_edge():
    """Test edge cases."""
    rng = np.random.default_rng(43)
    n = 6
    m = 8
    pts_x = rng.normal(0, 1, (n, 2))
    Cx = [[math.sqrt(sum((pts_x[i][k] - pts_x[j][k]) ** 2 for k in range(2)))
           for j in range(n)] for i in range(n)]
    pts_y = rng.normal(0, 1, (m, 2))
    Cy = [[math.sqrt(sum((pts_y[i][k] - pts_y[j][k]) ** 2 for k in range(2)))
           for j in range(m)] for i in range(m)]

    a = np.abs(rng.normal(0, 1, n))
    b = np.abs(rng.normal(0, 1, m))

    result = ot_gromov_sinkhorn(Cx, Cy, a, b, epsilon=0.05, max_iter=3, inner_iter=50)
    assert isinstance(result, dict)
    assert "cost" in result
    assert math.isfinite(result["cost"])
    assert result["cost"] >= 0.0
    assert result["GW"] >= 0.0
    assert math.isfinite(result["GW"])
    assert result["n"] == n
    assert result["m"] == m
