"""Tests for getisg.getis_ord_g (re-export of getsorg.getis_ord_g)."""

from morie.fn.getisg import getis_ord_g


def ring(n):
    return [[1.0 if abs(i - j) in (1, n - 1) else 0.0 for j in range(n)] for i in range(n)]


def test_getisg_basic():
    x = [float(v) for v in range(1, 11)]
    W = ring(10)
    result = getis_ord_g(x, W)
    num = sum(W[i][j] * x[i] * x[j] for i in range(10) for j in range(10) if i != j)
    den = sum(x[i] * x[j] for i in range(10) for j in range(10) if i != j)
    assert abs(result["estimate"] - num / den) < 1e-15
    assert result["expected"] == 20 / 90


def test_getisg_edge():
    result = getis_ord_g([1.0, 2.0, 3.0, 4.0], [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])
    assert abs(result["estimate"] - 40 / 70) < 1e-15
