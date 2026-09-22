"""Tests for fzbfkf.fauzi_bdfree_kdfe_test."""

from math import log

from morie.fn import _array_core as np

from morie.fn.fzbfkf import fauzi_bdfree_kdfe_test


def test_fzbfkf_basic():
    """Test basic functionality against Eq. (5.5)."""
    rng = np.random.default_rng(42)
    x = rng.exponential(1.0, 100)
    h = 0.3
    result = fauzi_bdfree_kdfe_test(x, grid=None, ginv=log, h=h)
    assert isinstance(result, dict)
    for key in ("estimate", "grid", "h", "n", "method"):
        assert key in result
    assert result["n"] == 100
    assert result["h"] == h
    assert len(result["estimate"]) == 100
    assert len(result["grid"]) == 100

    xv = np.asarray(x, dtype=float).ravel()
    ys = [log(float(t)) for t in xv]
    mean_y = sum(ys) / len(ys)
    var_y = sum((v - mean_y) ** 2 for v in ys) / len(ys)
    sd_y = var_y ** 0.5
    sorted_x = sorted(float(t) for t in xv)
    from statistics import NormalDist
    nd = NormalDist(0.0, 1.0)
    expected = [
        sum(nd.cdf((log(float(t)) - yi) / h) for yi in ys) / len(ys)
        for t in sorted_x
    ]
    got = [float(v) for v in result["estimate"]]
    for g_val, e_val in zip(got, expected):
        assert abs(g_val - e_val) < 1e-9


def test_fzbfkf_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    x = rng.exponential(1.0, 50)
    h = 0.3
    result = fauzi_bdfree_kdfe_test(x, grid=None, ginv=log, h=h)
    assert isinstance(result, dict)
    for key in ("estimate", "grid", "h", "n", "method"):
        assert key in result
    assert result["n"] == 50
    assert result["h"] == h
    for v in result["estimate"]:
        fv = float(v)
        assert 0.0 <= fv <= 1.0
