"""Tests for spackf.schabenberger_autocorrelation_function."""

import math

import pytest

from morie.fn.spackf import schabenberger_autocorrelation_function


def _grid():
    coords = [[float(i), float(j)] for i in range(4) for j in range(3)]
    z = [math.sin(0.9 * x) + 0.5 * math.cos(1.3 * y) + 0.1 * k for k, (x, y) in enumerate(coords)]
    return coords, z


def test_spackf_basic():
    """C(0) = mean squared deviation; each lag class averages the
    deviation cross-products of its pairs (i < j, lo < h <= edge);
    R(h) = C(h)/C(0)."""
    coords, z = _grid()
    n = len(z)
    zb = sum(z) / n
    d = [t - zb for t in z]
    c0 = sum(t * t for t in d) / n
    pairs = [(math.dist(coords[i], coords[j]), d[i] * d[j]) for i in range(n) for j in range(i + 1, n)]
    edges = [1.0, 1.5, 2.5, 4.0]
    r = schabenberger_autocorrelation_function(coords, z, bins=edges)
    lo = 0.0
    for k, e in enumerate(edges):
        cls = [p for h, p in pairs if lo < h <= e]
        assert r["npairs"][k] == len(cls)
        assert r["cov"][k] == pytest.approx(sum(cls) / len(cls), abs=1e-12)
        assert r["acf"][k] == pytest.approx(sum(cls) / len(cls) / c0, abs=1e-12)
        assert r["centres"][k] == pytest.approx(0.5 * (lo + e), abs=1e-15)
        lo = e
    assert r["c0"] == pytest.approx(c0, abs=1e-12)
    # the default is ten equal classes up to the largest separation
    rd = schabenberger_autocorrelation_function(coords, z)
    hmax = max(h for h, _ in pairs)
    assert rd["lags"] == pytest.approx([hmax * (k + 1) / 10 for k in range(10)], abs=1e-12)
    assert sum(rd["npairs"]) == n * (n - 1) // 2


def test_spackf_edge():
    """Two sites, a constant field, bad bins and mismatched lengths raise;
    an empty class reports nan."""
    coords, z = _grid()
    with pytest.raises(ValueError):
        schabenberger_autocorrelation_function([[0.0, 0.0], [1.0, 0.0]], [3.0, 4.0])
    with pytest.raises(ValueError):
        schabenberger_autocorrelation_function(coords, [1.0] * len(z))
    with pytest.raises(ValueError):
        schabenberger_autocorrelation_function(coords, z, bins=[2.0, 1.0])
    with pytest.raises(ValueError):
        schabenberger_autocorrelation_function(coords, z[:-1])
    r = schabenberger_autocorrelation_function(coords, z, bins=[0.5, 1.0])
    assert r["npairs"][0] == 0 and math.isnan(r["acf"][0])
