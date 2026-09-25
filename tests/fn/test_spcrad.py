"""Tests for spcrad.spectral_radius."""

import math

import pytest

from morie.fn.spcrad import spectral_radius


def _path(n):
    return [[1.0 if abs(i - j) == 1 else 0.0 for j in range(n)] for i in range(n)]


def test_spcrad_basic():
    """A path on n nodes has eigenvalues 2 cos(k pi/(n+1)), so its
    spectral radius is 2 cos(pi/(n+1)) and the Perron vector is
    sin(j pi/(n+1)) normalised.  The path is bipartite -- -rho is also an
    eigenvalue -- which is the case plain power iteration gets wrong."""
    for n in (4, 7):
        r = spectral_radius(_path(n))
        rho = 2 * math.cos(math.pi / (n + 1))
        assert r["rho"] == pytest.approx(rho, rel=1e-12)
        assert r["dominant_eigenvalue"] == pytest.approx(rho, rel=1e-12)
        assert r["sar_rho_bound"] == pytest.approx(1 / rho, rel=1e-12)
        v = [math.sin((j + 1) * math.pi / (n + 1)) for j in range(n)]
        s = math.sqrt(sum(t * t for t in v))
        assert r["eigenvector"] == pytest.approx([t / s for t in v], abs=1e-9)
    # a 3x3 rook grid: rho = 2 cos(pi/4) + 2 cos(pi/4) = 2 sqrt 2
    G = [[1.0 if abs(i % 3 - j % 3) + abs(i // 3 - j // 3) == 1 else 0.0 for j in range(9)] for i in range(9)]
    assert spectral_radius(G)["rho"] == pytest.approx(2 * math.sqrt(2), rel=1e-12)


def test_spcrad_edge():
    """A matrix whose dominant eigenvalue is negative reports it with its
    sign; asymmetric, zero and 1x1 inputs raise."""
    # [[0,1],[1,0]] - 2I has eigenvalues -1 and -3
    r = spectral_radius([[-2.0, 1.0], [1.0, -2.0]])
    assert r["rho"] == pytest.approx(3.0, rel=1e-12)
    assert r["dominant_eigenvalue"] == pytest.approx(-3.0, rel=1e-12)
    assert r["eigenvector"] == pytest.approx([1 / math.sqrt(2), -1 / math.sqrt(2)], abs=1e-12)
    with pytest.raises(ValueError):
        spectral_radius([[0.0, 1.0], [0.0, 0.0]])
    with pytest.raises(ValueError):
        spectral_radius([[0.0, 0.0], [0.0, 0.0]])
    with pytest.raises(ValueError):
        spectral_radius([[1.0]])
