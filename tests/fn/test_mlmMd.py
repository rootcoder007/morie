"""Tests for mlmMd.multilevel_mediation."""

from morie.fn import _array_core as np
import pytest

from morie.fn.mlmMd import multilevel_mediation


def test_mlmMd_basic():
    rng = np.random.default_rng(42)
    J, npc = 60, 40
    n = J * npc
    cl = np.repeat(np.arange(J), npc)
    xb = rng.normal(size=J)[cl]
    xw = rng.normal(size=n)
    x = xb + xw
    m = 1.0 * xb + 0.2 * xw + rng.normal(scale=0.3, size=n)
    mb = np.array([m[cl == j].mean() for j in range(J)])[cl]
    y = 1.0 * mb + 0.2 * (m - mb) + rng.normal(scale=0.3, size=n)
    out = multilevel_mediation(y, x, m, cl)
    assert out["indirect_between"] > out["indirect_within"]
    assert out["n_clusters"] == J


def test_mlmMd_edge():
    with pytest.raises(ValueError):
        multilevel_mediation(np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10))  # 1 cluster
