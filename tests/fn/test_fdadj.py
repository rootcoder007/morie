"""Tests for fdadj.frontdoor_adjustment."""

import numpy as np
import pytest

from morie.fn.fdadj import frontdoor_adjustment


def _confounded(seed=42, n=8000):
    """X <- U -> Y with X -> Z -> Y: the front-door path is identified."""
    rng = np.random.default_rng(seed)
    u = (rng.random(n) < 0.5).astype(int)
    x = (rng.random(n) < 0.2 + 0.6 * u).astype(int)
    z = (rng.random(n) < 0.1 + 0.8 * x).astype(int)
    y = (rng.random(n) < 0.1 + 0.5 * z + 0.3 * u).astype(int)
    return x, z, y


def test_fdadj_basic():
    x, z, y = _confounded()
    out = frontdoor_adjustment(x, z, y)
    dist = out["distribution"]
    # true P(Y=1|do(x)) contrast = 0.8 * 0.5 = 0.4
    contrast = dist[1][1] - dist[0][1]
    assert contrast == pytest.approx(0.4, abs=0.06)
    assert out["incomplete_cells"] == []


def test_fdadj_edge():
    x, z, y = _confounded()
    # each do(x) row is a proper distribution
    for row in frontdoor_adjustment(x, z, y)["distribution"].values():
        assert sum(row.values()) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        frontdoor_adjustment(x[:10], z, y)  # length mismatch
