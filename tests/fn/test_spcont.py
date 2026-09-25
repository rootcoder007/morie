"""Tests for spcont.schabenberger_spatial_continuity."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.spcont import schabenberger_spatial_continuity


def _exp(h):
    return np.exp(-3.0 * np.asarray(h, dtype=float))


def _nugget(h):
    """Nugget 1 plus exponential sill 2: C(0) = 3, C(0+) = 2."""
    h = np.asarray(h, dtype=float)
    return np.where(h == 0.0, 3.0, 2.0 * np.exp(-h))


def test_spcont_basic():
    """A continuous exponential covariance is MS continuous with no
    nugget; the gaps are C(0) - C(h) at h = 1e-2..1e-6."""
    r = schabenberger_spatial_continuity(_exp)
    assert r["is_continuous"] is True
    assert r["nugget"] == 0.0 and r["gamma_limit"] == 0.0
    assert r["c0"] == 1.0
    for h, g in zip([1e-2, 1e-3, 1e-4, 1e-5, 1e-6], r["gaps"]):
        assert float(g) == pytest.approx(1.0 - math.exp(-3.0 * h), rel=1e-12)
    assert r["gap_ratio"] == pytest.approx((1 - math.exp(-3e-6)) / (1 - math.exp(-3e-2)), rel=1e-9)


def test_spcont_edge():
    """A nugget leaves a plateau: not continuous, and the reported gap is
    C(0) - C(1e-6) = 3 - 2 exp(-1e-6), twice that being the limit of
    E[(Z(s) - Z(s+h))^2]; a non-callable raises."""
    r = schabenberger_spatial_continuity(_nugget)
    g = 3.0 - 2.0 * math.exp(-1e-6)
    assert r["is_continuous"] is False
    assert r["gap"] == pytest.approx(g, rel=1e-12)
    assert r["nugget"] == pytest.approx(g, rel=1e-12)
    assert r["gamma_limit"] == pytest.approx(2 * g, rel=1e-12)
    with pytest.raises(TypeError):
        schabenberger_spatial_continuity(1.0)
