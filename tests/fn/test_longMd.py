"""Tests for longMd.longitudinal_mediation."""

from morie.fn import _array_core as np
import pytest

from morie.fn.longMd import longitudinal_mediation


def test_longMd_basic():
    rng = np.random.default_rng(42)
    n = 3000
    x1 = rng.normal(size=n)
    m1 = rng.normal(size=n)
    y1 = rng.normal(size=n)
    m2 = 0.5 * x1 + 0.4 * m1 + rng.normal(scale=0.5, size=n)
    y2 = 0.3 * m1 + 0.4 * y1 + rng.normal(scale=0.5, size=n)
    y3 = 0.9 * m2 + 0.1 * x1 + 0.4 * y2 + rng.normal(scale=0.5, size=n)
    out = longitudinal_mediation(np.c_[x1, x1, x1], np.c_[m1, m2, m2], np.c_[y1, y2, y3])
    assert out["a"] == pytest.approx(0.5, abs=0.06)
    assert out["b"] == pytest.approx(0.9, abs=0.06)


def test_longMd_edge():
    z = np.zeros((20, 2))
    with pytest.raises(ValueError):
        longitudinal_mediation(z, z, z)  # only 2 waves
