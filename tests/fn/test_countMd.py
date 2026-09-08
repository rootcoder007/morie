"""Tests for countMd.count_mediation."""

from morie.fn import _array_core as np
import pytest

from morie.fn.countMd import count_mediation


def test_countMd_basic():
    rng = np.random.default_rng(42)
    n = 6000
    x = rng.normal(size=n)
    m = 0.5 * x + rng.normal(scale=0.5, size=n)
    y = rng.poisson(np.exp(0.3 * x + 0.4 * m))
    out = count_mediation(y, x, m)
    assert out["coefficients"]["theta1"] == pytest.approx(0.3, abs=0.06)
    assert out["rr_total"] == pytest.approx(out["rr_nde"] * out["rr_nie"])


def test_countMd_edge():
    with pytest.raises(ValueError):
        count_mediation([-1.0] * 20, np.zeros(20), np.zeros(20))  # negative counts
    with pytest.raises(ValueError):
        count_mediation([1.0] * 5, np.zeros(5), np.zeros(5))  # too few obs
