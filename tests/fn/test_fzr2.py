"""Tests for fzr2.fauzi_r2_integral."""

from morie.fn import _array_core as np
from morie.fn import _stats_core as stats

from morie.fn.fzr2 import fauzi_r2_integral


def _r2_manual(a, lo=-8.0, hi=8.0, ngrid=4001):
    """Independent reference computation of r_2 from Eq. (2.10).

    r_2 = ∫ y [ K(y) W(y/a) + (1/a) W(y) K(y/a) ] dy
    """
    a = float(a)
    y = np.linspace(float(lo), float(hi), int(ngrid))
    kfun = lambda t: stats.norm.pdf(t)
    wfun = lambda t: stats.norm.cdf(t)
    term = kfun(y) * wfun(y / a) + (1.0 / a) * wfun(y) * kfun(y / a)
    return float(np.trapezoid(y * term, y))


def test_fzr2_basic():
    """Test basic functionality with the documented signature."""
    a = 0.5
    result = fauzi_r2_integral(a)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert result["a"] == a
    assert result["method"] == "r_2 cross-kernel constant (Eq. 2.10)"
    expected = _r2_manual(a)
    assert abs(result["estimate"] - expected) < 1e-4


def test_fzr2_edge():
    """Test edge cases: a -> 1+ and a -> 1- should both diverge."""
    a_below = 0.99
    a_above = 1.01
    r_below = fauzi_r2_integral(a_below)["estimate"]
    r_above = fauzi_r2_integral(a_above)["estimate"]
    expected_below = _r2_manual(a_below)
    expected_above = _r2_manual(a_above)
    assert abs(r_below - expected_below) < 1e-4
    assert abs(r_above - expected_above) < 1e-4
