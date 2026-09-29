"""Tests for gb_t1e.gibbons_type1_error."""

from morie.fn import _array_core as np
from morie.fn.gb_t1e import gibbons_type1_error


def test_gb_t1e_basic():
    """Test basic functionality."""
    pmf = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_type1_error(pmf)
    assert isinstance(result, dict)
    assert "sizes" in result


def test_gb_t1e_edge():
    """Test edge cases."""
    pmf = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_type1_error(pmf)
    assert isinstance(result, dict)


def test_attainable_sizes_for_binomial_five_half():
    """Gibbons-Chakraborti Sec 1.2.9: Bin(5, 1/2), upper tail."""
    import math

    import pytest

    pmf = [math.comb(5, k) / 32 for k in range(6)]
    upper = [sum(pmf[k:]) for k in range(6)]
    r = gibbons_type1_error(pmf, alpha=0.2, upper=True)
    assert r["sizes"] == pytest.approx(upper, rel=1e-14)
    assert r["alpha_exact"] == pytest.approx(6 / 32, rel=1e-14)
    assert r["cut"] == 4


def test_lower_tail_and_no_attainable_size():
    import math

    pmf = [math.comb(5, k) / 32 for k in range(6)]
    lo = gibbons_type1_error(pmf, alpha=0.2, upper=False)
    assert lo["alpha_exact"] == 6 / 32 and lo["cut"] == 1
    none = gibbons_type1_error(pmf, alpha=0.01)
    assert math.isnan(none["alpha_exact"]) and none["cut"] == -1
