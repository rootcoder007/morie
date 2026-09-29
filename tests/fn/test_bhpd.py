"""Tests for morie.fn.bhpd -- HPD interval."""

from morie.fn import _array_core as np
from morie.fn.bhpd import hpd_interval


def test_returns_dict():
    result = hpd_interval([1, 2, 3, 4, 5])
    assert isinstance(result, dict)
    assert "hpd_lower" in result
    assert "hpd_upper" in result


def test_symmetric_distribution():
    rng = np.random.default_rng(42)
    samples = rng.normal(0, 1, 10000)
    result = hpd_interval(samples, prob=0.95)
    assert abs(result["hpd_lower"] + result["hpd_upper"]) < 0.3


def test_hpd_shorter_than_equal_tailed():
    rng = np.random.default_rng(42)
    samples = rng.lognormal(0, 1, 10000)
    hpd = hpd_interval(samples, prob=0.95)
    hpd_width = hpd["width"]
    q_lo = np.percentile(samples, 2.5)
    q_hi = np.percentile(samples, 97.5)
    et_width = q_hi - q_lo
    assert hpd_width <= et_width + 0.1


def test_width_positive():
    rng = np.random.default_rng(42)
    result = hpd_interval(rng.normal(0, 1, 100))
    assert result["width"] > 0


def test_too_short():
    try:
        hpd_interval([1])
        assert False
    except ValueError:
        pass


def test_invalid_prob():
    try:
        hpd_interval([1, 2, 3], prob=0)
        assert False
    except ValueError:
        pass


def test_hpd_is_the_shortest_window_recomputed():
    import pytest

    x = [3.2, 0.1, 0.4, 0.2, 0.9, 5.5, 0.35, 0.6, 1.4, 0.05, 2.2, 0.8]
    s = sorted(x)
    size = int(0.5 * len(s))
    widths = [s[i + size] - s[i] for i in range(len(s) - size)]
    i = widths.index(min(widths))
    r = hpd_interval(x, prob=0.5)
    assert (r["hpd_lower"], r["hpd_upper"]) == pytest.approx((s[i], s[i + size]), rel=1e-15)
    assert r["width"] == pytest.approx(widths[i], rel=1e-14)
