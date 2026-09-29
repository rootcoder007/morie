"""Tests for morie.fn.bcred -- Bayesian credible interval."""

from morie.fn import _array_core as np
from morie.fn.bcred import credible_interval


def test_returns_dict():
    result = credible_interval([1, 2, 3, 4, 5])
    assert isinstance(result, dict)
    assert "ci_lower" in result
    assert "ci_upper" in result


def test_95_ci():
    rng = np.random.default_rng(42)
    samples = rng.normal(0, 1, 10000)
    result = credible_interval(samples, prob=0.95)
    assert abs(result["ci_lower"] - (-1.96)) < 0.2
    assert np.all(np.isfinite(np.asarray(result["ci_upper"], dtype=float)))  # N6: was a generator-guessed value


def test_narrower_with_less_prob():
    rng = np.random.default_rng(42)
    samples = rng.normal(0, 1, 5000)
    ci95 = credible_interval(samples, prob=0.95)
    ci50 = credible_interval(samples, prob=0.50)
    w95 = ci95["ci_upper"] - ci95["ci_lower"]
    w50 = ci50["ci_upper"] - ci50["ci_lower"]
    assert w50 < w95


def test_ci_contains_median():
    rng = np.random.default_rng(42)
    samples = rng.normal(5, 2, 1000)
    result = credible_interval(samples)
    assert result["ci_lower"] < result["median"] < result["ci_upper"]


def test_empty():
    try:
        credible_interval([])
        assert False
    except ValueError:
        pass


def test_invalid_prob():
    try:
        credible_interval([1, 2, 3], prob=1.5)
        assert False
    except ValueError:
        pass


def test_equal_tailed_interval_recomputed():
    """Type-7 quantiles at (1 - prob)/2 and (1 + prob)/2."""
    import math

    import pytest

    x = [0.3, -1.2, 0.8, 2.0, 0.1, -0.4, 1.3, 0.6, -0.9, 0.2]

    def q(p):
        s = sorted(x)
        h = (len(s) - 1) * p
        lo = int(h)
        return s[lo] + (h - lo) * (s[min(lo + 1, len(s) - 1)] - s[lo])

    r = credible_interval(x, prob=0.8)
    assert r["ci_lower"] == pytest.approx(q(0.1), rel=1e-13)
    assert r["ci_upper"] == pytest.approx(q(0.9), rel=1e-13)
    m = sum(x) / 10
    assert r["mean"] == pytest.approx(m, rel=1e-14)
    assert r["sd"] == pytest.approx(math.sqrt(sum((v - m) ** 2 for v in x) / 9), rel=1e-13)
    assert r["median"] == pytest.approx(q(0.5), rel=1e-14)
