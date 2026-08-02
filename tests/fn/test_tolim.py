"""Tests for tolim.tolerance_limits."""

from morie.fn import _array_core as np
import pytest

from morie.fn.tolim import tolerance_limits


def test_tolim_interval_lies_within_the_data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    r = tolerance_limits(x, coverage=0.9, confidence=0.95)
    assert x.min() <= float(r["lower"]) < float(r["upper"]) <= x.max()


def test_tolim_achieved_confidence_meets_the_request():
    rng = np.random.default_rng(1)
    r = tolerance_limits(rng.normal(size=500), coverage=0.9, confidence=0.95)
    assert float(r["confidence_achieved"]) >= 0.95


def test_tolim_confidence_follows_the_wilks_formula_exactly():
    """The interval is always [X_(1), X_(n)] by design; what varies with
    coverage is the ACHIEVED CONFIDENCE, via Wilks (1941):
    1 - n b^(n-1) + (n-1) b^n. Check it to machine precision and check
    it falls as the requested coverage rises."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=50)
    n = 50
    for b in (0.80, 0.90, 0.95):
        r = tolerance_limits(x, coverage=b)
        want = 1 - n * b ** (n - 1) + (n - 1) * b**n
        assert float(r["confidence_achieved"]) == pytest.approx(want, rel=1e-12)
    c80 = float(tolerance_limits(x, coverage=0.80)["confidence_achieved"])
    c99 = float(tolerance_limits(x, coverage=0.99)["confidence_achieved"])
    assert c80 > c99


def test_tolim_calibration_across_samples():
    """Wilks guarantee: with coverage 0.8 / confidence 0.9 the interval
    contains >= 80 percent of the TRUE distribution in at least ~90
    percent of repetitions. Measured 20/20 across seeds at n = 200."""
    from morie.fn import _stats_core as stats
    hits = 0
    for s in range(20):
        rng = np.random.default_rng(s)
        r = tolerance_limits(rng.normal(size=200), coverage=0.8, confidence=0.9)
        mass = stats.norm.cdf(float(r["upper"])) - stats.norm.cdf(float(r["lower"]))
        hits += mass >= 0.8
    assert hits >= 16
