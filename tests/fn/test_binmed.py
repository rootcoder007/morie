"""Tests for binmed.binary_outcome_mediation (Tchetgen Tchetgen 2013)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.binmed import binary_outcome_mediation


def _binary(seed=0, n=4000, a=1.0, b=1.0, direct=0.5):
    """X -> M -> Y with a binary outcome; a and b set the mediated path."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, n).astype(float)
    m = a * x + rng.normal(0, 1, n)
    eta = -0.5 + direct * x + b * m
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
    return x, m, y


def test_total_decomposes_into_direct_plus_indirect():
    """By construction: indirect is defined as total minus direct."""
    x, m, y = _binary(seed=1)
    r = binary_outcome_mediation(x, m, y)
    assert r["total"] == pytest.approx(r["direct"] + r["indirect"], rel=1e-12)


def test_odds_ratios_are_the_exponentiated_log_odds():
    x, m, y = _binary(seed=2)
    r = binary_outcome_mediation(x, m, y)
    assert r["or_total"] == pytest.approx(np.exp(r["total"]), rel=1e-12)
    assert r["or_indirect"] == pytest.approx(np.exp(r["indirect"]), rel=1e-12)


def test_a_real_mediated_path_gives_a_positive_indirect_effect():
    x, m, y = _binary(seed=3, a=1.2, b=1.2, direct=0.3)
    assert binary_outcome_mediation(x, m, y)["indirect"] > 0.1


def test_no_mediated_path_gives_a_near_zero_indirect_effect():
    """M unrelated to X, so nothing can flow through it."""
    rng = np.random.default_rng(4)
    n = 4000
    x = rng.integers(0, 2, n).astype(float)
    m = rng.normal(0, 1, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-0.5 + 0.8 * x + 0.8 * m)))).astype(float)
    assert abs(binary_outcome_mediation(x, m, y)["indirect"]) < 0.15


def test_stronger_mediation_gives_a_larger_indirect_effect():
    weak = binary_outcome_mediation(*_binary(seed=5, a=0.3, b=0.3))["indirect"]
    strong = binary_outcome_mediation(*_binary(seed=5, a=1.5, b=1.5))["indirect"]
    assert strong > weak


def test_no_standard_error_unless_bootstrapped():
    """The weights are estimated, so a model-based SE would understate."""
    x, m, y = _binary(seed=6)
    assert binary_outcome_mediation(x, m, y)["se"] is None
    r = binary_outcome_mediation(x, m, y, B=40, seed=1)
    assert set(r["se"]) == {"total", "direct", "indirect"}
    assert all(v > 0 for v in r["se"].values())


def test_bootstrap_interval_brackets_the_point_estimate():
    x, m, y = _binary(seed=7, a=1.2, b=1.2)
    r = binary_outcome_mediation(x, m, y, B=60, seed=2)
    assert r["ci_low"]["indirect"] < r["indirect"] < r["ci_high"]["indirect"]


def test_covariates_are_accepted():
    rng = np.random.default_rng(8)
    x, m, y = _binary(seed=8, n=2000)
    c = rng.normal(0, 1, (2000, 2))
    assert np.isfinite(binary_outcome_mediation(x, m, y, C=c)["indirect"])


def test_validates_inputs():
    x, m, y = _binary(seed=9, n=400)
    with pytest.raises(ValueError, match="share a length"):
        binary_outcome_mediation(x, m[:-1], y)
    with pytest.raises(ValueError, match="X must be binary"):
        binary_outcome_mediation(m, m, y)
    with pytest.raises(ValueError, match="Y must be binary"):
        binary_outcome_mediation(x, m, m)
    with pytest.raises(ValueError, match="at least 2 units per exposure arm"):
        binary_outcome_mediation(np.zeros_like(x), m, y)
    with pytest.raises(ValueError, match="must be finite"):
        bad = m.copy(); bad[0] = np.nan
        binary_outcome_mediation(x, bad, y)
