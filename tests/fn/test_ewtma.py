"""ewtma: RiskMetrics EWMA volatility (J.P. Morgan/Reuters, RiskMetrics 1996).

    sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_{t-1}^2
"""

import numpy as np
import pytest

from morie.fn.ewtma import ewma_volatility as ew


def test_ewtma_follows_the_recursion_exactly():
    r = np.array([0.01, -0.02, 0.015, -0.005, 0.03])
    lam = 0.94
    got = np.asarray(ew(r, lambda_=lam)["conditional_variance"])
    # Replay the recursion independently, seeded the same way the module is.
    v = np.empty(got.size)
    v[0] = got[0]
    for t in range(1, got.size):
        v[t] = lam * v[t - 1] + (1 - lam) * r[t - 1] ** 2
    assert got == pytest.approx(v)


def test_ewtma_volatility_is_the_square_root_of_the_variance():
    rng = np.random.default_rng(2401)
    r = ew(rng.normal(0, 0.01, 200))
    assert np.asarray(r["conditional_volatility"]) == pytest.approx(
        np.sqrt(np.asarray(r["conditional_variance"]))
    )
    assert r["last_volatility"] == pytest.approx(np.sqrt(r["last_variance"]))


def test_ewtma_default_lambda_is_the_riskmetrics_value():
    """RiskMetrics fixes lambda = 0.94 for daily data. That constant is the
    whole reason the method is named after them."""
    assert ew(np.random.default_rng(2411).normal(0, 0.01, 50))["lambda"] == pytest.approx(0.94)


def test_ewtma_higher_lambda_gives_a_smoother_path():
    """lambda is the memory: closer to 1 means slower to react."""
    rng = np.random.default_rng(2417)
    r = np.concatenate([rng.normal(0, 0.005, 100), rng.normal(0, 0.05, 100)])
    rough = np.diff(np.asarray(ew(r, lambda_=0.80)["conditional_volatility"]))
    smooth = np.diff(np.asarray(ew(r, lambda_=0.99)["conditional_volatility"]))
    assert np.std(smooth) < np.std(rough)


def test_ewtma_reacts_to_a_volatility_regime_shift():
    """The reason for using EWMA at all: it must track a change in variance."""
    rng = np.random.default_rng(2423)
    r = np.concatenate([rng.normal(0, 0.002, 300), rng.normal(0, 0.05, 300)])
    vol = np.asarray(ew(r)["conditional_volatility"])
    assert vol[-1] > 10 * vol[250]


def test_ewtma_constant_returns_converge_to_that_squared_return():
    """With r_t = c forever the recursion has fixed point sigma^2 = c^2."""
    c = 0.02
    v = np.asarray(ew(np.full(500, c))["conditional_variance"])
    assert v[-1] == pytest.approx(c**2, rel=1e-6)


def test_ewtma_rejects_lambda_outside_the_unit_interval():
    with pytest.raises(ValueError):
        ew(np.zeros(10), lambda_=1.5)
