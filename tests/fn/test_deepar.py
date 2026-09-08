"""Tests for deepar. Full anchor: wave3/anchor_intermittent.py."""
import math
import pytest
from morie.fn import _array_core as np
from morie.fn import _s03core as k
from morie.fn.deepar import (deepar_fit, deepar_forecast,
                             negative_binomial_loglik,
                             sample_negative_binomial, scale_factor)


def test_the_scale_factor_protects_an_all_zero_series():
    """The +1 is not decoration -- intermittent series are often all
    zeros over a window."""
    assert scale_factor([0.0] * 10) == pytest.approx(1.0)
    assert scale_factor([1.0, 3.0]) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        scale_factor([], t0=0)


def test_the_negative_binomial_has_variance_mu_one_plus_mu_alpha():
    rng = np.random.default_rng(17)
    mu, alpha = 4.0, 0.5
    draws = [sample_negative_binomial(mu, alpha, rng)
             for _ in range(15000)]
    assert k.mean(draws) == pytest.approx(mu, abs=0.2)
    want = mu * (1.0 + mu * alpha)
    assert k.variance(draws) == pytest.approx(want, rel=0.15)


def test_alpha_to_zero_is_poisson():
    rng = np.random.default_rng(3)
    pois = [sample_negative_binomial(4.0, 0.0, rng)
            for _ in range(15000)]
    assert abs(k.variance(pois) - k.mean(pois)) < 0.25
    got = negative_binomial_loglik(3.0, 4.0, 1e-12)
    want = 3.0 * math.log(4.0) - 4.0 - math.log(6.0)
    assert got == pytest.approx(want, abs=1e-6)


def test_the_forecast_quantiles_are_ordered_and_widen():
    rng = np.random.default_rng(9)
    cnt = [max(0.0, math.floor(3.0 + 2.0 * rng.standard_normal()))
           for _ in range(120)]
    r = deepar_forecast(cnt, 6, n_samples=300, seed=1)
    for h in range(6):
        assert (r["quantiles"][0.1][h] <= r["quantiles"][0.5][h]
                <= r["quantiles"][0.9][h])
    assert r["width"][-1] >= r["width"][0] - 1e-9
    assert r["nu"] == pytest.approx(1.0 + k.mean(cnt))


def test_argument_checks():
    rng = np.random.default_rng(9)
    cnt = [max(0.0, math.floor(3.0 + 2.0 * rng.standard_normal()))
           for _ in range(60)]
    with pytest.raises(ValueError):
        negative_binomial_loglik(-1.0, 2.0, 0.5)
    with pytest.raises(ValueError):
        negative_binomial_loglik(1.0, 2.0, -0.5)
    with pytest.raises(ValueError):
        deepar_forecast(cnt, 4, quantiles=(1.5,))
    with pytest.raises(ValueError):
        deepar_fit(cnt, likelihood="nope")
    with pytest.raises(ValueError):
        deepar_fit([1.0, 2.0], n_lags=2)
