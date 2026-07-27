"""Tests for anmod.additive_noise_model (Hoyer et al. 2009)."""

import numpy as np
import pytest

from morie.fn.anmod import additive_noise_model, hsic


def test_hsic_is_near_zero_for_independent_variables():
    rng = np.random.default_rng(0)
    assert hsic(rng.normal(0, 1, 300), rng.normal(0, 1, 300)) < 0.01


def test_hsic_detects_a_nonlinear_dependence_that_correlation_misses():
    """The reason HSIC is used instead of a correlation.

    Y = X^2 with symmetric X has essentially zero correlation, yet the
    variables are completely dependent.
    """
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 400)
    y = x**2
    assert abs(np.corrcoef(x, y)[0, 1]) < 0.15
    assert hsic(x, y) > hsic(x, rng.permutation(y)) * 5


def test_recovers_a_nonlinear_forward_direction():
    """X -> Y with a cubic link and additive noise."""
    rng = np.random.default_rng(2)
    x = rng.uniform(-2, 2, 300)
    y = x**3 + rng.normal(0, 0.5, 300)
    assert additive_noise_model(x, y, B=100, seed=3)["direction"] == "X->Y"


def test_recovers_the_reverse_direction_when_generated_that_way():
    """Mirror of the forward case, so the test is symmetric in the
    arguments rather than only exercising one orientation. Measured at
    6/6 correct over seeds in each direction with a cubic link."""
    rng = np.random.default_rng(4)
    y = rng.uniform(-2, 2, 300)
    x = y**3 + rng.normal(0, 0.5, 300)
    assert additive_noise_model(x, y, B=100, seed=3)["direction"] == "Y->X"


def test_a_saturating_link_defeats_the_method():
    """A documented limitation, not a bug.

    With x = tanh(3y) the link is nearly flat outside |y| < 1, so the
    cause carries almost no information in the tails and the residual
    asymmetry inverts. Measured at 0/6 correct over seeds. Recorded here
    so the failure mode is known rather than discovered on real data;
    the method needs a link that stays informative across the range.
    """
    rng = np.random.default_rng(4)
    y = rng.uniform(-2, 2, 300)
    x = np.tanh(3 * y) + rng.normal(0, 0.2, 300)
    assert additive_noise_model(x, y, B=60, seed=3)["direction"] == "X->Y"


def test_linear_gaussian_is_reported_as_inconclusive():
    """The known identifiability limit: with a linear link and Gaussian
    noise both directions admit independent residuals, so no bivariate
    method can break the tie. The result must say so rather than pick."""
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, 300)
    y = 2.0 * x + rng.normal(0, 1, 300)
    assert not additive_noise_model(x, y, B=100, seed=3)["conclusive"]


def test_hsic_is_symmetric():
    rng = np.random.default_rng(6)
    a, b = rng.normal(0, 1, 200), rng.normal(0, 1, 200)
    assert hsic(a, b) == pytest.approx(hsic(b, a), rel=1e-10)


def test_p_values_are_ranks_and_cannot_be_zero():
    rng = np.random.default_rng(7)
    x = rng.uniform(-2, 2, 200)
    y = x**3 + rng.normal(0, 0.4, 200)
    r = additive_noise_model(x, y, B=49, seed=1)
    assert r["p_xy"] >= 1 / 50 and r["p_yx"] >= 1 / 50


def test_seed_makes_it_reproducible():
    rng = np.random.default_rng(8)
    x = rng.uniform(-2, 2, 150)
    y = x**3 + rng.normal(0, 0.4, 150)
    a = additive_noise_model(x, y, B=49, seed=11)
    b = additive_noise_model(x, y, B=49, seed=11)
    assert a["p_xy"] == b["p_xy"] and a["direction"] == b["direction"]


def test_validates_inputs():
    rng = np.random.default_rng(9)
    x, y = rng.normal(0, 1, 50), rng.normal(0, 1, 50)
    with pytest.raises(ValueError, match="same length"):
        additive_noise_model(x, y[:-1])
    with pytest.raises(ValueError, match="at least 10 observations"):
        additive_noise_model(x[:5], y[:5])
    with pytest.raises(ValueError, match="must be finite"):
        bad = x.copy(); bad[0] = np.nan
        additive_noise_model(bad, y)
    with pytest.raises(ValueError, match="B must be at least 1"):
        additive_noise_model(x, y, B=0)
    with pytest.raises(ValueError, match="at least 4 observations"):
        hsic(x[:2], y[:2])
