"""Nonparametric additive models (Horowitz Sec. 3.1)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.hrzmir import horowitz_marginal_integration
from morie.fn.hrzora import horowitz_two_step_oracle


def _additive(n=600, seed=0, d=2):
    """mu + m1 + m2 with m1 = sin(pi x), m2 = x^2 - 1/3, both centred."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, d))
    m1 = np.sin(np.pi * X[:, 0])
    m2 = X[:, 1] ** 2 - 1.0 / 3.0
    y = 2.0 + m1 + m2 + rng.standard_normal(n) * 0.2
    return X, y


def test_marginal_integration_recovers_a_component():
    X, y = _additive(500)
    out = horowitz_marginal_integration(X, y, j=0)
    truth = np.sin(np.pi * out["grid"])
    truth = truth - truth.mean()
    got = out["m_hat"] - np.nanmean(out["m_hat"])
    assert np.corrcoef(got[~np.isnan(got)], truth[~np.isnan(got)])[0, 1] > 0.9
    # mu = E(Y) exactly, by the location normalisation (3.7)
    assert out["mu_hat"] == pytest.approx(y.mean())
    # and the fitted component is centred, as (3.6) requires
    assert abs(out["mean_of_m_hat"]) < 0.25


def test_marginal_integration_reports_the_curse_it_carries():
    X, y = _additive(200, d=3)
    out = horowitz_marginal_integration(X, y, j=1)
    assert out["curse_of_dimensionality"] is True
    # Theorem 3.1(b) needs q > d - 1 derivatives: the requirement
    # GROWS with the dimension, which is the whole reason the chapter
    # develops the two-step estimator
    assert out["smoothness_required"] == 3
    assert horowitz_marginal_integration(
        *_additive(200, d=5), j=0)["smoothness_required"] == 5
    assert out["h1"] > 0 and out["h2"] > 0
    with pytest.raises(ValueError):
        horowitz_marginal_integration(X, y, j=9)
    with pytest.raises(ValueError):
        horowitz_marginal_integration(X[:, :1], y, j=0)


def test_two_step_recovers_both_components_at_once():
    X, y = _additive(800)
    out = horowitz_two_step_oracle(X, y)
    g = out["grid"]
    t1 = np.sin(np.pi * g); t1 -= t1.mean()
    t2 = g ** 2 - 1.0 / 3.0; t2 -= t2.mean()
    assert np.corrcoef(out["m_hat"][0], t1)[0, 1] > 0.95
    assert np.corrcoef(out["m_hat"][1], t2)[0, 1] > 0.95
    assert out["mu_hat"] == pytest.approx(2.0, abs=0.15)


def test_two_step_is_oracle_efficient_non_iterative_and_uncursed():
    X, y = _additive(400)
    out = horowitz_two_step_oracle(X, y)
    # the claims the section actually makes, kept as checkable keys
    assert out["oracle_efficient"] is True
    assert out["iterative"] is False          # unlike backfitting
    assert out["rate_exponent"] == pytest.approx(-0.4)
    assert out["max_smoothing_dimension"] == 1
    assert out["curse_of_dimensionality"] is False


def test_two_step_holds_its_rate_as_the_dimension_grows():
    # The claim that matters: adding components must not degrade the
    # accuracy of the component of interest. Marginal integration
    # would need ever more smoothness; this must not.
    def err(d):
        X, y = _additive(800, seed=3, d=d)
        out = horowitz_two_step_oracle(X, y)
        t = np.sin(np.pi * out["grid"]); t -= t.mean()
        return float(np.sqrt(np.mean((out["m_hat"][0] - t) ** 2)))
    e2, e5 = err(2), err(5)
    assert e5 < 3 * e2
    assert e5 < 0.5


def test_two_step_validates_and_exposes_its_tuning():
    X, y = _additive(200)
    out = horowitz_two_step_oracle(X, y, kappa=4, bandwidth=0.3)
    assert out["kappa"] == 4
    assert np.allclose(out["bandwidth"], 0.3)
    nw = horowitz_two_step_oracle(X, y, kappa=4, bandwidth=0.3,
                                  local_linear=False)
    assert nw["m_hat"].shape == out["m_hat"].shape
    with pytest.raises(ValueError):
        horowitz_two_step_oracle(X, y, kappa=1)
    with pytest.raises(ValueError):
        horowitz_two_step_oracle(X, y, bandwidth=-1.0)
    with pytest.raises(ValueError):
        horowitz_two_step_oracle(X[:10], y[:10])
