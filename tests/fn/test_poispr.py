"""Tests for poispr.poisson_predictive (Gamma-Poisson conjugacy)."""

import pytest

from morie.fn.poispr import poisson_predictive


def test_poispr_basic():
    """alpha' = alpha + sum y, beta' = beta + sum e; the predictive for
    one unit of exposure is negative binomial with mean alpha'/beta' and
    variance alpha'/beta' (1 + 1/beta'), recomputed."""
    y, e = [2, 0, 5, 3, 1], [1.0, 0.5, 2.0, 1.5, 1.0]
    r = poisson_predictive(y, 2.0, 0.8, exposure=e)
    assert isinstance(r, dict)
    ap, bp = 2.0 + 11, 0.8 + 6.0
    assert r["alpha_post"] == ap and r["beta_post"] == pytest.approx(bp, rel=1e-15)
    assert r["rate_mean"] == pytest.approx(ap / bp, rel=1e-15)
    assert r["rate_var"] == pytest.approx(ap / bp ** 2, rel=1e-15)
    assert r["pred_mean"] == pytest.approx(ap / bp, rel=1e-15)
    assert r["pred_var"] == pytest.approx(ap / bp * (1 + 1 / bp), rel=1e-15)
    assert r["overdispersion"] == pytest.approx(1 + 1 / bp, rel=1e-15)


def test_poispr_edge():
    """Negative counts and a non-positive prior are refused."""
    with pytest.raises(ValueError):
        poisson_predictive([1, -1], 1.0, 1.0)
    with pytest.raises(ValueError):
        poisson_predictive([1, 2], 0.0, 1.0)
