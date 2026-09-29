"""Tests for map_estimate."""

import pytest

from morie.fn import _array_core as np
from morie.fn.mapst import map_estimate, mapst


def test_shrinkage():
    x = np.array([10.0, 10.0, 10.0])
    r = map_estimate(x, prior_mu=0.0, prior_sigma=1.0)
    assert r.estimate < 10.0
    assert r.estimate > 0.0


def test_alias():
    assert mapst is map_estimate


def test_bad_sigma():
    with pytest.raises(ValueError):
        map_estimate([1, 2], prior_sigma=0.0)


def test_large_n_approaches_mle():
    rng = np.random.default_rng(42)
    x = rng.normal(5.0, 1.0, 10000)
    r = map_estimate(x, prior_mu=0.0, prior_sigma=1.0)
    assert abs(r.estimate - 5.0) < 0.1


def test_map_is_the_precision_weighted_mean():
    x = [4.0, 5.5, 3.5, 6.0]
    n = 4
    xb = sum(x) / n
    s2 = sum((v - xb) ** 2 for v in x) / n
    pd, pp = n / s2, 1 / 2.0**2
    r = map_estimate(x, prior_mu=1.0, prior_sigma=2.0)
    assert r.estimate == pytest.approx((pd * xb + pp * 1.0) / (pd + pp), rel=1e-14)
    assert r.extra["posterior_var"] == pytest.approx(1 / (pd + pp), rel=1e-14)
