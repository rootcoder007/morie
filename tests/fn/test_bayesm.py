"""Tests for bayesm.dp_bayesian_mechanism."""

import pytest

from morie.fn._rng import random_uniform
from morie.fn.bayesm import dp_bayesian_mechanism

POST = [0.3, -1.2, 0.8, 1.5, -0.4, 0.1, 2.0, 1.1]


def test_bayesm_releases_one_philox_chosen_draw():
    r = dp_bayesian_mechanism([1.0] * 20, POST, epsilon=0.5, B=2.0, seed=11)
    j = int(float(random_uniform(1, seed=11, stream=0)[0]) * len(POST))
    assert r["draw_index"] == j
    assert r["released"] == POST[j] and r["estimate"] == POST[j]
    assert r["temperature"] == pytest.approx(2 * 2.0 / 0.5, rel=1e-15)
    assert r["eps_free"] == 4.0
    assert r["laplace_scale"] == pytest.approx((1 / 20) / 0.5, rel=1e-15)
    m = sum(POST) / len(POST)
    assert r["posterior_mean"] == pytest.approx(m, rel=1e-14)


def test_bayesm_edge():
    with pytest.raises(ValueError, match="posterior_sample is required"):
        dp_bayesian_mechanism([1.0, 2.0])
