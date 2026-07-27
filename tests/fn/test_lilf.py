"""Tests for lilf.lilliefors_test."""

import numpy as np
import pytest

from morie.fn.lilf import lilliefors_test


def test_lilf_size_is_nominal_not_conservative():
    """The whole point of the Monte Carlo null: with fitted parameters the
    classical KS p-value under-rejects badly, the Lilliefors null does
    not. Measured 3/40 rejections at alpha = 0.05 (nominal 2)."""
    rej = 0
    for s in range(40):
        rng = np.random.default_rng(s)
        r = lilliefors_test(rng.standard_normal(60), n_mc=400, seed=s)
        rej += r.p_value < 0.05
    assert 0 <= rej <= 8


def test_lilf_rejects_uniform_data():
    rng = np.random.default_rng(1)
    r = lilliefors_test(rng.uniform(0, 1, 150), n_mc=400)
    assert r.p_value < 0.01


def test_lilf_is_seeded_and_reproducible():
    rng = np.random.default_rng(2)
    x = rng.exponential(1.0, 80)
    a = lilliefors_test(x, n_mc=300, seed=7)
    b = lilliefors_test(x, n_mc=300, seed=7)
    assert a.p_value == b.p_value
    assert a.statistic == b.statistic


def test_lilf_zero_variance_and_short_input():
    r = lilliefors_test([3.0, 3.0, 3.0, 3.0])
    assert r.p_value == 1.0
    with pytest.raises(ValueError, match="at least 4"):
        lilliefors_test([1.0, 2.0, 3.0])
