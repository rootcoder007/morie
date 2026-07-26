"""drpfw: inverted dropout forward pass (Srivastava et al. 2014)."""

import numpy as np
import pytest

from morie.fn.drpfw import dropout_forward as do


def test_drpfw_eval_mode_is_the_identity():
    """At inference dropout must do nothing at all."""
    rng = np.random.default_rng(1901)
    x = rng.standard_normal((10, 8))
    assert np.asarray(do(x, p=0.5, training=False)["y"]) == pytest.approx(x)


def test_drpfw_p_zero_keeps_everything():
    rng = np.random.default_rng(1907)
    x = rng.standard_normal(40)
    r = do(x, p=0.0, seed=1)
    assert np.asarray(r["y"]) == pytest.approx(x)
    assert r["kept_fraction"] == pytest.approx(1.0)


def test_drpfw_surviving_units_are_scaled_by_one_over_one_minus_p():
    """"Inverted" dropout: the scaling happens at TRAIN time, which is what
    lets eval mode be a plain identity."""
    rng = np.random.default_rng(1913)
    x = np.ones(2000)
    p = 0.4
    y = np.asarray(do(x, p=p, seed=7)["y"])
    survivors = y[y != 0]
    assert survivors == pytest.approx(np.full(survivors.size, 1 / (1 - p)))


def test_drpfw_preserves_the_expectation_of_its_input():
    """E[y] = E[x]. This is the whole reason for the 1/(1-p) factor."""
    rng = np.random.default_rng(1931)
    x = rng.normal(3.0, 1.0, 200_000)
    y = np.asarray(do(x, p=0.5, seed=11)["y"])
    assert y.mean() == pytest.approx(x.mean(), rel=0.02)


def test_drpfw_kept_fraction_tracks_one_minus_p():
    rng = np.random.default_rng(1933)
    x = rng.standard_normal(100_000)
    for p in (0.1, 0.5, 0.9):
        assert do(x, p=p, seed=3)["kept_fraction"] == pytest.approx(1 - p, abs=0.01)


def test_drpfw_is_reproducible_for_a_fixed_seed():
    rng = np.random.default_rng(1949)
    x = rng.standard_normal(500)
    a = np.asarray(do(x, p=0.3, seed=42)["y"])
    b = np.asarray(do(x, p=0.3, seed=42)["y"])
    assert a == pytest.approx(b, abs=0.0)


def test_drpfw_different_seeds_give_different_masks():
    rng = np.random.default_rng(1951)
    x = np.ones(500)
    a = np.asarray(do(x, p=0.5, seed=1)["mask"])
    b = np.asarray(do(x, p=0.5, seed=2)["mask"])
    assert not np.array_equal(a, b)


def test_drpfw_rejects_an_out_of_range_rate():
    with pytest.raises(ValueError):
        do(np.ones(5), p=1.5)
