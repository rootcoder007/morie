"""Test bfdr."""

from morie.fn import _array_core as np
from morie.fn.bfdr import bayesian_fdr


def test_bfdr_basic():
    pp = np.array([0.9, 0.8, 0.3, 0.1, 0.95])
    r = bayesian_fdr(pp, threshold=0.5)
    assert 0.0 <= r.value <= 1.0
    assert r.extra["n_discoveries"] == 3


def test_bfdr_no_discoveries():
    pp = np.array([0.1, 0.2, 0.3])
    r = bayesian_fdr(pp, threshold=0.9)
    assert r.value == 0.0
    assert r.extra["n_discoveries"] == 0


def test_bfdr_is_mean_null_probability_among_discoveries():
    import pytest

    pp = [0.9, 0.8, 0.3, 0.1, 0.95, 0.72]
    disc = [p for p in pp if p >= 0.7]
    r = bayesian_fdr(pp, threshold=0.7)
    assert r.value == pytest.approx(sum(1 - p for p in disc) / len(disc), rel=1e-14)
    assert r.extra["mean_posterior_h1"] == pytest.approx(sum(pp) / 6, rel=1e-14)
