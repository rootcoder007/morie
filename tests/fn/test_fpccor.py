"""fpccor: integrated functional correlation (Ramsay & Silverman 2005)."""

import numpy as np
import pytest

from morie.fn.fpccor import functional_correlation as fcorr


def test_fpccor_identical_samples_give_one():
    rng = np.random.default_rng(1039)
    X = rng.standard_normal((30, 50))
    assert fcorr(X, X)["estimate"] == pytest.approx(1.0)


def test_fpccor_negated_sample_gives_minus_one():
    rng = np.random.default_rng(1049)
    X = rng.standard_normal((30, 50))
    assert fcorr(X, -X)["estimate"] == pytest.approx(-1.0)


def test_fpccor_independent_samples_give_approximately_zero():
    rng = np.random.default_rng(1051)
    X = rng.standard_normal((4000, 20))
    Y = rng.standard_normal((4000, 20))
    assert abs(fcorr(X, Y)["estimate"]) < 0.05


def test_fpccor_recovers_a_planted_amplitude_coupling():
    """Curves share a scalar amplitude with known correlation rho.

    X_i(t) = a_i f(t), Y_i(t) = b_i f(t) with corr(a,b) = rho makes the
    integrated functional correlation equal rho exactly, so the answer is
    known in advance rather than read off the implementation.
    """
    rng = np.random.default_rng(1061)
    n, rho = 8000, 0.7
    t = np.linspace(0, 1, 40)
    f = np.sin(2 * np.pi * t) + 2.0
    a = rng.standard_normal(n)
    b = rho * a + np.sqrt(1 - rho**2) * rng.standard_normal(n)
    X = np.outer(a, f)
    Y = np.outer(b, f)
    assert fcorr(X, Y)["estimate"] == pytest.approx(rho, abs=0.03)


def test_fpccor_stays_within_the_correlation_bounds():
    rng = np.random.default_rng(1063)
    for _ in range(50):
        X = rng.standard_normal((15, 25))
        Y = rng.standard_normal((15, 25))
        assert -1.0 - 1e-12 <= fcorr(X, Y)["estimate"] <= 1.0 + 1e-12


def test_fpccor_is_invariant_to_scaling_each_sample():
    rng = np.random.default_rng(1069)
    X = rng.standard_normal((40, 30))
    Y = rng.standard_normal((40, 30))
    assert fcorr(3.0 * X, 0.2 * Y)["estimate"] == pytest.approx(
        fcorr(X, Y)["estimate"]
    )


def test_fpccor_irregular_grid_changes_the_weighting():
    """On an uneven grid the trapezoid weights differ from a plain sum.

    Ignoring argvals overweights the densely sampled region, so the two
    answers must actually differ -- otherwise the parameter does nothing.
    """
    rng = np.random.default_rng(1087)
    n_pts = 30
    t = np.sort(rng.uniform(0, 1, n_pts))
    t[0], t[-1] = 0.0, 1.0
    X = rng.standard_normal((60, n_pts))
    Y = 0.5 * X + rng.standard_normal((60, n_pts))
    assert fcorr(X, Y, argvals=t)["estimate"] != pytest.approx(
        fcorr(X, Y)["estimate"], abs=1e-9
    )


def test_fpccor_rejects_mismatched_or_degenerate_input():
    rng = np.random.default_rng(1091)
    with pytest.raises(ValueError, match="same shape"):
        fcorr(rng.standard_normal((10, 5)), rng.standard_normal((10, 6)))
    with pytest.raises(ValueError, match="at least 2 curves"):
        fcorr(np.zeros((1, 5)), np.zeros((1, 5)))
    with pytest.raises(ValueError, match="zero integrated variance"):
        fcorr(np.ones((4, 5)), rng.standard_normal((4, 5)))


def test_fpccor_rejects_bad_argvals():
    rng = np.random.default_rng(1093)
    X = rng.standard_normal((5, 4))
    with pytest.raises(ValueError, match="must have length"):
        fcorr(X, X, argvals=np.arange(3.0))
    with pytest.raises(ValueError, match="strictly increasing"):
        fcorr(X, X, argvals=np.array([0.0, 1.0, 1.0, 2.0]))
