"""Tests for morie.fn.lornz -- Lorenz curve and the Gini it implies (Lorenz 1905)."""

import numpy as np
import pytest

from morie.fn.lornz import lorenz_curve


def test_lornz_perfect_equality_is_the_diagonal():
    """Equal incomes put the curve on y = x, so the area under it is 1/2 and
    Gini = 1 - 2*(1/2) = 0."""
    r = lorenz_curve(incomes=np.full(100, 50_000.0))
    assert r.extra["area_under_curve"] == pytest.approx(0.5, abs=1e-12)
    assert r.extra["gini_from_lorenz"] == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(r.extra["y"], r.extra["x"])


def test_lornz_maximal_inequality_approaches_one():
    """One person holds everything: Gini -> (n-1)/n, not 1, for finite n."""
    n = 100
    incomes = np.zeros(n)
    incomes[-1] = 1.0
    g = lorenz_curve(incomes=incomes).extra["gini_from_lorenz"]
    assert g == pytest.approx((n - 1) / n, abs=1e-9)


def test_lornz_gini_matches_the_direct_mean_absolute_difference_formula():
    """G = sum_i sum_j |x_i - x_j| / (2 n^2 xbar), computed independently.

    The trapezoid-on-the-Lorenz-curve route and the pairwise-difference route
    are different computations of the same quantity, so agreement is evidence
    rather than restatement.
    """
    rng = np.random.default_rng(2026)
    x = rng.lognormal(mean=10.0, sigma=0.8, size=400)
    n = x.size
    direct = np.abs(x[:, None] - x[None, :]).sum() / (2 * n**2 * x.mean())
    got = lorenz_curve(incomes=x).extra["gini_from_lorenz"]
    # The two agree to trapezoid discretisation error (~1e-5 at n = 400),
    # which is the residual from integrating a piecewise-linear curve rather
    # than any difference in what is being estimated.
    assert got == pytest.approx(direct, rel=1e-4)


def test_lornz_is_scale_invariant():
    """Gini measures relative inequality: doubling every income changes nothing."""
    rng = np.random.default_rng(11)
    x = rng.uniform(1_000, 100_000, 200)
    a = lorenz_curve(incomes=x).extra["gini_from_lorenz"]
    b = lorenz_curve(incomes=x * 2.5).extra["gini_from_lorenz"]
    assert a == pytest.approx(b, rel=1e-12)


def test_lornz_curve_is_monotone_and_convex_and_anchored():
    """Structural properties of any Lorenz curve."""
    rng = np.random.default_rng(7)
    r = lorenz_curve(incomes=rng.lognormal(9.0, 1.0, 150))
    x, y = np.asarray(r.extra["x"]), np.asarray(r.extra["y"])
    assert x[0] == 0.0 and y[0] == 0.0
    assert x[-1] == pytest.approx(1.0) and y[-1] == pytest.approx(1.0)
    assert np.all(np.diff(y) >= -1e-12)              # non-decreasing
    assert np.all(np.diff(np.diff(y)) >= -1e-9)      # convex
    assert np.all(y <= x + 1e-12)                    # never above the diagonal


def test_lornz_gini_lies_in_the_unit_interval():
    rng = np.random.default_rng(3)
    for sigma in (0.1, 0.5, 1.0, 2.0):
        g = lorenz_curve(incomes=rng.lognormal(9.0, sigma, 200)).extra["gini_from_lorenz"]
        assert 0.0 <= g < 1.0


def test_lornz_more_dispersion_means_more_inequality():
    rng = np.random.default_rng(5)
    ginis = [
        lorenz_curve(incomes=rng.lognormal(9.0, s, 3000)).extra["gini_from_lorenz"]
        for s in (0.2, 0.6, 1.2)
    ]
    assert ginis == sorted(ginis)


def test_lornz_is_order_invariant():
    rng = np.random.default_rng(13)
    x = rng.uniform(1, 100, 80)
    assert lorenz_curve(incomes=x).extra["gini_from_lorenz"] == pytest.approx(
        lorenz_curve(incomes=rng.permutation(x)).extra["gini_from_lorenz"]
    )


def test_cheatsheet():
    from morie.fn.lornz import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
