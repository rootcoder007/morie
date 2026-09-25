"""Tests for wsmemt.wasserman_em_algorithm."""

import math

import pytest

from morie.fn.wsmemt import wasserman_em_algorithm

# Two tight clusters of four points each, eight points in all.
_X = [0.0, 0.1, -0.1, 0.05, 10.0, 10.1, 9.9, 10.05]
_LOW = _X[:4]
_HIGH = _X[4:]


def _mixture_loglik(X, pi, mu1, mu2, sd1, sd2):
    """The observed-data log-likelihood, written out here so the value the
    module reports is checked against an independent computation."""
    total = 0.0
    for x in X:
        d1 = (1.0 - pi) * math.exp(-0.5 * ((x - mu1) / sd1) ** 2) / (
            sd1 * math.sqrt(2.0 * math.pi)
        )
        d2 = pi * math.exp(-0.5 * ((x - mu2) / sd2) ** 2) / (
            sd2 * math.sqrt(2.0 * math.pi)
        )
        total += math.log(d1 + d2)
    return total


def test_wsmemt_basic():
    """Well-separated clusters: each component collapses onto one cluster."""
    out = wasserman_em_algorithm(_X, (0.5, -1.0, 11.0, 1.0, 1.0))

    assert out["n"] == len(_X)
    assert out["converged"] is True
    assert out["iterations"] >= 2
    assert out["estimate"] == out["pi"]

    # Four points in each cluster, so the mixing weight is exactly 1/2.
    assert out["pi"] == pytest.approx(len(_HIGH) / len(_X), abs=1e-9)

    # Each component mean is the plain mean of its cluster.
    mu_low = sum(_LOW) / len(_LOW)
    mu_high = sum(_HIGH) / len(_HIGH)
    assert out["mu1"] == pytest.approx(mu_low, abs=1e-6)
    assert out["mu2"] == pytest.approx(mu_high, abs=1e-6)

    # And each sd is that cluster's population sd (divide by n, not n-1).
    sd_low = math.sqrt(sum((v - mu_low) ** 2 for v in _LOW) / len(_LOW))
    sd_high = math.sqrt(sum((v - mu_high) ** 2 for v in _HIGH) / len(_HIGH))
    assert out["sd1"] == pytest.approx(sd_low, abs=1e-6)
    assert out["sd2"] == pytest.approx(sd_high, abs=1e-6)

    # The reported log-likelihood is the mixture log-likelihood at the
    # reported parameters.
    assert out["log_likelihood"] == pytest.approx(
        _mixture_loglik(
            _X, out["pi"], out["mu1"], out["mu2"], out["sd1"], out["sd2"]
        ),
        rel=1e-6,
    )


def test_wsmemt_likelihood_is_monotone_and_label_swap_symmetric():
    """Swapping the two initial centres swaps the fitted components."""
    a = wasserman_em_algorithm(_X, (0.5, -1.0, 11.0, 1.0, 1.0))
    b = wasserman_em_algorithm(_X, (0.5, 11.0, -1.0, 1.0, 1.0))

    assert a["mu1"] == pytest.approx(b["mu2"], abs=1e-6)
    assert a["mu2"] == pytest.approx(b["mu1"], abs=1e-6)
    assert a["sd1"] == pytest.approx(b["sd2"], abs=1e-6)
    assert a["sd2"] == pytest.approx(b["sd1"], abs=1e-6)
    assert a["pi"] == pytest.approx(1.0 - b["pi"], abs=1e-6)
    # The same mode, so the same likelihood either way round.
    assert a["log_likelihood"] == pytest.approx(b["log_likelihood"], rel=1e-9)

    # Capping the iterations cannot beat running to convergence.
    early = wasserman_em_algorithm(_X, (0.5, -1.0, 11.0, 1.0, 1.0), max_iter=2)
    assert early["iterations"] == 2
    assert early["converged"] is False
    assert early["log_likelihood"] <= a["log_likelihood"] + 1e-9


def test_wsmemt_single_cluster_data():
    """One cluster: both components land on the same sample mean."""
    X = [1.0, 1.2, 0.8, 1.1, 0.9, 1.05]
    out = wasserman_em_algorithm(X, (0.5, 0.9, 1.1, 0.5, 0.5))
    xbar = sum(X) / len(X)
    # Symmetric start, symmetric data: the fit does not split the sample.
    assert out["mu1"] == pytest.approx(xbar, abs=0.25)
    assert out["mu2"] == pytest.approx(xbar, abs=0.25)
    assert 0.0 < out["pi"] < 1.0
    assert out["log_likelihood"] == pytest.approx(
        _mixture_loglik(
            X, out["pi"], out["mu1"], out["mu2"], out["sd1"], out["sd2"]
        ),
        rel=1e-6,
    )


def test_wsmemt_edge():
    """Sample size, mixing weight and sd validation."""
    with pytest.raises(ValueError):
        wasserman_em_algorithm([1.0], (0.5, 0.0, 1.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        wasserman_em_algorithm(_X, (1.5, 0.0, 1.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        wasserman_em_algorithm(_X, (0.0, 0.0, 1.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        wasserman_em_algorithm(_X, (0.5, 0.0, 1.0, 0.0, 1.0))
    with pytest.raises(ValueError):
        wasserman_em_algorithm(_X, (0.5, 0.0, 1.0, 1.0, -2.0))

    # Two points, one per component: the means are the points themselves.
    two = wasserman_em_algorithm([0.0, 5.0], (0.5, -1.0, 6.0, 1.0, 1.0))
    assert two["n"] == 2
    assert two["mu1"] == pytest.approx(0.0, abs=1e-6)
    assert two["mu2"] == pytest.approx(5.0, abs=1e-6)
    assert two["pi"] == pytest.approx(0.5, abs=1e-6)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.wsmemt as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
