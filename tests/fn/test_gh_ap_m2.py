"""Tests for gh_ap_m2.ghosal_gibbs_sampler."""

from morie.fn.gh_ap_m2 import ghosal_gibbs_sampler


def test_gh_ap_m2_basic():
    """The sampled correlation recovers rho within ~5 Monte Carlo SE."""
    for rho in (0.6, -0.3):
        r = ghosal_gibbs_sampler(rho=rho, n_draws=20000, seed=5)
        assert abs(r["estimate"] - rho) < 0.035
        assert r["gap"] == abs(r["estimate"] - rho)


def test_gh_ap_m2_edge():
    """rho = 0 gives independent coordinates."""
    r = ghosal_gibbs_sampler(rho=0.0, n_draws=20000, seed=6)
    assert abs(r["estimate"]) < 0.035


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.gh_ap_m2 as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
