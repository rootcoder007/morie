"""Tests for gh_ap_m1.ghosal_mh_sampler."""

from morie.fn.gh_ap_m1 import ghosal_mh_sampler


def test_gh_ap_m1_basic():
    """Stationary N(0, 1): variance 1 and mean 0 within ~5 Monte Carlo SE."""
    r = ghosal_mh_sampler(n_draws=20000, seed=3)
    assert abs(r["estimate"] - 1.0) < 0.09
    assert abs(r["mean"]) < 0.07


def test_gh_ap_m1_edge():
    """Acceptance rate (2/pi) arctan 2 for an N(0, 1) proposal; seeded runs
    reproduce exactly."""
    import math
    r = ghosal_mh_sampler(n_draws=20000, seed=4)
    assert abs(r["accept_rate"] - 2 / math.pi * math.atan(2)) < 0.02
    assert ghosal_mh_sampler(n_draws=500, seed=9)["estimate"] == \
        ghosal_mh_sampler(n_draws=500, seed=9)["estimate"]


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.gh_ap_m1 as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
