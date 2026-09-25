"""Tests for wsmgib.wasserman_gibbs_sampler."""

import pytest

from morie.fn.wsmgib import wasserman_gibbs_sampler


def test_wsmgib_basic():
    """The chain's correlation recovers rho; lag-one autocorrelation rho^2
    leaves about n (1 - rho^2) / (1 + rho^2) effective draws, so with
    rho = 0.9 and n = 20000 the standard error is near 0.003."""
    out = wasserman_gibbs_sampler(0.9, [0.0, 0.0], 20000)
    assert abs(out["estimate"] - 0.9) < 0.015
    assert abs(out["mean_x"]) < 0.15 and abs(out["mean_y"]) < 0.15


def test_wsmgib_edge():
    """|rho| must be below 1; seeded chains repeat."""
    a = wasserman_gibbs_sampler(-0.5, [1.0, -1.0], 500, seed=3)
    b = wasserman_gibbs_sampler(-0.5, [1.0, -1.0], 500, seed=3)
    assert a["estimate"] == b["estimate"]
    with pytest.raises(ValueError, match="rho"):
        wasserman_gibbs_sampler(1.0, [0.0, 0.0], 100)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.wsmgib as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
