"""Tests for gh_ap_g3.ghosal_dir_marginal."""

import pytest

from morie.fn.gh_ap_g3 import ghosal_dir_marginal


def test_gh_ap_g3_basic():
    """Prop. G.3: merging cells of Dir(2, 3, 5) gives Be(5, 5): mean 1/2,
    variance 25 / (100 * 11)."""
    r = ghosal_dir_marginal([2.0, 3.0, 5.0], merge_idx=(0, 1))
    assert r["beta_params"] == [5.0, 5.0]
    assert r["estimate"] == 0.5
    assert r["variance"] == pytest.approx(25 / 1100, rel=1e-15)


def test_gh_ap_g3_edge():
    """A single cell is its own Beta marginal; indices must exist."""
    r = ghosal_dir_marginal([2.0, 3.0, 5.0], merge_idx=(2,))
    assert r["beta_params"] == [5.0, 5.0]
    with pytest.raises(ValueError, match="merge_idx"):
        ghosal_dir_marginal([42.0])


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.gh_ap_g3 as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
