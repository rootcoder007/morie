"""Tests for ghs022.ghosal_ch3_tailfree_cell_counts.

This is a count, not an estimate, so the anchors are exact: cells of a
tiling partition must account for every observation exactly once, and
the half-open convention [lo, hi) must place a point sitting on a
breakpoint in the upper cell only.
"""

import pytest

from morie.fn.ghs022 import ghosal_ch3_tailfree_cell_counts as cells

X = [0.31, 0.87, 1.02, 1.44, 1.91, 2.05, 2.63, 3.10, 3.88, 5.02, 6.41, 8.20]


def test_tiling_partition_accounts_for_every_point():
    res = cells(X, [(0.0, 2.0), (2.0, 4.0), (4.0, 100.0)])
    assert list(res["N_epsilon"]) == [5, 4, 3]
    assert sum(res["N_epsilon"]) == len(X)
    assert sum(float(p) for p in res["proportion"]) == pytest.approx(1.0)


def test_breakpoint_goes_to_the_upper_cell_only():
    res = cells([1.0, 2.0, 3.0], [(1.0, 2.0), (2.0, 3.0), (3.0, 4.0)])
    assert list(res["N_epsilon"]) == [1, 1, 1]


def test_single_cell_returns_a_scalar():
    res = cells(X, (0.0, 2.0))
    assert res["N_epsilon"] == 5
    assert res["proportion"] == pytest.approx(5 / 12)


def test_predicate_cells_are_accepted():
    res = cells(X, lambda v: v > 3.0)
    # 3.10, 3.88, 5.02, 6.41, 8.20
    assert res["N_epsilon"] == 5


def test_n_truncates_to_the_first_n_observations():
    assert cells(X, (0.0, 100.0), n=5)["N_epsilon"] == 5
    assert cells(X, (0.0, 100.0), n=0)["N_epsilon"] == 0


def test_rejects_out_of_range_n():
    with pytest.raises(ValueError):
        cells(X, (0.0, 1.0), n=99)


def test_returns_an_integer_count_not_a_probability():
    """The pasted KS body returned a number in [0, 1] here.  A count is
    an integer and may exceed 1."""
    res = cells(X, (0.0, 100.0))
    assert isinstance(res["N_epsilon"], int)
    assert res["N_epsilon"] == 12
