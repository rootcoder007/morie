"""Tests for jjmsta.join_count (Cliff and Ord 1981 join counts).

Anchored on a hand-counted map: an 8-node path coloured 1 1 1 1 0 0 0 0
has 3 black-black joins, 3 white-white and 1 black-white, so the three
counts must sum to the 7 edges.  The weight constants are S0 = 14,
S1 = 28, S2 = 104 and E[J_BB] = S0 n(n-1) / (2 N (N-1)) = 1.5.
"""

import pytest

from morie.fn import _array_core as np
from morie.fn.jjmsta import join_count


def _path(n):
    W = np.zeros((n, n))
    for i in range(n - 1):
        W[i, i + 1] = 1.0
        W[i + 1, i] = 1.0
    return W


def test_hand_counted_path():
    res = join_count([1, 1, 1, 1, 0, 0, 0, 0], _path(8))
    assert res["BB"] == 3.0
    assert res["WW"] == 3.0
    assert res["BW"] == 1.0
    assert res["BB"] + res["WW"] + res["BW"] == 7.0  # one per edge
    assert (res["S0"], res["S1"], res["S2"]) == (14.0, 28.0, 104.0)
    assert res["E_BB"] == pytest.approx(1.5)
    assert res["z_BB"] > 0.0  # clustered map


def test_alternating_map_is_dispersed():
    res = join_count([1, 0] * 4, _path(8))
    assert res["BB"] == 0.0
    assert res["WW"] == 0.0
    assert res["BW"] == 7.0
    assert res["z_BB"] < 0.0


def test_counts_are_colour_symmetric():
    a = join_count([1, 1, 1, 1, 0, 0, 0, 0], _path(8))
    b = join_count([0, 0, 0, 0, 1, 1, 1, 1], _path(8))
    assert a["BB"] == b["WW"]
    assert a["WW"] == b["BB"]
    assert a["BW"] == b["BW"]


def test_variance_undefined_below_four_of_a_colour():
    res = join_count([1, 1, 1, 0, 0, 0, 0, 0], _path(8))
    assert res["V_BB"] != res["V_BB"]  # NaN: Cliff-Ord needs n_k >= 4


def test_rejects_non_binary_input():
    with pytest.raises(ValueError):
        join_count([0, 1, 2, 0, 1, 0, 1, 0], _path(8))
