"""Tests for morie.fn.cuttip: Gomory cuts reach the brute-force integer optimum."""

import itertools

import pytest

from morie.fn.cuttip import cutting_plane


def _brute(c, A, b, hi=10):
    best = None
    for x in itertools.product(range(hi + 1), repeat=len(c)):
        if all(sum(a * v for a, v in zip(r, x)) <= bb for r, bb in zip(A, b)):
            val = sum(a * v for a, v in zip(c, x))
            best = val if best is None else max(best, val)
    return best


def test_textbook_example():
    r = cutting_plane([1.0, 1.0], [[-2.0, 2.0], [8.0, 10.0]], [1.0, 13.0])
    assert r["status"] == "optimal"
    assert r["objective"] == _brute([1, 1], [[-2, 2], [8, 10]], [1, 13])
    assert r["lp_bound"] >= r["objective"]


@pytest.mark.parametrize(
    "c,A,b",
    [
        ([3, 2], [[2, 3], [4, 1], [1, 1]], [12, 10, 10]),
        ([5, 4, 3], [[2, 3, 1], [4, 1, 2], [3, 4, 2], [1, 1, 1]], [5, 11, 8, 10]),
        ([1, 3], [[-1, 3], [3, 2], [1, 1]], [6, 12, 10]),
    ],
)
def test_integer_optimum(c, A, b):
    r = cutting_plane(c, A, b)
    assert r["status"] == "optimal"
    assert abs(r["objective"] - _brute(c, A, b)) < 1e-9
    assert all(abs(v - round(v)) < 1e-9 for v in r["x"])


def test_mixed_integer_and_validation():
    # x0 integer, x1 continuous: max x0 + x1, 2 x0 + 3 x1 <= 7, x0 <= 2.5 -> x0 = 2, x1 = 1
    r = cutting_plane([1.0, 1.0], [[2.0, 3.0], [1.0, 0.0]], [7.0, 2.5], integer_indices=[0])
    assert abs(r["x"][0] - 2.0) < 1e-9 and abs(r["x"][1] - 1.0) < 1e-9
    with pytest.raises(ValueError):
        cutting_plane([1.0], [[1.0]], [-1.0])
