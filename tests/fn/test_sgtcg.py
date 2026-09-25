"""Tests for sgtcg.sgt_commute_distance (commute time = 2m x resistance)."""

import pytest

from morie.fn.sgtcg import sgt_commute_distance


def test_sgtcg_basic():
    """Path a-b-c: resistances 1, 1, 2 (series), m = 2, so commute
    times 4, 4, 8; the diagonal is 0."""
    r = sgt_commute_distance([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    C = r["C"]
    exp = [[0, 4, 8], [4, 0, 4], [8, 4, 0]]
    assert r["two_m"] == 4
    for a, b in zip(C, exp):
        assert list(a) == pytest.approx(b, abs=1e-10)


def test_sgtcg_edge():
    """Triangle: each pair is a 1-ohm edge in parallel with a 2-ohm
    path, R = 2/3; m = 3, so C = 2 * 3 * 2/3 = 4. A non-symmetric
    matrix is refused."""
    r = sgt_commute_distance([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    C = r["C"]
    assert C[0][1] == pytest.approx(4.0, rel=1e-10)
    with pytest.raises(ValueError):
        sgt_commute_distance([[0, 1], [0, 0]])
