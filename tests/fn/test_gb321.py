"""Tests for gb321 (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb321 import gibbons_runs_joint_dist


def test_gb321_basic():
    from math import comb
    # n1 = n2 = 2, r1 = r2 = 1: 2 * 1 * 1 / C(4,2) = 1/3
    assert gibbons_runs_joint_dist(1, 1, 2, 2)["pmf"] == pytest.approx(2 / comb(4, 2))


def test_gb321_edge():
    assert gibbons_runs_joint_dist(1, 3, 4, 4)["pmf"] == 0.0  # alternation
    with pytest.raises(ValueError):
        gibbons_runs_joint_dist(5, 1, 4, 4)
