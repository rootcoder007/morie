"""Tests for gb331 (Gibbons shelf)."""

import numpy as np
import pytest

from morie.fn.gb331 import gibbons_run_lengths_dist


def test_gb331_basic():
    from math import comb
    # 0011: L1 = (2,), L2 = (2,): c=2, perms 1*1 -> 2/C(4,2)
    assert gibbons_run_lengths_dist([2], [2])["pmf"] == pytest.approx(2 / comb(4, 2))


def test_gb331_edge():
    with pytest.raises(ValueError):
        gibbons_run_lengths_dist([2, 2], [1, 1, 1, 1, 1])
