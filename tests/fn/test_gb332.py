"""Tests for gb332 (Gibbons shelf)."""

import numpy as np
import pytest

from morie.fn.gb332 import gibbons_type1_run_lengths


def test_gb332_basic():
    from math import comb
    assert gibbons_type1_run_lengths([2], n2=2)["pmf"] == pytest.approx(
        comb(3, 1) / comb(4, 2))


def test_gb332_edge():
    with pytest.raises(ValueError):
        gibbons_type1_run_lengths([2], n2=None)
