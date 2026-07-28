"""Tests for gb2112 (Gibbons shelf)."""

import numpy as np
import pytest

from morie.fn.gb2112 import gibbons_block_freq_dist


def test_gb2112_basic():
    from math import comb
    assert gibbons_block_freq_dist(4, 3)["n_compositions"] == comb(7, 3)


def test_gb2112_edge():
    assert gibbons_block_freq_dist(4, 3, [3, 0, 0, 0, 0])["valid_composition"] is True
    with pytest.raises(ValueError):
        gibbons_block_freq_dist(4, 3, [1, 1, 1])  # wrong length
