"""Tests for foldp."""

import numpy as np
import pytest

from morie.fn.foldp import folding_problem


def test_foldp_basic():
    x = np.linspace(-1, 1, 15)
    y = np.linspace(-1, 1, 6)
    T = 5.0 - (x[:, None] - y[None, :]) ** 2
    assert folding_problem(T)["single_peaked_share"] == pytest.approx(1.0)


def test_foldp_edge():
    T = np.ones((4, 5))
    with pytest.raises(ValueError):
        folding_problem(T, stimulus_order=[0, 1, 2, 3, 3])  # not a permutation
    with pytest.raises(ValueError):
        folding_problem(np.ones(5))  # 1-D
