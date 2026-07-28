"""Tests for ksr051 (Kosorok shelf)."""

import numpy as np
import pytest

from morie.fn.ksr051 import kosorok_ch2_continuous_invertibility


def test_ksr051_basic():
    A = np.array([[2.0, 0.0], [0.0, 3.0]])
    out = kosorok_ch2_continuous_invertibility(lambda th: A @ th, np.zeros(2))
    assert out["c_estimate"] >= 1.9  # smallest singular value is 2


def test_ksr051_edge():
    B = np.array([[1.0, 1.0], [1.0, 1.0]])  # rank deficient
    assert kosorok_ch2_continuous_invertibility(lambda th: B @ th,
                                                np.zeros(2))["c_estimate"] < 0.5
