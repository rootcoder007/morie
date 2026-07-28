"""Tests for ksr062 (Kosorok shelf)."""

import numpy as np
import pytest

from morie.fn.ksr062 import kosorok_ch3_pathwise_derivative


def test_ksr062_basic():
    rng = np.random.default_rng(18)
    x = rng.standard_normal(300)
    out = kosorok_ch3_pathwise_derivative(x - x.mean(), x)
    assert out["mean_zero"] is True


def test_ksr062_edge():
    rng = np.random.default_rng(18)
    x = rng.standard_normal(300)
    assert kosorok_ch3_pathwise_derivative(x - x.mean() + 5.0, x)["mean_zero"] is False
