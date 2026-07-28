"""Tests for gb_c1 (Gibbons shelf)."""

import numpy as np
import pytest

from morie.fn.gb_c1 import gibbons_chebyshev


def test_gb_c1_basic():
    rng = np.random.default_rng(10)
    out = gibbons_chebyshev(2.0, rng.standard_normal(3000))
    assert out["empirical"] <= out["bound"]


def test_gb_c1_edge():
    assert gibbons_chebyshev(0.5)["bound"] == 1.0  # capped at 1
    with pytest.raises(ValueError):
        gibbons_chebyshev(-1.0)
