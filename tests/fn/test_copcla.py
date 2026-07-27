"""Tests for copcla."""

import numpy as np
import pytest

from morie.fn.copcla import clayton_copula

def test_copcla_basic():
    out = clayton_copula(0.5, 0.5, 2.0)
    assert out["cdf"] == pytest.approx((0.5**-2 + 0.5**-2 - 1) ** (-0.5))
    assert out["tau"] == pytest.approx(0.5)  # theta/(theta+2)


def test_copcla_edge():
    with pytest.raises(ValueError):
        clayton_copula(0.5, 0.5, 0.0)
    with pytest.raises(ValueError):
        clayton_copula(1.5, 0.5, 2.0)  # u outside [0, 1]
