"""Tests for copt."""

import numpy as np
import pytest

from morie.fn.copt import t_copula

def test_copt_basic():
    out = t_copula(0.4, 0.6, 0.5, nu=5.0)
    assert 0 < out["cdf"] < min(0.4, 0.6) + 1e-8
    assert out["tau"] == pytest.approx(2 / np.pi * np.arcsin(0.5))


def test_copt_edge():
    with pytest.raises(ValueError):
        t_copula(0.4, 0.6, 0.5, nu=-1.0)
    with pytest.raises(ValueError):
        t_copula(0.4, 0.6, 1.2, nu=5.0)
