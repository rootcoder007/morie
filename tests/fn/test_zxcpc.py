"""Tests for zxcpc."""

import numpy as np
import pytest

from morie.fn.zxcpc import copula_clayton_sp

def test_zxcpc_basic():
    rng = np.random.default_rng(42)
    cov = np.array([[1.0, 0.7], [0.7, 1.0]])
    X = rng.multivariate_normal(np.zeros(2), cov, size=300)
    out = copula_clayton_sp(X)
    assert out["theta_matrix"][0, 1] > 0
    assert out["tau_matrix"][0, 1] == pytest.approx(0.5, abs=0.15)


def test_zxcpc_edge():
    with pytest.raises(ValueError):
        copula_clayton_sp(np.zeros((10, 1)))  # needs >= 2 variables
