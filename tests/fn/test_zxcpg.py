"""Tests for zxcpg."""

import numpy as np
import pytest

from morie.fn.zxcpg import copula_gauss_sp

def test_zxcpg_basic():
    rng = np.random.default_rng(42)
    cov = np.array([[1.0, 0.7], [0.7, 1.0]])
    X = rng.multivariate_normal(np.zeros(2), cov, size=400)
    out = copula_gauss_sp(X)
    assert out["correlation"][0, 1] == pytest.approx(0.7, abs=0.1)
    assert out["positive_definite"] is True


def test_zxcpg_edge():
    with pytest.raises(ValueError):
        copula_gauss_sp(np.zeros((3, 2)))  # too few observations
