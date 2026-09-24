"""Tests for kmmamb.kamath_mamba_ssm."""

import pytest

from morie.fn import _array_core as np

from morie.fn.kmmamb import kamath_mamba_ssm


def test_kmmamb_basic():
    """Test basic functionality with shared (N,) B, C and a positive delta."""
    rng = np.random.default_rng(42)
    T, N = 40, 3
    x = rng.normal(0, 1, T)
    A = rng.uniform(-0.5, 0.0, N)
    B = rng.normal(0, 1, N)
    C = rng.normal(0, 1, N)
    delta = rng.uniform(0.05, 0.2, T)
    result = kamath_mamba_ssm(x, A, B, C, delta)
    assert isinstance(result, dict)
    assert result["n"] == T
    assert result["state_dim"] == N
    assert len(result["y"]) == T
    assert len(result["states"]) == T
    assert len(result["A_bar"]) == T
    assert result["estimate"] == result["y"][-1]
    assert result["method"] == "Mamba selective SSM scan (ZOH, diagonal A)"


def test_kmmamb_edge():
    """Test that a dense 2-D A is refused and the selective (T, N) path works."""
    rng = np.random.default_rng(42)
    T, N = 10, 2
    x = rng.normal(0, 1, T)
    A_dense = rng.normal(0, 1, (N, N))
    A_vec = rng.normal(0, 1, N)
    B = rng.normal(0, 1, (T, N))
    C = rng.normal(0, 1, (T, N))
    delta = rng.uniform(0.1, 1.0, T)
    with pytest.raises(ValueError):
        kamath_mamba_ssm(x, A_dense, B, C, delta)
    result = kamath_mamba_ssm(x, A_vec, B, C, delta)
    assert isinstance(result, dict)
    assert result["n"] == T
    assert result["state_dim"] == N
    assert len(result["y"]) == T
    assert len(result["states"]) == T
