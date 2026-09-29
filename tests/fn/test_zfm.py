"""Tests for zfm.z_transform."""

from morie.fn import _array_core as np
from morie.fn.zfm import z_transform


def test_zfm_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    z = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = z_transform(x, z)
    assert isinstance(result, dict)
    assert "coefficients" in result


def test_zfm_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    z = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = z_transform(x, z)
    assert isinstance(result, dict)


def test_z_transform_evaluated_directly():
    import pytest

    x = [1.0, -0.5, 0.25, 2.0]
    z = 1.5 + 0.5j
    want = sum(v * z ** (-n) for n, v in enumerate(x))
    r = z_transform(x, z)
    got = complex(r["X"] if not isinstance(r["X"], list) else r["X"][0])
    assert got == pytest.approx(want, rel=1e-13)
    shifted = z_transform(x, z, n0=2)
    got2 = complex(shifted["X"] if not isinstance(shifted["X"], list) else shifted["X"][0])
    assert got2 == pytest.approx(want * z ** (-2), rel=1e-13)
