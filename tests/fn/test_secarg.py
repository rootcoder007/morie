"""Tests for secarg.argon2id_kdf."""

from morie.fn import _array_core as np

from morie.fn.secarg import argon2id_kdf


def test_secarg_basic():
    """Test basic functionality."""
    password = np.random.default_rng(42).normal(0.0, 1.0, 40)
    salt = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = argon2id_kdf(password, salt)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_secarg_edge():
    """Test edge cases."""
    password = np.random.default_rng(42).normal(0.0, 1.0, 40)
    salt = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = argon2id_kdf(password, salt)
    assert isinstance(result, dict)
