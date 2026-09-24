"""Tests for secaead.aead_chacha20poly1305."""

from morie.fn import _array_core as np

from morie.fn.secaead import aead_encrypt


def test_secaead_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    key = rng.integers(0, 256, 32)
    nonce = rng.integers(0, 256, 12)
    plaintext = rng.integers(0, 256, 64)
    aad = rng.integers(0, 256, 16)
    result = aead_encrypt(key, nonce, plaintext, aad)
    assert isinstance(result, dict)
    assert "ciphertext" in result
    assert "tag" in result


def test_secaead_edge():
    """Test edge cases."""
    rng = np.random.default_rng(43)
    key = rng.integers(0, 256, 32)
    nonce = rng.integers(0, 256, 12)
    plaintext = []
    aad = []
    result = aead_encrypt(key, nonce, plaintext, aad)
    assert isinstance(result, dict)
    assert "ciphertext" in result
    assert "tag" in result
