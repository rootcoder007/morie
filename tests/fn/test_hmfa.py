"""Without music, life would be a mistake. — Friedrich Nietzsche"""

from morie.fn import _array_core as np

from morie.fn.hmfa import geron_flash_attention


def test_hmfa_basic():
    """Test basic functionality."""
    Q = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    K = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    V = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = geron_flash_attention(Q, K, V)
    assert isinstance(result, dict)
    assert "estimate" in result or "output" in result


def test_hmfa_edge():
    """Test edge cases."""
    Q = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    K = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    V = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = geron_flash_attention(Q, K, V)
    assert isinstance(result, dict)
