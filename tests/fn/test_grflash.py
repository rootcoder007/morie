"""Out of chaos, comes order. — Friedrich Nietzsche"""

from morie.fn import _array_core as np

from morie.fn.grflash import geron_flash_attention_tile


def test_grflash_basic():
    """Test basic functionality."""
    Q = [[1.0, 0.0]]
    K = [[1.0, 0.0]] * 8
    V = [[1.0, 0.0]] * 8
    result = geron_flash_attention_tile(Q, K, V)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grflash_edge():
    """Test edge cases."""
    Q = [[1.0, 0.0]]
    K = [[1.0, 0.0]] * 8
    V = [[1.0, 0.0]] * 8
    result = geron_flash_attention_tile(Q, K, V)
    assert isinstance(result, dict)
