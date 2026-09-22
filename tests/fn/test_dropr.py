"""Tests for dropr.dropout_regularization."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.dropr import dropout_regularization


def test_dropr_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    rate = 0.25
    mask = rng.integers(0, 2, 100).astype(float)
    result = dropout_regularization(x, mask, rate)
    assert isinstance(result, dict)
    # Documented return keys
    assert "activation" in result
    assert "kept" in result
    assert "dropped" in result
    assert "rate" in result
    assert "n" in result

    # The formula is a_i = x_i * m_i / (1 - rate)
    scale = 1.0 / (1.0 - rate)
    kept = int(sum(mask))
    expected_activation = [a * b * scale for a, b in zip(x, mask)]
    expected_dropped = len(x) - kept

    assert result["kept"] == kept
    assert result["dropped"] == expected_dropped
    assert result["rate"] == float(rate)
    assert result["n"] == len(x)
    assert len(result["activation"]) == len(x)
    for got, exp in zip(result["activation"], expected_activation):
        assert abs(got - exp) < 1e-12


def test_dropr_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    # rate = 0 means no dropout and no scaling
    rate = 0.0
    mask = [1.0] * 100
    result = dropout_regularization(x, mask, rate)
    assert isinstance(result, dict)
    assert result["rate"] == 0.0
    assert result["kept"] == 100
    assert result["dropped"] == 0
    assert result["n"] == 100
    for got, exp in zip(result["activation"], x):
        assert abs(got - exp) < 1e-12
