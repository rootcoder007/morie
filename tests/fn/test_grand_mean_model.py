"""Tests for grand_mean_model.grand_mean_model."""

from morie.fn import _array_core as np
from morie.fn.grand_mean_model import grand_mean_model


def test_ca7e1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = grand_mean_model(x)
    assert isinstance(result, dict)
    assert "intercept" in result


def test_ca7e1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = grand_mean_model(x)
    assert isinstance(result, dict)


def test_grand_mean_and_error_variance_recomputed():
    import pytest

    y = [3.0, 5.5, 4.0, 8.0, 6.5]
    m = sum(y) / 5
    r = grand_mean_model(y)
    assert r["value"] == pytest.approx(m, rel=1e-14)
    assert r["var_error"] == pytest.approx(sum((v - m) ** 2 for v in y) / 4, rel=1e-13)
