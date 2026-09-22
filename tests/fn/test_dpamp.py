"""Tests for dpamp.privacy_amplification."""

from morie.fn import _array_core as np

from morie.fn.dpamp import privacy_amplification


def test_dpamp_basic():
    """Test basic functionality."""
    epsilon = 1e-6
    q = 0.01
    result = privacy_amplification(epsilon, q)
    assert isinstance(result, dict)
    expected_eps = float(np.log1p(q * np.expm1(epsilon)))
    expected_delta = q * 0.0
    expected_ratio = expected_eps / epsilon
    expected_linear = q * epsilon
    assert result["epsilon_amplified"] == expected_eps
    assert result["delta_amplified"] == expected_delta
    assert result["ratio"] == expected_ratio
    assert result["linear_approx"] == expected_linear
    assert result["epsilon"] == epsilon
    assert result["q"] == q
    assert result["epsilon_amplified"] < epsilon
    assert abs(result["epsilon_amplified"] - result["linear_approx"]) < 1e-4


def test_dpamp_edge():
    """Test edge cases."""
    epsilon = 1e-6
    q = 1.0
    result = privacy_amplification(epsilon, q)
    assert isinstance(result, dict)
    expected_eps = float(np.log1p(q * np.expm1(epsilon)))
    assert result["epsilon_amplified"] == expected_eps
    assert result["epsilon_amplified"] == epsilon
