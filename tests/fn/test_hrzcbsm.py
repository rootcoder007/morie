"""Tests for hrzcbsm.horowitz_choice_based_sms."""

from morie.fn import _array_core as np
from morie.fn.hrzcbsm import horowitz_choice_based_sms


def test_hrzcbsm_basic():
    """Test basic functionality with valid inputs."""
    rng = np.random.default_rng(42)
    n, d = 40, 3
    x = rng.normal(0, 1, (n, d))
    y = [0.0] * (n // 2) + [1.0] * (n - n // 2)
    sampling_weights = 0.5
    result = horowitz_choice_based_sms(x, y, sampling_weights)
    assert isinstance(result, dict)
    assert "beta" in result
    assert "score" in result
    assert "pi1" in result
    assert "pi0" in result
    assert "n1" in result
    assert "n0" in result
    assert result["n"] == n
    assert result["d"] == d
    assert result["smoothed"] is True


def test_hrzcbsm_edge():
    """Test edge case: pair form of sampling weights with smoothed=False."""
    rng = np.random.default_rng(123)
    n, d = 40, 3
    x = rng.normal(0, 1, (n, d))
    y = [0.0] * (n // 2) + [1.0] * (n - n // 2)
    sampling_weights = [0.6, 0.4]
    result = horowitz_choice_based_sms(
        x, y, sampling_weights, smoothed=False, n_restarts=2, seed=7
    )
    assert isinstance(result, dict)
    assert "beta" in result
    assert "method" in result
    assert result["smoothed"] is False
