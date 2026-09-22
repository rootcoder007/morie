"""Tests for dprcl.dp_release_calibration."""

from morie.fn import _array_core as np

from morie.fn.dprcl import dp_release_calibration


def test_dprcl_basic():
    """Test basic functionality."""
    sensitivity = 1.0
    epsilon = 1e-6
    confidence = 0.95
    n = 1000
    result = dp_release_calibration(
        sensitivity=sensitivity, epsilon=epsilon,
        confidence=confidence, n=n,
    )
    assert isinstance(result, dict)
    assert "epsilon" in result
    assert "half_width" in result
    assert "noise_scale" in result
    assert "direction" in result

    # Recompute expected values from the documented formula.
    z = np.log(1.0 / (1.0 - confidence))
    expected_eps = float(epsilon)
    expected_w = sensitivity * z / (n * expected_eps)
    expected_b = sensitivity / (n * expected_eps)

    assert result["epsilon"] == expected_eps
    assert abs(result["half_width"] - expected_w) < 1e-12
    assert abs(result["noise_scale"] - expected_b) < 1e-12
    assert result["direction"] == "epsilon -> error"


def test_dprcl_edge():
    """Test edge cases."""
    sensitivity = 1.0
    target_error = 0.01
    confidence = 0.95
    n = 1000
    result = dp_release_calibration(
        sensitivity=sensitivity, target_error=target_error,
        confidence=confidence, n=n,
    )
    assert isinstance(result, dict)
    assert "epsilon" in result
    assert "half_width" in result

    # Round-trip: calibrate to an error, then read the error back.
    z = np.log(1.0 / (1.0 - confidence))
    expected_eps = sensitivity * z / (n * target_error)
    assert abs(result["epsilon"] - expected_eps) < 1e-12
    assert abs(result["half_width"] - target_error) < 1e-12
    assert result["direction"] == "error -> epsilon"
