"""Tests for ecsTCR.ecs_tcr."""

from morie.fn import _array_core as np

from morie.fn.ecsTCR import ecs_tcr


def test_ecsTCR_basic():
    """Test basic functionality with a provided climate feedback parameter."""
    lam = 1.2  # W m-2 K-1, climate feedback parameter
    result = ecs_tcr(lam=lam)
    assert isinstance(result, dict)
    # Core output keys that the function guarantees.
    assert "ecs" in result
    assert "tcr" in result
    assert "lambda" in result
    assert "f2x" in result
    assert "temperature" in result
    # The function does not return an "estimate" or "statistic" key.
    assert "estimate" not in result
    assert "statistic" not in result
    # ECS equals f2x / lam under the documented formula.
    expected_ecs = result["f2x"] / lam
    assert abs(result["ecs"] - expected_ecs) < 1e-12


def test_ecsTCR_edge():
    """Test edge cases with a supplied CO2 trajectory."""
    # Supply a CO2 trajectory that starts at 1.0 and reaches >= 2.0
    # within the integration window, as the function requires.
    CO2_traj = [1.0 + 0.01 * (i + 1) for i in range(70)]
    lam = 1.2
    result = ecs_tcr(CO2_traj=CO2_traj, lam=lam)
    assert isinstance(result, dict)
    assert "ecs" in result
    assert "tcr" in result
    # With route='parameters', no fitting is performed and the supplied
    # feedback parameter is echoed back.
    assert result["lambda"] == lam
