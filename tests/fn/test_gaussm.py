"""Tests for gaussm.gaussian_mechanism."""

import math

from morie.fn import _array_core as np

from morie.fn.gaussm import gaussian_mechanism


def test_gaussm_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    f_value = rng.normal(0, 1, 5)
    l2_sens = 1.5
    epsilon = 0.5
    delta = 1e-5
    result = gaussian_mechanism(f_value, l2_sens, epsilon, delta)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "released" in result
    assert "noise" in result
    assert "sigma" in result
    assert "c" in result
    assert "epsilon" in result
    assert "delta" in result
    assert "n" in result
    assert "method" in result

    # Independent computation of expected sigma and c from documented formula.
    c_expected = math.sqrt(2.0 * math.log(1.25 / delta))
    sigma_expected = c_expected * l2_sens / epsilon

    assert result["c"] == c_expected
    assert result["sigma"] == sigma_expected
    assert result["epsilon"] == epsilon
    assert result["delta"] == delta
    assert result["n"] == len(f_value)
    assert len(result["released"]) == len(f_value)
    assert len(result["noise"]) == len(f_value)

    # estimate must equal released[0] per implementation.
    assert result["estimate"] == result["released"][0]
    # released must equal f_value + noise, component-wise.
    for i in range(len(f_value)):
        assert result["released"][i] == f_value[i] + result["noise"][i]


def test_gaussm_edge():
    """Test edge cases: scalar/empty handling and validation."""
    rng = np.random.default_rng(42)
    f_value = rng.normal(0, 1, 3)
    l2_sens = 2.0
    epsilon = 0.1
    delta = 1e-6
    result = gaussian_mechanism(f_value, l2_sens, epsilon, delta)
    assert isinstance(result, dict)
    assert "released" in result
    assert "sigma" in result

    c_expected = math.sqrt(2.0 * math.log(1.25 / delta))
    sigma_expected = c_expected * l2_sens / epsilon
    assert result["sigma"] == sigma_expected
    assert result["n"] == 3
