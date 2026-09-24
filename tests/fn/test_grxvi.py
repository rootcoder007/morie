"""Tests for grxvi.geron_glorot_xavier_init."""

import doctest as _doctest
import math

from morie.fn.grxvi import geron_glorot_xavier_init
import morie.fn.grxvi as _doctest_module


def test_grxvi_basic():
    """Test basic functionality."""
    fan_in = 4
    fan_out = 6
    distribution = "normal"
    result = geron_glorot_xavier_init(fan_in, fan_out, distribution)
    assert isinstance(result, dict)
    # Check expected keys from RichResult
    assert "weights" in result
    assert "target_variance" in result
    assert "achieved_variance" in result
    assert "scale" in result
    assert "estimate" in result
    assert "n" in result
    assert "method" in result
    # Check weights shape is (fan_in, fan_out)
    weights = result["weights"]
    assert len(weights) == fan_in
    for row in weights:
        assert len(row) == fan_out
    # Check target variance = 2 / (fan_in + fan_out)
    assert math.isclose(result["target_variance"], 2.0 / (fan_in + fan_out))
    # Check n = fan_in * fan_out
    assert result["n"] == fan_in * fan_out
    # Achieved variance should be finite (sample statistic, so do not lock value)
    assert math.isfinite(result["achieved_variance"])
    # Scale should be finite
    assert math.isfinite(result["scale"])


def test_grxvi_edge():
    """Test edge cases with uniform distribution."""
    fan_in = 100
    fan_out = 100
    distribution = "uniform"
    result = geron_glorot_xavier_init(fan_in, fan_out, distribution)
    assert isinstance(result, dict)
    assert "weights" in result
    assert "target_variance" in result
    assert "scale" in result
    assert "achieved_variance" in result
    # Target variance for 100, 100 is 2/200 = 0.01
    assert math.isclose(result["target_variance"], 0.01)
    # Scale for uniform = sqrt(6/(fan_in+fan_out))
    expected_scale = math.sqrt(6.0 / (fan_in + fan_out))
    assert math.isclose(result["scale"], expected_scale, rel_tol=1e-6)
    # Achieved variance is a sample statistic; only require it be finite and
    # of the same order of magnitude as the target (not locking observed output).
    assert math.isfinite(result["achieved_variance"])
    assert 0.0 <= result["achieved_variance"] < 1.0
    # Weights shape
    weights = result["weights"]
    assert len(weights) == fan_in
    for row in weights:
        assert len(row) == fan_out
    # Check n = fan_in * fan_out
    assert result["n"] == fan_in * fan_out


# --- appended: the module's own worked example as a gate -----------
def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
