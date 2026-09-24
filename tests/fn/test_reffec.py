"""Tests for reffec.effective_reproduction."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.reffec import effective_reproduction


def test_reffec_basic():
    """Test basic functionality."""
    R0 = 2.5
    S = 70.0
    N = 100.0
    result = effective_reproduction(R0, S, N)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "Rt" in result
    assert "growing" in result
    assert "susceptible_fraction" in result
    assert "herd_immunity_threshold" in result
    assert "method" in result
    # Rt = R0 * S / N = 2.5 * 70 / 100 = 1.75
    assert math.isclose(result["estimate"], 1.75)
    assert math.isclose(result["Rt"], 1.75)
    # Rt > 1, so the outbreak is still growing
    assert result["growing"] == 1.0
    assert math.isclose(result["susceptible_fraction"], 0.7)
    # herd_immunity_threshold = 1 - 1/R0 = 1 - 0.4 = 0.6
    assert math.isclose(result["herd_immunity_threshold"], 0.6)


def test_reffec_edge():
    """Test edge cases."""
    # R0 = 0 yields Rt = 0, not growing, and an undefined herd-immunity
    # threshold (division by zero -> NaN).
    R0 = 0.0
    S = 50.0
    N = 100.0
    result = effective_reproduction(R0, S, N)
    assert isinstance(result, dict)
    assert math.isclose(result["estimate"], 0.0)
    assert math.isclose(result["Rt"], 0.0)
    assert result["growing"] == 0.0
    assert math.isclose(result["susceptible_fraction"], 0.5)
    assert math.isnan(result["herd_immunity_threshold"])

    # Invalid inputs raise ValueError per the docstring.
    with pytest.raises(ValueError):
        effective_reproduction(-1.0, 50.0, 100.0)
    with pytest.raises(ValueError):
        effective_reproduction(2.0, 50.0, 0.0)
    with pytest.raises(ValueError):
        effective_reproduction(2.0, -1.0, 100.0)
    with pytest.raises(ValueError):
        effective_reproduction(2.0, 200.0, 100.0)

    # Boundary case: S = N means the full population is susceptible, so
    # Rt == R0 and the herd-immunity threshold is correctly stated.
    result_full = effective_reproduction(3.0, 100.0, 100.0)
    assert math.isclose(result_full["Rt"], 3.0)
    assert result_full["growing"] == 1.0
    assert math.isclose(result_full["herd_immunity_threshold"], 2.0 / 3.0)
