"""Tests for f_nested_r2.f_nested_r2."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.f_nested_r2 import f_nested_r2


def test_ca2e19_basic():
    """Test basic functionality against Weisburd et al. (2022) eq. (2.19)."""
    # Plain inputs documenting the formula:
    # F = ((R2_F - R2_R)/(k_F - k_R)) / ((1 - R2_F)/(n - k_F - 1))
    r2_full = 0.35
    r2_restricted = 0.30
    k_full = 5
    k_restricted = 3
    n = 100

    expected_f = ((r2_full - r2_restricted) / (k_full - k_restricted)) \
        / ((1 - r2_full) / (n - k_full - 1))

    result = f_nested_r2(r2_full, r2_restricted, k_full, k_restricted, n)

    assert isinstance(result, dict)
    # The headline key is 'f' per the docstring.
    assert "f" in result
    assert np.isclose(result["f"], expected_f)
    # The full payload is also exposed under 'value'.
    assert np.isclose(result["value"], expected_f)


def test_ca2e19_edge():
    """Test that the function handles equal R^2 values (numerator zero)."""
    r2_full = 0.25
    r2_restricted = 0.25
    k_full = 4
    k_restricted = 2
    n = 50

    expected_f = ((r2_full - r2_restricted) / (k_full - k_restricted)) \
        / ((1 - r2_full) / (n - k_full - 1))

    result = f_nested_r2(r2_full, r2_restricted, k_full, k_restricted, n)

    assert isinstance(result, dict)
    assert "f" in result
    assert np.isclose(result["f"], expected_f)
    assert np.isclose(result["f"], 0.0)
