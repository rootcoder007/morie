"""Tests for f_nested_ss.f_nested_ss."""

from morie.fn import _array_core as np

from morie.fn.f_nested_ss import f_nested_ss


def test_ca2e18_basic():
    """Test basic functionality."""
    ss_resid_restricted = 200.0
    ss_resid_full = 150.0
    k_full = 5
    k_restricted = 3
    n = 100
    result = f_nested_ss(ss_resid_restricted, ss_resid_full, k_full, k_restricted, n)
    assert isinstance(result, dict)
    assert "f" in result

    # Independent computation per Weisburd et al. (2022) eq. (2.18):
    # F = (SS_resid(R) - SS_resid(F)) / (k_full - k_restricted)  /  (SS_resid(F) / (n - k_full))
    df_num = k_full - k_restricted
    df_den = n - k_full
    ms_resid_full = ss_resid_full / df_den
    expected_f = (ss_resid_restricted - ss_resid_full) / df_num / ms_resid_full
    assert result["f"] == expected_f


def test_ca2e18_edge():
    """Test edge cases."""
    ss_resid_restricted = 100.0
    ss_resid_full = 100.0
    k_full = 4
    k_restricted = 2
    n = 50
    result = f_nested_ss(ss_resid_restricted, ss_resid_full, k_full, k_restricted, n)
    assert isinstance(result, dict)
    assert "f" in result

    # When restricted and full models have identical residual SS, F should be 0.
    assert result["f"] == 0.0
