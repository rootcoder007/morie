"""Tests for gb_pg2.gibbons_page_asymp."""

import math

from morie.fn import _array_core as np

from morie.fn.gb_pg2 import gibbons_page_asymp


def test_gb_pg2_basic():
    """Test basic functionality against the literature formula."""
    k = 5
    n = 100
    L = 1234.5
    result = gibbons_page_asymp(L, k, n)
    # Result is a RichResult (mapping-like); check key names per docstring.
    assert "z" in result
    assert "p_value" in result
    assert "mean" in result
    assert "var" in result
    assert "statistic" in result
    assert "k" in result
    assert "n" in result
    assert "method" in result

    # Re-derive z, mean, var from the documented formula on the same inputs.
    e = L - 0.5  # default correct=True
    expected_z = (12.0 * e - 3.0 * k * n * (n + 1.0) ** 2) / (
        n * (n + 1.0) * math.sqrt(k * (n - 1.0))
    )
    expected_mean = k * n * (n + 1.0) ** 2 / 4.0
    expected_var = k * float(n) ** 2 * (n + 1.0) ** 2 * (n - 1.0) / 144.0

    assert result["z"] == expected_z
    assert result["mean"] == expected_mean
    assert result["var"] == expected_var
    assert result["statistic"] == L
    assert result["k"] == k
    assert result["n"] == n


def test_gb_pg2_no_continuity():
    """When correct=False, no 0.5 shift is applied."""
    k = 5
    n = 100
    L = 1234.5
    result = gibbons_page_asymp(L, k, n, correct=False)

    e = L  # no shift
    expected_z = (12.0 * e - 3.0 * k * n * (n + 1.0) ** 2) / (
        n * (n + 1.0) * math.sqrt(k * (n - 1.0))
    )
    assert result["z"] == expected_z


def test_gb_pg2_edge():
    """Edge values: minimum n and minimum positive variance case."""
    k = 1
    n = 2
    L = 0.75
    result = gibbons_page_asymp(L, k, n)
    assert "z" in result

    e = L - 0.5
    expected_z = (12.0 * e - 3.0 * k * n * (n + 1.0) ** 2) / (
        n * (n + 1.0) * math.sqrt(k * (n - 1.0))
    )
    assert result["z"] == expected_z
