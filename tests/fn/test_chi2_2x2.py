"""Tests for chi2_2x2.chi2_2x2."""

from morie.fn import _array_core as np

from morie.fn.chi2_2x2 import chi2_2x2


def test_ca9e4_basic():
    """Test basic functionality."""
    # chi2_2x2 takes four non-negative integer frequencies (a, b, c, d)
    # forming a 2x2 contingency table:
    #     | col1 | col2
    # -----------------
    # row1|  a   |  b
    # row2|  c   |  d
    a, b, c, d = 10, 20, 30, 40
    result = chi2_2x2(a, b, c, d)

    # The function returns a RichResult (dict subclass).
    assert isinstance(result, dict)

    # The headline key is 'chi2' (the statistic), with the same value
    # stored under 'value'.
    assert "chi2" in result
    assert "value" in result
    assert result["chi2"] == result["value"]

    # Independent computation of the Weisburd et al. (2022) eq. (9.4)
    # chi-square statistic for a 2x2 table: (ad - bc)^2 * N /
    # [(a+b)(c+d)(a+c)(b+d)]
    N = a + b + c + d
    expected_chi2 = (a * d - b * c) ** 2 * N / (
        (a + b) * (c + d) * (a + c) * (b + d)
    )
    assert result["chi2"] == expected_chi2

    # The method should be documented in the payload.
    assert "method" in result
    assert "Weisburd" in result["method"]


def test_ca9e4_edge():
    """Test edge cases."""
    # Use small non-negative integer frequencies.
    a, b, c, d = 1, 1, 1, 1
    result = chi2_2x2(a, b, c, d)

    assert isinstance(result, dict)
    assert "chi2" in result

    # With a 1:1:1:1 table, ad - bc == 0, so chi2 must be 0.0.
    N = a + b + c + d
    expected_chi2 = (a * d - b * c) ** 2 * N / (
        (a + b) * (c + d) * (a + c) * (b + d)
    )
    assert result["chi2"] == expected_chi2
    assert result["chi2"] == 0.0
