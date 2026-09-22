"""Tests for gb434.gibbons_ks_one_sided_dist."""

import math

from morie.fn import _array_core as np

from morie.fn.gb434 import gibbons_ks_one_sided_dist


def test_gb434_basic():
    """Test basic functionality."""
    c = 0.3
    n = 10
    result = gibbons_ks_one_sided_dist(c, n)
    assert isinstance(result, dict)
    # Documented keys per the docstring
    for key in ("sf", "cdf", "terms", "n", "c", "method"):
        assert key in result
    # Round-trip of the scalar inputs
    assert result["c"] == float(c)
    assert result["n"] == int(n)
    # cdf = 1 - sf
    assert math.isclose(result["cdf"], 1.0 - result["sf"], rel_tol=1e-12)
    # sf must lie in [0, 1]
    assert 0.0 <= result["sf"] <= 1.0
    # Number of terms accumulated: floor(n*(1-c)) + 1
    assert result["terms"] == int(math.floor(n * (1.0 - c))) + 1

    # Independent computation of the closed-form sum in the docstring:
    # P(D+_n >= c) = c * sum_{j=0..floor(n(1-c))} C(n,j)
    #     * (c + j/n)**(j-1) * (1 - c - j/n)**(n-j)
    total = 0.0
    for j in range(int(math.floor(n * (1.0 - c))) + 1):
        total += (
            math.comb(n, j)
            * (c + j / n) ** (j - 1)
            * (1.0 - c - j / n) ** (n - j)
        )
    expected_sf = min(1.0, max(0.0, c * total))
    assert math.isclose(result["sf"], expected_sf, rel_tol=1e-10)


def test_gb434_edge():
    """Test edge cases."""
    # c at the upper boundary: documented to return sf = 0, cdf = 1
    result_hi = gibbons_ks_one_sided_dist(1.0, 5)
    assert isinstance(result_hi, dict)
    assert result_hi["sf"] == 0.0
    assert result_hi["cdf"] == 1.0

    # c at the lower boundary: documented to return sf = 1, cdf = 0
    result_lo = gibbons_ks_one_sided_dist(0.0, 5)
    assert isinstance(result_lo, dict)
    assert result_lo["sf"] == 1.0
    assert result_lo["cdf"] == 0.0

    # n = 1, c in (0, 1): only j = 0 term contributes, so
    # sf = c * (c + 0)**(-1) * (1 - c)**(1) = (1 - c).
    n = 1
    c = 0.4
    result_one = gibbons_ks_one_sided_dist(c, n)
    assert isinstance(result_one, dict)
    assert math.isclose(result_one["sf"], 1.0 - c, rel_tol=1e-12)
    assert math.isclose(result_one["cdf"], c, rel_tol=1e-12)
    assert result_one["terms"] == 1
