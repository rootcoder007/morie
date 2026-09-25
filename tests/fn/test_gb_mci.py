"""Tests for gb_mci.gibbons_mcnemar_ci."""

import math

import pytest

from morie.fn.gb_mci import mcnemarci


def test_gb_mci_basic():
    """Eq. (14.5.2): theta_12 - theta_21 = (b - c) / n with Wald variance
    (p12 + p21 - (p12 - p21)^2) / n; the H0 error drops the square."""
    r = mcnemarci([[30, 12], [5, 53]])
    n = 100.0
    p12, p21 = 12 / n, 5 / n
    se = math.sqrt((p12 + p21 - (p12 - p21) ** 2) / n)
    z = 1.959963984540054
    assert r["estimate"] == pytest.approx(0.07, rel=1e-14)
    assert (r["lower"], r["upper"]) == pytest.approx((0.07 - z * se, 0.07 + z * se), rel=1e-12)
    assert r["se_null"] == pytest.approx(math.sqrt((p12 + p21) / n), rel=1e-14)


def test_gb_mci_edge():
    with pytest.raises(ValueError, match="2 x 2"):
        mcnemarci([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError, match="alpha"):
        mcnemarci([[1, 2], [3, 4]], alpha=1.5)


