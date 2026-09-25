"""Tests for true_confidence_level (Bilder & Loughin 2025, eq. 1.6)."""

import math

import pytest

from morie.fn.true_confidence_level import true_confidence_level


def _wald(w, n):
    p = w / n
    h = 1.959963984540054 * math.sqrt(p * (1 - p) / n)
    return p - h, p + h


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo1e6_basic():
    """C(pi) = sum over w of I(pi in interval(w)) C(n,w) pi^w (1-pi)^(n-w),
    summed here directly for the Wald interval at n = 40, pi = 0.157."""
    n, p = 40, 0.157
    c = sum(math.comb(n, w) * p ** w * (1 - p) ** (n - w)
            for w in range(n + 1) if _wald(w, n)[0] <= p <= _wald(w, n)[1])
    assert true_confidence_level(n, p, _wald)["value"] == pytest.approx(c, abs=1e-14)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo1e6_edge():
    """An interval that always covers has level 1; invalid inputs raise."""
    assert true_confidence_level(10, 0.3, lambda w, n: (0.0, 1.0))["value"] == pytest.approx(1.0, abs=1e-14)
    with pytest.raises(ValueError):
        true_confidence_level(10, 1.0, _wald)
    with pytest.raises(ValueError):
        true_confidence_level(0, 0.5, _wald)
