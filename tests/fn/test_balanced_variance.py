"""Tests for morie.fn.balanced_variance.

Brus, D. J. (2022). Spatial Sampling with R, eq. (9.3).
Inputs are chosen so the expected value is exact by hand:
  e/pi = 4, 8, 12; squares 16, 64, 144; times c = 1 sums to 224
  n/(n-p) = 3/2, so 224*1.5 = 336; divided by N^2 = 100 gives 3.36
"""

import pytest

from morie.fn.balanced_variance import balanced_variance


def test_balanced_variance_matches_the_book_equation():
    r = balanced_variance([2.0, 4.0, 6.0], [0.5, 0.5, 0.5],
                           [1.0, 1.0, 1.0], 10.0, 1)
    assert r["value"] == pytest.approx(3.36, abs=1e-12)


def test_balanced_variance_scales_with_the_inverse_square_of_population_size():
    a = balanced_variance([2.0, 4.0, 6.0], [0.5, 0.5, 0.5],
                          [1.0, 1.0, 1.0], 10.0, 1)["value"]
    b = balanced_variance([2.0, 4.0, 6.0], [0.5, 0.5, 0.5],
                          [1.0, 1.0, 1.0], 20.0, 1)["value"]
    assert b == pytest.approx(a / 4, abs=1e-12)


def test_balanced_variance_rejects_bad_input():
    with pytest.raises(ValueError):
        balanced_variance([2.0], [0.5], [1.0], 10.0, 1)          # n <= p
    with pytest.raises(ValueError):
        balanced_variance([2.0, 4.0], [0.5, 0.0], [1.0, 1.0], 10.0, 1)
