"""Tests for morie.fn.local_mean_variance.

Brus, D. J. (2022). Spatial Sampling with R, eq. (9.10).
Inputs are chosen so the expected value is exact by hand:
  e/pi = 4, 8, 12; minus ebar 3 gives 1, 5, 9; squares 1, 25, 81
  times (1-pi) = 0.5: 0.5 + 12.5 + 40.5 = 53.5
  factor (n/(n-p))(p/(p+1)) = (6/4)(2/3) = 1, so V = 53.5
"""

import pytest

from morie.fn.local_mean_variance import local_mean_variance


def test_local_mean_variance_matches_the_book_equation():
    r = local_mean_variance([2.0, 4.0, 6.0], [0.5, 0.5, 0.5],
                              [3.0, 3.0, 3.0], 6, 2)
    assert r["value"] == pytest.approx(53.5, abs=1e-12)


def test_local_mean_variance_is_zero_when_residuals_match_their_local_mean():
    r = local_mean_variance([2.0, 4.0, 6.0], [0.5, 0.5, 0.5],
                            [4.0, 8.0, 12.0], 6, 2)
    assert r["value"] == pytest.approx(0.0, abs=1e-12)


def test_local_mean_variance_rejects_bad_input():
    with pytest.raises(ValueError):
        local_mean_variance([2.0, 4.0], [0.5, 0.5], [3.0, 3.0], 2, 2)  # n <= p
    with pytest.raises(ValueError):
        local_mean_variance([2.0, 4.0], [0.0, 0.5], [3.0, 3.0], 6, 2)  # pi = 0
