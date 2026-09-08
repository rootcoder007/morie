"""Tests for evgofg -- Anderson-Darling A^2 for a fitted GPD.

Anchored on the xi = 0 limit, where Coles (2001) eq (4.2) collapses to
the exponential cdf 1 - exp(-y/sigma), written out by hand here.
"""

import math

import pytest

from morie.fn.evgofg import evt_gpd_anderson_darling, gpd_cdf
from morie.fn.hedderich7e33 import ad_statistic

Y = [0.12, 0.35, 0.61, 0.94, 1.28, 1.77, 2.40, 3.31, 4.62, 6.90]


def test_zero_shape_is_the_exponential():
    for y in (0.25, 1.0, 2.0, 5.0):
        assert float(gpd_cdf([y], 2.0, 0.0)[0]) == pytest.approx(
            1.0 - math.exp(-y / 2.0), abs=1e-15
        )


def test_cdf_is_monotone_and_in_the_unit_interval():
    for xi in (-0.4, 0.0, 0.3):
        vals = [float(gpd_cdf([y], 2.0, xi)[0]) for y in [0.1, 0.5, 1, 2, 4, 8, 16]]
        assert all(0.0 <= v <= 1.0 for v in vals)
        assert all(b >= a for a, b in zip(vals, vals[1:]))


def test_statistic_is_the_shared_ad_core_on_the_pit():
    res = evt_gpd_anderson_darling(Y, 2.0, 0.3)
    assert res["statistic"] == pytest.approx(ad_statistic(gpd_cdf(Y, 2.0, 0.3)))
    assert res["n"] == len(Y)


def test_excesses_must_be_positive():
    with pytest.raises(ValueError):
        evt_gpd_anderson_darling([0.0, 1.0, 2.0], 1.0, 0.0)
    with pytest.raises(ValueError):
        evt_gpd_anderson_darling([-1.0, 1.0], 1.0, 0.0)


def test_negative_shape_has_a_finite_upper_endpoint():
    # xi < 0 bounds the excess at -sigma/xi = 2/0.4 = 5
    assert float(gpd_cdf([100.0], 2.0, -0.4)[0]) == 1.0


def test_rejects_nonpositive_scale():
    with pytest.raises(ValueError):
        gpd_cdf([1.0], 0.0, 0.0)
