"""Tests for morie.fn.twostage_variance_components.

Brus, D. J. (2022). Spatial Sampling with R, eq. (7.3).
Inputs are chosen so the expected value is exact by hand:
  V = S2_b/n + S2_w/(n m) = 40/5 + 60/(5*3) = 8 + 4 = 12
"""

import pytest

from morie.fn.twostage_variance_components import twostage_variance_components


def test_twostage_variance_components_matches_the_book_equation():
    r = twostage_variance_components(40.0, 60.0, 5, 3)
    assert r["value"] == pytest.approx(12.0, abs=1e-12)


def test_twostage_variance_components_within_term_shrinks_with_more_units():
    # doubling m halves only the within-PSU contribution
    a = twostage_variance_components(40.0, 60.0, 5, 3)["value"]
    b = twostage_variance_components(40.0, 60.0, 5, 6)["value"]
    assert a - b == pytest.approx(2.0, abs=1e-12)


def test_twostage_variance_components_rejects_bad_input():
    with pytest.raises(ValueError):
        twostage_variance_components(-1.0, 60.0, 5, 3)
