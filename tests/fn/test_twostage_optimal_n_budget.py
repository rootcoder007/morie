"""Tests for morie.fn.twostage_optimal_n_budget.

Brus, D. J. (2022). Spatial Sampling with R, eq. (7.11).
Inputs are chosen so the expected value is exact by hand:
  n = C_max S_b / (S_w sqrt(c1 c2) + S_b c1)
    = 1000*4 / (8*sqrt(9) + 4*9) = 4000 / (24 + 36) = 4000/60
"""

import pytest

from morie.fn.twostage_optimal_n_budget import twostage_optimal_n_budget


def test_twostage_optimal_n_budget_matches_the_book_equation():
    r = twostage_optimal_n_budget(8.0, 4.0, 9.0, 1.0, 1000.0)
    assert r["value"] == pytest.approx(4000.0 / 60.0, abs=1e-12)


def test_twostage_optimal_n_budget_is_proportional_to_the_budget():
    a = twostage_optimal_n_budget(8.0, 4.0, 9.0, 1.0, 1000.0)["value"]
    b = twostage_optimal_n_budget(8.0, 4.0, 9.0, 1.0, 2000.0)["value"]
    assert b == pytest.approx(2 * a, abs=1e-12)


def test_twostage_optimal_n_budget_rejects_nonpositive_input():
    with pytest.raises(ValueError):
        twostage_optimal_n_budget(8.0, 4.0, 9.0, 1.0, 0.0)
