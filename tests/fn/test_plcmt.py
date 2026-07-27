"""Tests for plcmt.rank_placements."""

import numpy as np
import pytest

from morie.fn.plcmt import rank_placements


def test_plcmt_hand_computed_placements():
    """x = (1, 3), y = (2, 4): placements of y among x are (1, 2), so
    U_y = 3; under H0 E[U] = mn/2 = 2."""
    r = rank_placements(np.array([1.0, 3.0]), np.array([2.0, 4.0]))
    assert float(r["U_y"]) == pytest.approx(3.0, abs=1e-12)
    assert float(r["E_U"]) == pytest.approx(2.0, abs=1e-12)


def test_plcmt_u_matches_the_mann_whitney_count():
    """U_y counts pairs with y > x -- cross-check by brute force."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=15)
    y = rng.normal(0.5, 1.0, 12)
    r = rank_placements(x, y)
    brute = sum((yy > xx) for yy in y for xx in x)
    assert float(r["U_y"]) == pytest.approx(float(brute), abs=1e-9)


def test_plcmt_variance_matches_the_null_formula():
    """Var U = mn(m+n+1)/12 without ties."""
    x = np.arange(1.0, 9.0)          # m = 8
    y = np.arange(1.5, 7.0)          # n = 6, no ties with x
    r = rank_placements(x, y)
    m, n = 8, 6
    assert float(r["Var_U"]) == pytest.approx(m * n * (m + n + 1) / 12, rel=1e-9)
