"""Tests for shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a5e1 (Deshmukh and Kashikar eq. 5.1, independence of k events)."""

import pytest

from morie.fn.shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a5e1 import (
    shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a_chapter_5_equation_1 as indevk,
)


def test_shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a5e1_basic():
    """Two fair coins, A1 = head on the first, A2 = head on the second,
    A3 = the coins agree: every pair meets P(AB) = 1/4 = P(A)P(B) but
    P(A1 A2 A3) = 1/4 != 1/8, so the three are pairwise but not mutually
    independent; 2^3 - 3 - 1 = 4 conditions are checked."""
    joint = [1.0, 0.5, 0.5, 0.25, 0.5, 0.25, 0.25, 0.25]
    r = indevk([0.5, 0.5, 0.5], joint)
    assert r["n_conditions"] == 4 and r["k"] == 3
    assert r["max_deviation"] == pytest.approx(0.125, abs=1e-15)
    assert r["independent"] is False
    joint[7] = 0.125
    assert indevk([0.5, 0.5, 0.5], joint)["independent"] is True


def test_shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a5e1_edge():
    """The joint table must have one entry per subset mask."""
    with pytest.raises(ValueError):
        indevk([0.5, 0.5], [1.0, 0.5, 0.5])
