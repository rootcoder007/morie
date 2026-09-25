"""Tests for shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a5e3 (Deshmukh and Kashikar eq. 5.3, independence of two variables)."""

import pytest

from morie.fn.shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a5e3 import (
    shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a_chapter_5_equation_3 as indrv2,
)


def test_shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a5e3_basic():
    """The outer product of margins (0.4, 0.6) and (0.3, 0.7) factorises;
    moving 0.05 of mass off the diagonal breaks it by exactly the largest
    |p_ij - p_i. p_.j|."""
    tab = [[0.12, 0.28], [0.18, 0.42]]
    r = indrv2(tab)
    assert r["margin_row"] == pytest.approx([0.4, 0.6], abs=1e-15)
    assert r["margin_col"] == pytest.approx([0.3, 0.7], abs=1e-15)
    assert r["independent"] is True
    d = indrv2([[0.17, 0.23], [0.13, 0.47]])
    rows, cols = [0.4, 0.6], [0.3, 0.7]
    want = max(abs(v - rows[i] * cols[j]) for i, row in enumerate([[0.17, 0.23], [0.13, 0.47]]) for j, v in enumerate(row))
    assert d["max_deviation"] == pytest.approx(want, abs=1e-15)
    assert d["independent"] is False


def test_shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a5e3_edge():
    """A table that does not sum to one raises."""
    with pytest.raises(ValueError):
        indrv2([[0.5, 0.2], [0.1, 0.1]])
