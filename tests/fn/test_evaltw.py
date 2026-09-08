"""Tests for evaltw.e_value_unmeasured_confounding."""

import pytest

from morie.fn.evaltw import e_value_unmeasured_confounding


def test_evaltw_basic():
    out = e_value_unmeasured_confounding(2.0, 1.2, 3.3)
    assert out["evalue"] == pytest.approx(2 + 2**0.5)
    # the CI E-value uses the limit closer to the null (1.2 here)
    assert out["evalue_ci"] == pytest.approx(1.2 + (1.2 * 0.2) ** 0.5)
    assert out["evalue_ci"] < out["evalue"]


def test_evaltw_edge():
    # a CI crossing 1 cannot rule out the null: its E-value is 1
    crossing = e_value_unmeasured_confounding(2.0, 0.8, 4.0)
    assert crossing["evalue_ci"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        e_value_unmeasured_confounding(-1.0, 0.5, 2.0)
