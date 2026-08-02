"""Tests for engrgr."""

from morie.fn import _array_core as np
import pytest

from morie.fn.engrgr import engle_granger

def test_engrgr_basic():
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.standard_normal(300))
    y = -1.5 * x + rng.standard_normal(300)
    out = engle_granger(y, x)
    assert out["beta"][0] == pytest.approx(-1.5, abs=0.15)
    assert out["n_vars"] == 2


def test_engrgr_edge():
    rng = np.random.default_rng(2)
    x = np.cumsum(rng.standard_normal(300))
    # MacKinnon k=2 value is stricter than the plain ADF k=1 one
    assert engle_granger(2 * x + rng.standard_normal(300), x)[
        "critical_values"
    ][0.05] == pytest.approx(-3.33613)
    with pytest.raises(ValueError):
        engle_granger(np.ones(50), np.ones(30))
