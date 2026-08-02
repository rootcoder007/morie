"""Tests for medfm.mediation_formula."""

from morie.fn import _array_core as np
import pytest

from morie.fn.medfm import mediation_formula


def test_medfm_basic():
    # discrete chain X -> M -> Y with no direct edge: NDE must be ~0
    rng = np.random.default_rng(42)
    n = 6000
    x = (rng.random(n) < 0.5).astype(int)
    m = (rng.random(n) < 0.2 + 0.6 * x).astype(int)
    y = (rng.random(n) < 0.1 + 0.7 * m).astype(int)
    out = mediation_formula(x, m, y, x1=1, x0=0)
    assert out["nde"] == pytest.approx(0.0, abs=0.05)
    assert out["nie"] == pytest.approx(0.6 * 0.7, abs=0.06)
    assert out["te"] == pytest.approx(out["nde"] + out["nie"])


def test_medfm_edge():
    rng = np.random.default_rng(0)
    n = 4000
    x = (rng.random(n) < 0.5).astype(int)
    m = (rng.random(n) < 0.5).astype(int)  # mediator unrelated to x
    y = (rng.random(n) < 0.2 + 0.5 * x).astype(int)
    out = mediation_formula(x, m, y, x1=1, x0=0)
    assert out["nie"] == pytest.approx(0.0, abs=0.05)  # nothing flows through m
    assert out["nde"] == pytest.approx(0.5, abs=0.06)
