"""Tests for mssm.marginal_structural_med."""

import numpy as np
import pytest

from morie.fn.mssm import marginal_structural_med


def test_mssm_basic():
    rng = np.random.default_rng(42)
    n = 3000
    c = rng.normal(size=n)
    x = (rng.random(n) < 1 / (1 + np.exp(-c))).astype(float)
    m = 0.8 * x + 0.5 * c + rng.normal(scale=0.6, size=n)
    y = 0.4 * x + 1.0 * m + 0.5 * c + rng.normal(scale=0.6, size=n)
    out = marginal_structural_med(x, m, y, c=c)
    assert out["nie"] == pytest.approx(0.8, abs=0.25)
    assert out["te"] == pytest.approx(out["nde"] + out["nie"])


def test_mssm_edge():
    with pytest.raises(ValueError):
        marginal_structural_med(np.zeros(20), np.zeros(20), np.zeros(20))  # one arm only
    with pytest.raises(ValueError):
        marginal_structural_med([0.5, 1.0], [1.0, 2.0], [1.0, 2.0])  # non-binary x
