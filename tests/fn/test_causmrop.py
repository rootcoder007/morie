"""Tests for causmrop.causal_robins_g_formula."""

import numpy as np
import pytest

from morie.fn.causmrop import causal_robins_g_formula


def test_causmrop_basic():
    rng = np.random.default_rng(42)
    L = rng.normal(size=2500)
    A = (rng.random(2500) < 1 / (1 + np.exp(-1.5 * L))).astype(float)
    y = 2.0 * A + 1.5 * L + rng.normal(scale=0.5, size=2500)
    result = causal_robins_g_formula(y, A, L)
    assert result["ate"] == pytest.approx(2.0, abs=0.15)  # measured ~2.01
    assert result["ate"] == pytest.approx(result["EY1"] - result["EY0"])


def test_causmrop_edge():
    with pytest.raises(ValueError):
        causal_robins_g_formula([1.0, 2.0], [0.5, 1.0], [0.0, 1.0])  # non-binary A
    with pytest.raises(ValueError):
        causal_robins_g_formula([1.0, 2.0], [1, 1], [0.0, 1.0])  # no controls
