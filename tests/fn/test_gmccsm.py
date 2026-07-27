"""Tests for gmccsm.g_methods_consistency."""

import numpy as np
import pytest

from morie.fn.gmccsm import g_methods_consistency


def test_gmccsm_basic():
    rng = np.random.default_rng(42)
    L = rng.normal(size=2500)
    A = (rng.random(2500) < 1 / (1 + np.exp(-1.5 * L))).astype(float)
    y = 2.0 * A + 1.5 * L + rng.normal(scale=0.5, size=2500)
    result = g_methods_consistency(y, A, L, tau=0.3)
    assert result["consistent"] is True  # measured max_divergence ~0.05
    for k in ("ate_gformula", "ate_ipw", "ate_aipw"):
        assert result[k] == pytest.approx(2.0, abs=0.4)
    assert result["max_divergence"] < 0.3


def test_gmccsm_edge():
    with pytest.raises(ValueError):
        g_methods_consistency([1.0, 2.0], [1, 0], [0.0, 1.0], tau=0.0)  # bad tau
    with pytest.raises(ValueError):
        g_methods_consistency([1.0, 2.0], [2, 0], [0.0, 1.0])  # non-binary A
