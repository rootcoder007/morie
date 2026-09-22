"""Tests for gh_c10_12.ghosal_modsel_bic."""

import math

from morie.fn import _array_core as np

from morie.fn.gh_c10_12 import ghosal_modsel_bic


def test_gh_c10_12_basic():
    """Test basic functionality under H1: log-BF should be positive."""
    n = 2000
    seed = 42
    result = ghosal_modsel_bic(truth_in_H1=True, n=n, seed=seed)
    assert "estimate" in result
    est = float(np.asarray(result["estimate"], dtype=float))
    assert math.isfinite(est)
    assert est > 0  # BF -> infinity under H1, so log-BF > 0


def test_gh_c10_12_edge():
    """Test edge case under H0: log-BF should be negative."""
    n = 2000
    seed = 42
    result = ghosal_modsel_bic(truth_in_H1=False, n=n, seed=seed)
    assert "estimate" in result
    est = float(np.asarray(result["estimate"], dtype=float))
    assert math.isfinite(est)
    assert est < 0  # BF -> 0 under H0, so log-BF < 0
