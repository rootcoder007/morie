"""Tests for gh_c14_11.ghosal_py_powerlaw."""

import math

from morie.fn import _array_core as np

from morie.fn.gh_c14_11 import ghosal_py_powerlaw


def test_gh_c14_11_basic():
    """Test basic functionality against the literature formula."""
    n = 5000
    d = 0.5
    theta = 1.0
    seed = 42
    result = ghosal_py_powerlaw(n=n, d=d, theta=theta, seed=seed)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))

    # Documented formula: E K_n ~ (Gamma(theta+1) / (d * Gamma(theta+d))) * n^d
    expected_theory = (
        math.gamma(theta + 1.0) / (d * math.gamma(theta + d)) * float(n) ** d
    )
    K = float(result["estimate"])
    assert K > 0.0
    # K_n grows as a power law in n; should be in the same ballpark as theory.
    assert 0.2 * expected_theory <= K <= 2.0 * expected_theory


def test_gh_c14_11_edge():
    """Test edge cases."""
    result = ghosal_py_powerlaw(n=1, d=0.5, theta=1.0, seed=42)
    # A single draw from the PY process must start with at least one table.
    assert int(result["estimate"]) >= 1
    # Sanity check on documented keys.
    assert "theory" in result
    assert "ratio" in result
