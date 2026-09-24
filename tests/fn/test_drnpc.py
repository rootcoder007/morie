"""Tests for drnpc.dr_did_neg_control."""

import math

from morie.fn import _array_core as np

from morie.fn.drnpc import dr_did_neg_control


def test_drnpc_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 100
    y_main = rng.normal(0, 1, n)
    y_neg = rng.normal(0, 1, n)
    D = rng.integers(0, 2, n)
    if sum(D) == 0:
        D[0] = 1
    elif sum(D) == n:
        D[0] = 0
    X = rng.normal(0, 1, (n, 5))
    result = dr_did_neg_control(y_main, y_neg, D, X)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "tau_main" in result
    assert "tau_neg" in result
    assert "se_main" in result
    assert "se_neg" in result
    assert "z_neg" in result
    assert "crit" in result
    assert "falsified" in result
    assert "tau_adj" in result
    assert "n" in result
    assert result["n"] == n
    assert result["estimate"] == result["tau_main"]
    assert result["tau_adj"] == result["tau_main"] - result["tau_neg"]
    assert math.isfinite(result["crit"])
    assert result["falsified"] in (0.0, 1.0)


def test_drnpc_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 40
    y_main = rng.normal(0, 1, n)
    y_neg = rng.normal(0, 1, n)
    D = rng.integers(0, 2, n)
    if sum(D) == 0:
        D[0] = 1
    elif sum(D) == n:
        D[0] = 0
    result = dr_did_neg_control(y_main, y_neg, D, X=None, alpha=0.10)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "tau_main" in result
    assert "tau_neg" in result
    assert "se_main" in result
    assert "se_neg" in result
    assert "z_neg" in result
    assert "crit" in result
    assert "falsified" in result
    assert "tau_adj" in result
    assert "n" in result
    assert result["n"] == n
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["crit"])
    assert result["falsified"] in (0.0, 1.0)
