"""Tests for dvres.deviance_residual_cox."""
import math

from morie.fn import _array_core as np

from morie.fn.dvres import deviance_residual_cox
from morie.fn.efrnt import efron_tie_correction


def test_dvres_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    p = 3
    X = rng.normal(0, 1, (n, p))
    t = rng.uniform(0, 10, n)
    e = rng.integers(0, 2, n)
    fit = efron_tie_correction(t, e, X)
    result = deviance_residual_cox(fit)
    assert isinstance(result, dict)
    for key in ("residuals", "martingale", "n_extreme", "mean", "sd"):
        assert key in result
    assert len(result["residuals"]) == n
    assert len(result["martingale"]) == n
    assert isinstance(result["n_extreme"], int)
    assert math.isfinite(result["mean"])
    assert math.isfinite(result["sd"])
    assert result["n_extreme"] >= 0


def test_dvres_edge():
    """Test edge cases."""
    rng = np.random.default_rng(7)
    n = 10
    p = 3
    X = rng.normal(0, 1, (n, p))
    t = rng.uniform(0, 10, n)
    e = rng.integers(0, 2, n)
    fit = efron_tie_correction(t, e, X)
    result = deviance_residual_cox(fit)
    assert isinstance(result, dict)
    for key in ("residuals", "martingale", "n_extreme", "mean", "sd"):
        assert key in result
    assert len(result["residuals"]) == n
    assert len(result["martingale"]) == n
    assert isinstance(result["n_extreme"], int)
    assert math.isfinite(result["mean"])
    assert math.isfinite(result["sd"])
