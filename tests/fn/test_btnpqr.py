"""Tests for btnpqr.boot_quantile_regression."""

import math
import pytest
from morie.fn import _array_core as np
from morie.fn.btnpqr import boot_quantile_regression


def test_btnpqr_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 3))
    y = rng.normal(0, 1, 40)
    tau = 0.5
    B = 10
    alpha = 0.05
    seed = 42
    result = boot_quantile_regression(X, y, tau=tau, B=B, alpha=alpha, seed=seed)
    assert isinstance(result, dict)
    expected_keys = (
        "beta_b", "beta_hat", "se", "lo", "hi", "loss",
        "tau", "n", "p", "B", "estimate", "method",
    )
    for key in expected_keys:
        assert key in result
    assert result["n"] == 40
    assert result["p"] == 3
    assert result["B"] == B
    assert result["tau"] == tau
    assert len(result["beta_b"]) == B
    assert len(result["se"]) == 3
    assert len(result["lo"]) == 3
    assert len(result["hi"]) == 3
    assert math.isfinite(result["loss"])


def test_btnpqr_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 3))
    y = rng.normal(0, 1, 40)
    # tau must be strictly between 0 and 1
    with pytest.raises(ValueError):
        boot_quantile_regression(X, y, tau=0.0, B=10, alpha=0.05, seed=1)
