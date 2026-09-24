"""Tests for chrwgt.censoring_at_risk_weight."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.chrwgt import censoring_at_risk_weight


def test_chrwgt_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    T = rng.uniform(0.1, 10.0, n)
    C = rng.uniform(0.1, 10.0, n)
    time = [min(T[i], C[i]) for i in range(n)]
    censor = [1.0 if C[i] < T[i] else 0.0 for i in range(n)]
    result = censoring_at_risk_weight(time, censor)
    assert isinstance(result, dict)
    for key in ("weights", "G", "max_weight_share", "ess", "n_censored"):
        assert key in result
    assert len(result["weights"]) == n
    assert len(result["G"]) == n
    assert result["n_censored"] == int(sum(censor))
    assert math.isfinite(result["ess"])
    assert 0.0 <= result["max_weight_share"] <= 1.0


def test_chrwgt_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 20
    T = rng.uniform(0.1, 5.0, n)
    C = rng.uniform(0.1, 5.0, n)
    time = [min(T[i], C[i]) for i in range(n)]
    censor = [1.0 if C[i] < T[i] else 0.0 for i in range(n)]
    result = censoring_at_risk_weight(time, censor, at=2.0, stabilize=False)
    assert isinstance(result, dict)
    assert "weights" in result
    assert len(result["weights"]) == n
    with pytest.raises(ValueError):
        censoring_at_risk_weight([1.0, 2.0], [0.0, 2.0])
