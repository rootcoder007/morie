"""Tests for mhrate.mantel_haenszel_rate."""

from morie.fn import _array_core as np
import math
import pytest

from morie.fn.mhrate import mantel_haenszel_rate


def test_mhrate_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n_strata = 3
    strata = []
    for _ in range(n_strata):
        a = int(rng.integers(1, 20))
        b = int(rng.integers(1, 20))
        T1 = float(rng.uniform(100, 1000))
        T0 = float(rng.uniform(100, 1000))
        strata.append((a, T1, b, T0))
    result = mantel_haenszel_rate(strata)
    assert isinstance(result, dict)
    for key in ("estimate", "ln_estimate", "se_ln", "ci_lower", "ci_upper",
                "numerator", "denominator", "n_strata", "confidence", "method"):
        assert key in result
    assert result["n_strata"] == n_strata
    assert math.isfinite(result["estimate"])
    assert result["estimate"] > 0
    assert result["ci_lower"] <= result["estimate"] <= result["ci_upper"]
    assert result["confidence"] == 0.95


def test_mhrate_edge():
    """Test edge cases."""
    with pytest.raises(ValueError):
        mantel_haenszel_rate([])
