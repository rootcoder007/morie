"""Tests for postrt.post_stratification."""

import math

from morie.fn import _array_core as np

from morie.fn.postrt import post_stratification


def test_postrt_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(43)
    y = rng.normal(0, 1, 100)
    weights = rng.uniform(0.1, 2.0, 100)
    stratum = [str(v) for v in rng.integers(0, 3, 100)]
    N_h = {"0": 200, "1": 150, "2": 180}
    result = post_stratification(y, weights, stratum, N_h)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "weights" in result
    assert "factors" in result
    assert math.isfinite(result["estimate"])
    assert len(result["weights"]) == 100


def test_postrt_edge():
    """Test edge cases - small input with known calibration property."""
    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    weights = [1.0, 1.0, 1.0, 1.0, 1.0]
    stratum = ["0", "0", "1", "1", "2"]
    N_h = {"0": 10, "1": 10, "2": 5}
    result = post_stratification(y, weights, stratum, N_h)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert len(result["weights"]) == 5
    assert len(result["factors"]) == 3
    # With unit design weights, adjusted weights sum to N_h per stratum
    assert abs(sum(result["weights"][0:2]) - 10.0) < 1e-9
    assert abs(sum(result["weights"][2:4]) - 10.0) < 1e-9
    assert abs(result["weights"][4] - 5.0) < 1e-9
