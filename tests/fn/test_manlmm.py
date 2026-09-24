"""Tests for manlmm.ma_network_lme."""

import math

from morie.fn import _array_core as np

from morie.fn.manlmm import ma_network_lme


def test_manlmm_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    yi = rng.normal(0, 1, n)
    vi = [abs(x) + 0.1 for x in rng.normal(0, 1, n)]
    design = []
    for _ in range(n):
        a = int(rng.uniform(0, 3))
        b = int(rng.uniform(0, 3))
        if a == b:
            b = (b + 1) % 3
        design.append([a, b])
    result = ma_network_lme(yi, vi, design)
    assert isinstance(result, dict)
    for key in ("theta", "se_theta", "ranks", "tau2", "QE", "treatments", "n", "T"):
        assert key in result
    assert result["n"] == n
    assert result["T"] >= 2
    assert len(result["theta"]) == result["T"]
    assert len(result["se_theta"]) == result["T"]
    assert len(result["ranks"]) == result["T"]
    assert len(result["treatments"]) == result["T"]
    assert result["tau2"] >= 0.0
    assert math.isfinite(result["QE"])


def test_manlmm_edge():
    """Test edge cases."""
    rng = np.random.default_rng(7)
    n = 10
    yi = rng.normal(0, 1, n)
    vi = [abs(x) + 0.1 for x in rng.normal(0, 1, n)]
    design = []
    for _ in range(n):
        a = int(rng.uniform(0, 3))
        b = int(rng.uniform(0, 3))
        if a == b:
            b = (b + 1) % 3
        design.append([a, b])
    result = ma_network_lme(yi, vi, design)
    assert isinstance(result, dict)
    assert result["n"] == n
    assert result["T"] >= 2
    assert len(result["theta"]) == result["T"]
