"""Tests for icc1.icc_one_way."""

import math

from morie.fn import _array_core as np

from morie.fn.icc1 import icc_one_way


def test_icc1_basic():
    """Test basic functionality with a balanced design."""
    rng = np.random.default_rng(43)
    n = 10
    k = 4
    subject_effects = rng.normal(0, 1, n)
    y = []
    cluster = []
    for i in range(n):
        for j in range(k):
            y.append(subject_effects[i] + rng.normal(0, 0.5))
            cluster.append(i)
    result = icc_one_way(y, cluster)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert -1.0 <= result["estimate"] <= 1.0


def test_icc1_edge():
    """Test edge cases - minimal balanced design with 2 subjects, 2 ratings each."""
    rng = np.random.default_rng(43)
    n = 2
    k = 2
    subject_effects = rng.normal(0, 1, n)
    y = []
    cluster = []
    for i in range(n):
        for j in range(k):
            y.append(subject_effects[i] + rng.normal(0, 0.1))
            cluster.append(i)
    result = icc_one_way(y, cluster)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert -1.0 <= result["estimate"] <= 1.0
