"""Tests for grimp.geron_simple_imputer."""

import math

from morie.fn import _array_core as np
from morie.fn.grimp import geron_simple_imputer


def test_grimp_basic():
    """Test basic functionality with 2-D input and mean strategy."""
    rng = np.random.default_rng(42)
    X = []
    for i in range(40):
        row = []
        for j in range(3):
            v = float(rng.normal(0, 1))
            if float(rng.uniform(0, 1)) < 0.2:
                v = float("nan")
            row.append(v)
        X.append(row)

    result = geron_simple_imputer(X, "mean")
    assert isinstance(result, dict)
    assert "imputed" in result
    assert "statistics" in result
    assert "n_missing" in result
    assert result["n_missing"] > 0
    assert len(result["statistics"]) == 3


def test_grimp_edge():
    """Test edge case: mean and median strategies on a small input with NaN."""
    X = [[1.0], [2.0], [300.0], [float("nan")]]
    result = geron_simple_imputer(X, "mean")
    assert isinstance(result, dict)
    assert "imputed" in result
    assert "statistics" in result
    assert "n_missing" in result
    assert len(result["statistics"]) == 1
    assert math.isfinite(result["statistics"][0])

    result2 = geron_simple_imputer(X, "median")
    assert isinstance(result2, dict)
    assert "statistics" in result2
    assert len(result2["statistics"]) == 1
    assert math.isfinite(result2["statistics"][0])


def test_imputation_statistics_recomputed():
    import pytest

    nan = float("nan")
    X = [[1.0, 10.0], [nan, 12.0], [3.0, nan], [4.0, 30.0], [2.0, 12.0]]
    obs0 = [1.0, 3.0, 4.0, 2.0]
    obs1 = [10.0, 12.0, 30.0, 12.0]
    r = geron_simple_imputer(X, "mean")
    assert r["statistics"] == pytest.approx([sum(obs0) / 4, sum(obs1) / 4], rel=1e-15)
    assert r["imputed"][1][0] == pytest.approx(2.5, rel=1e-15)
    med = geron_simple_imputer(X, "median")
    assert med["statistics"] == pytest.approx([2.5, 12.0], rel=1e-15)
    assert geron_simple_imputer(X, "most_frequent")["statistics"][1] == 12.0
