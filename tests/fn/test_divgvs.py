"""Tests for divgvs.discriminant_validity."""

import math

from morie.fn import _array_core as np

from morie.fn.divgvs import discriminant_validity


def test_divgvs_basic():
    """Test basic functionality."""
    AVE = [0.50, 0.60, 0.70]
    factor_correlations = [
        [1.0, 0.30, 0.40],
        [0.30, 1.0, 0.50],
        [0.40, 0.50, 1.0],
    ]
    result = discriminant_validity(AVE, factor_correlations)
    assert isinstance(result, dict)
    assert "estimate" in result

    sqrt_ave = [math.sqrt(v) for v in AVE]
    margins = []
    for i in range(len(AVE)):
        for j in range(len(AVE)):
            if i == j:
                continue
            margins.append(sqrt_ave[i] - abs(factor_correlations[i][j]))
    expected_estimate = min(margins)
    assert result["estimate"] == expected_estimate

    assert result["sqrt_ave"] == sqrt_ave
    assert result["k"] == 3
    assert result["discriminant"] == 1
    assert result["n_violations"] == 0
    assert result["pass_factor"] == [1, 1, 1]


def test_divgvs_edge():
    """Test edge cases."""
    AVE = [0.50, 0.60, 0.70]
    factor_correlations = [
        [1.0, 0.30, 0.40],
        [0.30, 1.0, 0.50],
        [0.40, 0.50, 1.0],
    ]
    result = discriminant_validity(AVE, factor_correlations)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert result["k"] == 3
