"""Tests for botUp.bottom_up_aggregation."""

from morie.fn import _array_core as np

from morie.fn.botUp import bottom_up_aggregation


def test_botUp_basic():
    """Test basic functionality.

    Bottom-level forecasts of length m=3 are reconciled through a
    summing matrix S (k x m) whose rows tell which bottom series
    contribute to each aggregated series (Hyndman & Athanasopoulos,
    FPP3, Section 11.3: ytilde = S @ yhat_bottom).
    """
    bottoms = [10.0, 20.0, 30.0]
    S = [
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ]
    result = bottom_up_aggregation(bottoms, S)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "total" in result
    assert "bottom" in result
    assert "n" in result
    assert "method" in result

    # Independent reference computation of S @ bottoms.
    expected_total = sum(bottoms)
    expected_estimate = [sum(S[i][j] * bottoms[j] for j in range(len(bottoms)))
                         for i in range(len(S))]

    assert list(result["bottom"]) == bottoms
    assert result["total"] == expected_total
    assert list(result["estimate"]) == expected_estimate
    assert result["n"] == len(S)


def test_botUp_edge():
    """Edge case: two bottom-level series and a single total above them."""
    bottoms = [5.0, 7.0]
    S = [
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    result = bottom_up_aggregation(bottoms, S)
    assert isinstance(result, dict)
    assert "estimate" in result

    expected_estimate = [sum(S[i][j] * bottoms[j] for j in range(len(bottoms)))
                         for i in range(len(S))]
    assert list(result["estimate"]) == expected_estimate
    assert result["total"] == sum(bottoms)
    assert result["n"] == len(S)
