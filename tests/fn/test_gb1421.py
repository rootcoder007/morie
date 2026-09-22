"""Tests for gb1421.gibbons_chisq_contingency."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.gb1421 import gibbons_chisq_contingency


def test_gb1421_basic():
    """Test basic functionality with a 2x2 contingency table."""
    # Build a small 2x2 contingency table of non-negative integer counts.
    table = [
        [10, 20],
        [30, 40],
    ]
    result = gibbons_chisq_contingency(table)
    assert isinstance(result, dict)
    assert "statistic" in result
    assert "df" in result
    assert "p_value" in result
    assert "expected" in result
    assert "n" in result
    assert "r" in result
    assert "c" in result
    assert "method" in result

    # Dimensions
    assert result["r"] == 2
    assert result["c"] == 2
    assert result["df"] == (2 - 1) * (2 - 1)
    assert result["n"] == float(10 + 20 + 30 + 40)

    # Expected frequencies: e_ij = row_i * col_j / N
    N = 10 + 20 + 30 + 40
    r1, r2 = 10 + 20, 30 + 40
    c1, c2 = 10 + 30, 20 + 40
    exp00 = r1 * c1 / N
    exp01 = r1 * c2 / N
    exp10 = r2 * c1 / N
    exp11 = r2 * c2 / N
    assert result["expected"] == [
        [exp00, exp01],
        [exp10, exp11],
    ]

    # Pearson Q = sum_ij (f_ij - e_ij)^2 / e_ij
    f = [[10.0, 20.0], [30.0, 40.0]]
    e = [[exp00, exp01], [exp10, exp11]]
    q = sum((f[i][j] - e[i][j]) ** 2 / e[i][j] for i in range(2) for j in range(2))
    assert result["statistic"] == q


def test_gb1421_edge():
    """Test that a 2x2 table with Yates's correction is applied only then."""
    table = [
        [5, 10],
        [15, 20],
    ]
    result = gibbons_chisq_contingency(table, correct=True)
    assert isinstance(result, dict)
    assert "statistic" in result
    assert "df" in result
    assert "p_value" in result
    assert "expected" in result

    # Yates-corrected 2x2 statistic:
    # Q_y = sum_ij (|f_ij - e_ij| - 0.5)^2 / e_ij
    N = 5 + 10 + 15 + 20
    r1, r2 = 5 + 10, 15 + 20
    c1, c2 = 5 + 15, 10 + 20
    exp = [
        [r1 * c1 / N, r1 * c2 / N],
        [r2 * c1 / N, r2 * c2 / N],
    ]
    f = [[5.0, 10.0], [15.0, 20.0]]
    q_y = 0.0
    for i in range(2):
        for j in range(2):
            d = abs(f[i][j] - exp[i][j]) - 0.5
            if d < 0.0:
                d = 0.0
            q_y += d * d / exp[i][j]
    assert result["statistic"] == q_y
