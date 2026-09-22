"""Tests for gb1221.gibbons_friedman."""

from morie.fn import _array_core as np

from morie.fn.gb1221 import gibbons_friedman


def test_gb1221_basic():
    """Test basic functionality with a small k x n table."""
    # k=3 blocks, n=4 treatments, no ties -- the documented shape
    data = [
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 1.0],
        [4.0, 1.0, 2.0, 3.0],
    ]
    result = gibbons_friedman(data)
    assert isinstance(result, dict)
    assert "statistic" in result
    assert "p_value" in result
    assert "rank_sums" in result

    # Independent computation of Q from eq. (12.2.8) using midranks.
    # Within each block (row), values are all distinct, so ranks are
    # ordinary ranks 1..4 with rank sums per treatment:
    #   block 1: values [1,2,3,4] -> ranks [1,2,3,4]
    #   block 2: values [2,3,4,1] -> ranks [2,3,4,1]
    #   block 3: values [4,1,2,3] -> ranks [4,1,2,3]
    # Column rank sums: col0 = 1+2+4=7, col1=2+3+1=6, col2=3+4+2=9, col3=4+1+3=8
    rsum = [7.0, 6.0, 9.0, 8.0]
    k = 3
    n = 4
    expected_q = 12.0 / (k * n * (n + 1.0)) * sum(v * v for v in rsum) - 3.0 * k * (n + 1.0)
    # No ties, so tie correction is a no-op and statistic == q_raw
    assert abs(result["statistic"] - expected_q) < 1e-12
    assert abs(result["q_raw"] - expected_q) < 1e-12
    assert result["k"] == k
    assert result["n"] == n
    assert result["df"] == n - 1


def test_gb1221_edge():
    """Test edge case: minimum valid size (k=2, n=2)."""
    data = [
        [1.0, 2.0],
        [2.0, 1.0],
    ]
    result = gibbons_friedman(data)
    assert isinstance(result, dict)
    assert "statistic" in result
    assert "p_value" in result
    assert result["k"] == 2
    assert result["n"] == 2
    assert result["df"] == 1
