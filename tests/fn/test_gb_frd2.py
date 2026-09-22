"""Tests for gb_frd2.gibbons_friedman_ties."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.gb_frd2 import gibbons_friedman_ties


def test_gb_frd2_basic():
    """Test basic functionality against the documented formula.

    Formula (Gibbons & Chakraborti 2011, eq. 12.2.12, p. 445):

        Q_raw  = 12 * S / (k * n * (n + 1))
        Q_corr = 12 * (n - 1) * S / (k * n * (n^2 - 1) - tie_sum)

    where S = sum_j (R_j - k*(n+1)/2)^2 over the n treatment rank sums,
    and the double sum tie_sum = sum_t t*(t^2 - 1) over every tied
    set of size t in every block.
    """
    # Two blocks (k=2), three treatments (n=3); the second block has
    # a tie so the correction factor is non-trivial but < 1.
    data = [
        [1.0, 2.0, 3.0],
        [3.0, 1.0, 1.0],
    ]

    # Hand-rank each block (1-based midranks for ties) and sum ranks.
    # Block 1: ranks [1, 2, 3] -> rank sums contribution [1, 2, 3]
    # Block 2: sorted values 1,1,3 -> midranks (1+2)/2=1.5, 1.5, 3
    #          positions: index 1 and 2 both get 1.5, index 0 gets 3
    #          -> contribution [3, 1.5, 1.5]
    rsum = [1.0 + 3.0, 2.0 + 1.5, 3.0 + 1.5]  # [4.0, 3.5, 4.5]
    k = 2
    n = 3
    S = sum((r - k * (n + 1) / 2.0) ** 2 for r in rsum)
    tiesum = 2.0 * (2.0 * 2.0 - 1.0)  # one tie of size 2 in block 2
    q_raw_expected = 12.0 * S / (k * n * (n + 1.0))
    den_expected = k * n * (float(n) ** 2 - 1.0) - tiesum
    q_corr_expected = 12.0 * (n - 1.0) * S / den_expected
    factor_expected = q_corr_expected / q_raw_expected

    result = gibbons_friedman_ties(data)

    # The RichResult exposes the documented keys (no p_value/estimate).
    assert "statistic" in result
    assert "q_raw" in result
    assert "tiesum" in result
    assert "factor" in result
    assert "s" in result
    assert "rank_sums" in result
    assert "k" in result
    assert "n" in result
    assert "method" in result

    assert result["k"] == k
    assert result["n"] == n
    assert result["tiesum"] == tiesum
    assert result["s"] == S
    assert list(result["rank_sums"]) == rsum
    assert result["q_raw"] == q_raw_expected
    assert result["statistic"] == q_corr_expected
    assert result["factor"] == factor_expected


def test_gb_frd2_edge():
    """Test edge cases: no ties -> factor is exactly 1."""
    # 3 blocks, 4 treatments, no ties anywhere -> tiesum = 0 and the
    # corrected Q equals the uncorrected Q, so factor == 1.
    data = [
        [4.0, 2.0, 1.0, 3.0],
        [1.0, 3.0, 4.0, 2.0],
        [2.0, 4.0, 3.0, 1.0],
    ]

    # No ties -> tiesum = 0, so the denominators satisfy
    #   12*(n-1)*S / (k*n*(n^2-1)) == 12*S / (k*n*(n+1)),
    # i.e. factor == 1 exactly.
    result = gibbons_friedman_ties(data)

    assert result["tiesum"] == 0.0
    assert result["k"] == 3
    assert result["n"] == 4
    # Independently check factor == corrected / raw == 1.
    assert result["factor"] == result["statistic"] / result["q_raw"]
    assert result["factor"] == 1.0
