"""Tests for gb1451.gibbons_mcnemar."""

from morie.fn import _array_core as np

from morie.fn.gb1451 import gibbons_mcnemar


def _expected(table):
    """Compute the documented Q statistic and p-values directly.

    Q = (X12 - X21)^2 / (X12 + X21)            (chi-square, 1 df)
    exact p  = min(1, 2 * sum_{i=0..k} C(n,i) * 0.5^n), k = min(X12, X21)
    """
    import math

    x12 = float(table[0][1])
    x21 = float(table[1][0])
    nd = x12 + x21
    q = (x12 - x21) ** 2 / nd
    k = int(round(min(x12, x21)))
    n = int(round(nd))
    pex = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) * 0.5 ** n)
    # chi-square survival function with 1 df: sf(x,1) = 1 - erf(sqrt(x/2))
    # but we already let the function own its own p_value; the test only
    # verifies the statistic and exact p.
    return q, x12, x21, nd, pex


def test_gb1451_basic():
    """Test basic functionality with a hand-crafted 2x2 table of discordant counts."""
    table = [[12.0, 7.0],
             [5.0,  3.0]]
    result = gibbons_mcnemar(table)

    assert isinstance(result, dict)
    # Documented return keys.
    for key in ("statistic", "df", "p_value", "p_exact",
                "x12", "x21", "ndisc", "method"):
        assert key in result, f"missing key {key!r} in result"

    q_exp, x12_exp, x21_exp, nd_exp, pex_exp = _expected(table)
    assert result["x12"] == x12_exp
    assert result["x21"] == x21_exp
    assert result["ndisc"] == nd_exp
    assert result["statistic"] == q_exp
    assert result["df"] == 1
    assert result["p_exact"] == pex_exp


def test_gb1451_correct():
    """Continuity correction subtracts 1 from |X12 - X21| before squaring."""
    table = [[12.0, 7.0],
             [5.0,  3.0]]
    result = gibbons_mcnemar(table, correct=True)

    x12, x21 = 7.0, 5.0
    d = max(0.0, abs(x12 - x21) - 1.0)
    q_exp = d * d / (x12 + x21)
    assert result["statistic"] == q_exp


def test_gb1451_edge():
    """Edge case: minimal discordant counts (X12 = X21 = 1)."""
    table = [[5.0, 1.0],
             [1.0, 4.0]]
    result = gibbons_mcnemar(table)

    assert isinstance(result, dict)
    # When X12 == X21 the statistic is 0 and the exact two-sided p = 1.
    assert result["statistic"] == 0.0
    assert result["p_exact"] == 1.0
    assert result["x12"] == 1.0
    assert result["x21"] == 1.0
    assert result["ndisc"] == 2.0
    assert result["df"] == 1
