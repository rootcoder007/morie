"""Tests for gb231.gibbons_edf_binomial."""

from morie.fn import _array_core as np

from morie.fn.gb231 import gibbons_edf_binomial


def test_gb231_basic():
    """Test basic functionality."""
    n = 100
    fx = 0.3
    result = gibbons_edf_binomial(n, fx)
    assert isinstance(result, dict)
    # Edfbinom always returns 'mean', 'var', 'pmf', 'cdf', 'n', 'fx', 'method'.
    assert set(result.keys()) >= {"mean", "var", "pmf", "cdf", "n", "fx", "method"}
    # With i=None, pmf and cdf are NaN.
    assert result["pmf"] != result["pmf"]  # NaN != NaN
    assert result["cdf"] != result["cdf"]  # NaN != NaN
    # Mean and variance from the Binomial(n, fx) parameters n and fx.
    assert result["mean"] == n * fx
    assert result["var"] == n * fx * (1.0 - fx)
    assert result["n"] == n
    assert result["fx"] == fx
    assert "Binomial" in result["method"]


def test_gb231_edge():
    """Test edge cases: evaluation at i, and boundary values of fx."""
    n = 10
    fx = 0.0
    # fx = 0: with i = 0, pmf and cdf must be 1; with i > 0, pmf = 0.
    r0 = gibbons_edf_binomial(n, fx, i=0)
    assert r0["pmf"] == 1.0
    assert r0["cdf"] == 1.0
    r1 = gibbons_edf_binomial(n, fx, i=1)
    assert r1["pmf"] == 0.0
    assert r1["cdf"] == 1.0  # P(T <= 1) = P(T=0)+P(T=1) = 1+0 = 1

    # fx = 1: with i = n, pmf and cdf must be 1; with i = n-1, pmf = 0.
    fx1 = 1.0
    rn = gibbons_edf_binomial(n, fx1, i=n)
    assert rn["pmf"] == 1.0
    assert rn["cdf"] == 1.0
    rnm1 = gibbons_edf_binomial(n, fx1, i=n - 1)
    assert rnm1["pmf"] == 0.0
    assert rnm1["cdf"] == 0.0  # P(T <= n-1) = P(T=n-1) = 0

    # A non-trivial interior check written out from the documented formula:
    # P[n S_n(x) = i] = C(n,i) * fx**i * (1-fx)**(n-i)
    # and CDF = sum_{k<=i} P[n S_n(x) = k].
    fx2 = 0.4
    i = 4
    r = gibbons_edf_binomial(n, fx2, i=i)
    expected_pmf = 1.0
    for k in range(i + 1):
        term = 1.0
        for _ in range(k):
            term *= (n - _) / (_ + 1)  # C(n,k) built multiplicatively
        term *= fx2**k * (1.0 - fx2) ** (n - k)
        # accumulate the pmf term for the targeted k
        if k == i:
            expected_pmf = term
    # Recompute expected pmf cleanly for the targeted k only.
    comb_ni = 1.0
    for _ in range(i):
        comb_ni *= (n - _) / (_ + 1)
    expected_pmf = comb_ni * fx2**i * (1.0 - fx2) ** (n - i)
    # CDF: sum of C(n,k)*fx^k*(1-fx)^(n-k) for k = 0..i
    expected_cdf = 0.0
    for k in range(i + 1):
        c = 1.0
        for _ in range(k):
            c *= (n - _) / (_ + 1)
        expected_cdf += c * fx2**k * (1.0 - fx2) ** (n - k)
    assert r["pmf"] == expected_pmf
    assert r["cdf"] == expected_cdf
    assert r["mean"] == n * fx2
    assert r["var"] == n * fx2 * (1.0 - fx2)
