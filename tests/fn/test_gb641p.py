"""Tests for gb641p.gibbons_median_test_power."""

from morie.fn import _array_core as np

from morie.fn.gb641p import gibbons_median_test_power


def _h0(u):
    """Identity link: F_Y(F_X^{-1}(u)) = u under H0."""
    return u


def test_gb641p_basic():
    """Under H0, power equals the significance level (Type-I error rate).

    For gibbons_median_test_power with ``g(u) = u``, the r-th X order
    statistic lies at the r/(m+1) quantile of the Beta(m+1-2r, ...)-
    shaped latent variable, and P(W_r < wcrit) is computable in closed
    form as the regularized incomplete Beta function evaluated at
    wcrit with parameters (r, m-r+1).
    """
    m = 10
    n = 100
    r = 3
    wcrit = 5
    result = gibbons_median_test_power(m, n, r, wcrit, _h0)

    # Result exposes the documented keys (RichResult acts like a dict).
    assert isinstance(result, dict)
    for key in ("power", "pmf", "m", "n", "r", "wcrit", "method"):
        assert key in result, f"missing documented key {key!r}"

    # Stored metadata echoes the inputs.
    assert result["m"] == m
    assert result["n"] == n
    assert result["r"] == r
    assert result["wcrit"] == wcrit

    # pmf has length n+1 (one probability per i = 0..n).
    assert len(result["pmf"]) == n + 1
    # pmf entries are non-negative and sum to 1.
    assert all(p >= 0.0 for p in result["pmf"])
    assert abs(sum(result["pmf"]) - 1.0) < 1e-8

    # power = sum of pmf[i] for i < wcrit.
    expected_power = sum(result["pmf"][i] for i in range(min(wcrit, n + 1)))
    assert abs(result["power"] - expected_power) < 1e-12

    # Independent closed-form for H0: U = X_(r) ~ Beta(r, m-r+1) on
    # [0,1], and W_r | U=u ~ Binomial(n, u), so P(W_r < wcrit) =
    # E[I(U)(wcrit, n+1-wcrit)] / B(r, m-r+1).  The Beta CDF is
    # available in the function's array shim's special module.
    from math import lgamma, exp

    def log_beta(a, b):
        return lgamma(a) + lgamma(b) - lgamma(a + b)

    def log_binom(n_, k):
        if k < 0 or k > n_:
            return float("-inf")
        return lgamma(n_ + 1) - lgamma(k + 1) - lgamma(n_ - k + 1)

    # Compute E[ I(wcrit; n+1-wcrit) ] where I is the regularized
    # incomplete beta function, by direct summation of the equivalent
    # series expansion I_x(a, b) = sum_{j=a..a+b-1} C(a+b-1, j) x^j (1-x)^(a+b-1-j)
    # at x = U for U ~ Beta(r, m-r+1).  We instead compute the CDF of
    # the Binomial(n, U) mixture analytically using the identity
    # E[Beta(r, m-r+1)-weighted F_Beta-incomplete] which equals the
    # regularized incomplete beta at x = wcrit/n with parameters
    # (wcrit, n - wcrit + 1) averaged against Beta(r, m-r+1).  This is
    # exactly E[I_U(wcrit, n-wcrit+1)] where U ~ Beta(r, m-r+1), and
    # admits a closed form via Beta-Binomial identity:
    #   sum_{i=0}^{wcrit-1} C(n, i) B(i+r, n-i+m-r+1) / B(r, m-r+1)
    a, b = r, m - r + 1
    log_norm = log_beta(a, b)
    h0_power = 0.0
    for i in range(min(wcrit, n + 1)):
        log_pmf_i = (
            log_binom(n, i)
            + log_beta(i + a, n - i + b)
            - log_norm
        )
        h0_power += exp(log_pmf_i)

    assert abs(result["power"] - h0_power) < 1e-4


def test_gb641p_edge():
    """Edge cases: wcrit = 0 yields zero power; wcrit = n+1 yields power = 1.

    With wcrit <= 0, the rejection region W_r < wcrit is empty so the
    power must be exactly 0.  With wcrit > n, every i in 0..n satisfies
    i < wcrit, so the power equals the total pmf mass, which is 1.
    """
    m = 10
    n = 100
    r = 3

    # wcrit = 0: empty rejection region.
    result = gibbons_median_test_power(m, n, r, 0, _h0)
    assert isinstance(result, dict)
    assert result["power"] == 0.0
    # pmf is still a valid distribution.
    assert len(result["pmf"]) == n + 1
    assert abs(sum(result["pmf"]) - 1.0) < 1e-8

    # wcrit = n + 1: every possible i falls in the rejection region.
    result = gibbons_median_test_power(m, n, r, n + 1, _h0)
    assert isinstance(result, dict)
    assert abs(result["power"] - 1.0) < 1e-8

    # Smallest valid inputs: m = n = 1, r = 1.
    result = gibbons_median_test_power(1, 1, 1, 1, _h0)
    assert isinstance(result, dict)
    assert len(result["pmf"]) == 2  # i = 0, 1
    assert abs(sum(result["pmf"]) - 1.0) < 1e-8
    assert abs(result["power"] - result["pmf"][0]) < 1e-12
