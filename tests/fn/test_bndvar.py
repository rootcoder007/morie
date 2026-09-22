"""Tests for bndvar.bound_variance_term."""

from morie.fn import _array_core as np

from morie.fn.bndvar import bound_variance_term


def test_bndvar_basic():
    """Test basic functionality: a moderate-width interval where c lies
    strictly between the one-sided and two-sided normal quantiles."""
    rng = np.random.default_rng(42)
    lower_hat = 0.0
    upper_hat = 1.0
    n = 100
    # se's are on the sqrt(n) scale per the docstring
    se_lower = float(rng.uniform(0.5, 1.5))
    se_upper = float(rng.uniform(0.5, 1.5))

    result = bound_variance_term(lower_hat, upper_hat, se_lower, se_upper, n)

    assert isinstance(result, dict)
    assert "ci" in result
    assert "c" in result
    assert "z_one_sided" in result
    assert "z_two_sided" in result
    assert "delta" in result
    assert "covers" in result
    assert "stoye_caveat" in result
    assert "n" in result
    assert "method" in result

    # delta matches the documented hat_delta = upper_hat - lower_hat
    assert result["delta"] == upper_hat - lower_hat
    assert result["n"] == n

    # c must lie in [z_one_sided, z_two_sided] for a non-degenerate set
    z1 = result["z_one_sided"]
    z2 = result["z_two_sided"]
    assert z1 <= result["c"] <= z2

    # CI endpoints follow the formula in the docstring
    ci_low, ci_high = result["ci"]
    assert ci_low == lower_hat - result["c"] * se_lower / np.sqrt(n)
    assert ci_high == upper_hat + result["c"] * se_upper / np.sqrt(n)

    # The CI must be wider than the identified set it is inverting
    assert ci_low < lower_hat
    assert ci_high > upper_hat


def test_bndvar_wide_set_equals_one_sided_z():
    """When delta is large relative to sqrt(n)*se, c collapses to z_{1-alpha}."""
    # Pick a very wide hat interval so shift is huge and gap(z1) >= 0
    lower_hat, upper_hat = 0.0, 100.0
    se_lower = se_upper = 1.0
    n = 10
    alpha = 0.05

    result = bound_variance_term(lower_hat, upper_hat, se_lower, se_upper,
                                 n, alpha=alpha)

    z1 = float(np.sqrt(2) * 0)  # placeholder to be replaced below
    # Recompute z_one_sided independently: z_{1-alpha} = sqrt(2)*erfinv(2*(1-alpha)-1)
    # Implemented here via a closed-form approximation using math only.
    import math
    z1_indep = math.sqrt(2) * math.erf(2 * (1 - alpha) - 1) if False else None

    # Use a plain arithmetic computation of Phi^{-1}(1 - alpha) via the
    # rational approximation baked into the docstring text.
    # We avoid importing scipy in the test; instead we rely on the known
    # identity and re-derive z1 from the result keys plus a numeric check
    # of the defining equation Phi(c + shift) - Phi(-c) = 1 - alpha.
    c = result["c"]
    # Independently compute shift using the documented formula
    shift = math.sqrt(n) * (upper_hat - lower_hat) / max(se_lower, se_upper)
    # Independently evaluate Phi via the standard normal CDF closed form
    # using math.erf to confirm the defining equation of c.
    def norm_cdf(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    lhs = norm_cdf(c + shift) - norm_cdf(-c)
    assert abs(lhs - (1 - alpha)) < 1e-8

    # For a wide set, c must equal z_{1-alpha} (the one-sided quantile)
    # Recompute z_{1-alpha} via the inverse-CDF bisection on math.erf.
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if norm_cdf(mid) < 1 - alpha:
            lo = mid
        else:
            hi = mid
    z_one_sided_indep = 0.5 * (lo + hi)

    assert abs(c - z_one_sided_indep) < 1e-6
    assert result["z_one_sided"] == z_one_sided_indep


def test_bndvar_edge():
    """Test edge case: a point-identified parameter collapses to the
    usual two-sided Wald interval, i.e. c = z_{1-alpha/2}."""
    lower_hat = upper_hat = 0.5
    se_lower = se_upper = 1.0
    n = 100
    alpha = 0.05

    result = bound_variance_term(lower_hat, upper_hat, se_lower, se_upper,
                                 n, alpha=alpha)

    assert isinstance(result, dict)
    assert result["delta"] == 0.0

    # When delta = 0, shift = 0 and the defining equation reduces to
    # Phi(c) - Phi(-c) = 1 - alpha  =>  2*Phi(c) - 1 = 1 - alpha
    # so Phi(c) = 1 - alpha/2, i.e. c = z_{1-alpha/2}.
    import math
    def norm_cdf(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    lo, hi = -10.0, 10.0
    target = 1.0 - alpha / 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if norm_cdf(mid) < target:
            lo = mid
        else:
            hi = mid
    z_two_sided_indep = 0.5 * (lo + hi)

    assert abs(result["c"] - z_two_sided_indep) < 1e-6
    assert result["z_two_sided"] == z_two_sided_indep
    # And c must be >= z_one_sided
    lo2, hi2 = -10.0, 10.0
    target1 = 1.0 - alpha
    for _ in range(200):
        mid = 0.5 * (lo2 + hi2)
        if norm_cdf(mid) < target1:
            lo2 = mid
        else:
            hi2 = mid
    z_one_sided_indep = 0.5 * (lo2 + hi2)
    assert result["c"] >= z_one_sided_indep - 1e-9
