"""mavi: variance inflation for correlated effects (Hedges, Tipton & Johnson 2010)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.mavi import ma_var_inflation_correlated as inflate


def test_mavi_equal_variances_give_the_closed_form_factor():
    """With k equal variances the factor is exactly 1 + (k-1)rho."""
    for k in (2, 3, 5, 10):
        for rho in (0.0, 0.3, 0.5, 1.0):
            r = inflate(np.full(k, 2.0), rho)
            assert r["inflation_factor"] == pytest.approx(1.0 + (k - 1) * rho)


def test_mavi_five_outcomes_at_rho_half_triple_the_variance():
    """The worked case from the docstring: 1 + 4*0.5 = 3."""
    r = inflate(np.full(5, 1.0), 0.5)
    assert r["V_naive"] == 5.0
    assert r["V_inflated"] == pytest.approx(15.0)
    assert r["inflation_factor"] == pytest.approx(3.0)


def test_mavi_rho_zero_is_the_naive_sum():
    v = np.array([1.0, 4.0, 9.0])
    r = inflate(v, 0.0)
    assert r["V_inflated"] == pytest.approx(r["V_naive"]) == pytest.approx(14.0)


def test_mavi_matches_an_explicit_covariance_matrix():
    """V* must equal 1' Sigma 1 with Sigma_ij = rho sqrt(Vi Vj), Sigma_ii = Vi."""
    v = np.array([1.0, 2.0, 5.0, 0.5])
    rho = 0.4
    sd = np.sqrt(v)
    Sigma = rho * np.outer(sd, sd)
    np.fill_diagonal(Sigma, v)
    assert inflate(v, rho)["V_inflated"] == pytest.approx(float(Sigma.sum()))


def test_mavi_variance_of_the_mean_divides_by_k_squared():
    """The usual slip is V*/k. The variance of a mean is V*/k^2."""
    v = np.full(4, 3.0)
    r = inflate(v, 0.25)
    assert r["V_mean_inflated"] == pytest.approx(r["V_inflated"] / 16.0)
    assert r["V_mean_inflated"] != pytest.approx(r["V_inflated"] / 4.0)


def test_mavi_perfect_correlation_matches_the_summed_standard_errors():
    """At rho = 1 the variables are collinear: V* = (sum of SEs)^2."""
    v = np.array([1.0, 4.0, 9.0])
    assert inflate(v, 1.0)["V_inflated"] == pytest.approx(float(np.sqrt(v).sum() ** 2))


def test_mavi_ignoring_dependence_understates_the_interval():
    """A naive CI at rho=0.5, k=5 is 1/sqrt(3) = 58% of the honest width."""
    r = inflate(np.full(5, 1.0), 0.5)
    naive_half_width = 1.96 * np.sqrt(r["V_naive"] / 25.0)
    honest_half_width = 1.96 * np.sqrt(r["V_mean_inflated"])
    assert naive_half_width / honest_half_width == pytest.approx(1 / np.sqrt(3.0))


def test_mavi_rejects_a_non_psd_correlation():
    """Compound symmetry needs rho >= -1/(k-1); below that no such variables exist."""
    with pytest.raises(ValueError, match="positive semi-definite"):
        inflate(np.ones(3), -0.9)
    # The boundary itself is admissible and gives V* = 0.
    assert inflate(np.ones(3), -0.5)["V_inflated"] == pytest.approx(0.0)


def test_mavi_rejects_negative_variance_and_out_of_range_rho():
    with pytest.raises(ValueError, match="non-negative"):
        inflate(np.array([1.0, -1.0]), 0.5)
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        inflate(np.ones(3), 1.5)
