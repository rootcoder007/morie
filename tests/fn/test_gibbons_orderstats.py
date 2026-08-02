"""Gibbons cluster C: order statistics, EDF, coverages (Ch 2).

Truth oracles: Monte Carlo with measured rates, scipy beta/binomial
identities, and the PDF-verified Theorem 2.11.1."""

from morie.fn import _array_core as np
import pytest
from morie.fn import _stats_core as stats

from morie.fn.gb221 import gibbons_quantile_deriv
from morie.fn.gb2111 import gibbons_tolerance_beta
from morie.fn.gb2111c import gibbons_elementary_coverage_beta
from morie.fn.gb2112 import gibbons_block_freq_dist
from morie.fn.gb2311 import gibbons_edf_mean_var
from morie.fn.gb2313 import gibbons_edf_joint_moment
from morie.fn.gb232 import gibbons_glivenko_cantelli
from morie.fn.gb233 import gibbons_edf_asymp_normal
from morie.fn.gb2431 import gibbons_binomial_beta_link
from morie.fn.gb251 import gibbons_pit
from morie.fn.gb_eqf import gibbons_emp_quantile
from morie.fn.gb_lsm import gibbons_large_sample_moments
from morie.fn.gb_med import gibbons_median_dist
from morie.fn.gb_pit2 import gibbons_pit_rng
from morie.fn.gb_rng import gibbons_range_dist
from morie.fn.gb_rnk import gibbons_rank_def


def test_tolerance_gamma_matches_monte_carlo_coverage():
    n, p = 50, 0.8
    out = gibbons_tolerance_beta(n=n, r=1, s=n, p=p)
    # Monte Carlo: fraction of uniform samples whose (U(1), U(n))
    # interval covers at least p
    rng = np.random.default_rng(0)
    hits = 0
    B = 4000
    for _ in range(B):
        u = np.sort(rng.random(n))
        hits += (u[-1] - u[0]) >= p
    mc = hits / B
    assert out["gamma"] == pytest.approx(mc, abs=0.02)  # measured 0.9987 vs MC
    assert out["coverage_dist"] == (n - 1, 2)
    # inverse: required n for gamma = 0.95, p = 0.9 -- then check it
    req = gibbons_tolerance_beta(gamma=0.95, p=0.9)["n_required"]
    assert gibbons_tolerance_beta(n=req, r=1, s=req, p=0.9)["gamma"] >= 0.95
    assert gibbons_tolerance_beta(n=req - 1, r=1, s=req - 1, p=0.9)["gamma"] < 0.95
    with pytest.raises(ValueError):
        gibbons_tolerance_beta(n=10, r=5, s=3, p=0.9)


def test_elementary_coverages_and_block_frequencies():
    n = 30
    out = gibbons_elementary_coverage_beta(n)
    assert out["mean"] == pytest.approx(1.0 / (n + 1))
    assert out["var"] == pytest.approx(stats.beta.var(1, n))
    # Monte Carlo: the FIRST and a MIDDLE gap have the same mean
    rng = np.random.default_rng(1)
    gaps = np.diff(np.sort(rng.random((3000, n)), axis=1), axis=1)
    assert gaps[:, 0].mean() == pytest.approx(out["mean"], abs=0.003)
    assert gaps[:, n // 2].mean() == pytest.approx(out["mean"], abs=0.003)
    # block frequencies: total compositions and uniformity
    bf = gibbons_block_freq_dist(4, 3)
    from math import comb

    assert bf["n_compositions"] == comb(7, 3)
    assert bf["pmf"] == pytest.approx(1 / comb(7, 3))
    assert gibbons_block_freq_dist(4, 3, [1, 1, 1, 0, 0])["valid_composition"] is True
    assert gibbons_block_freq_dist(4, 3, [1, 1, 1, 1, 0])["valid_composition"] is False


def test_quantile_derivative_identity_and_lsm():
    # normal: Q'(p) = 1/phi(z_p), exact
    out = gibbons_quantile_deriv(0.8, stats.norm())
    z = stats.norm.ppf(0.8)
    assert out["Q_prime"] == pytest.approx(1.0 / stats.norm.pdf(z), rel=1e-8)
    # Q'' = -f'/f^3 with f'(z) = -z phi(z)
    assert out["Q_double_prime"] == pytest.approx(
        z * stats.norm.pdf(z) / stats.norm.pdf(z) ** 3, rel=1e-4
    )
    # large-sample moments vs Monte Carlo for the normal median-ish stat
    r, n = 15, 29
    approx = gibbons_large_sample_moments(r, n)
    rng = np.random.default_rng(2)
    sims = np.sort(rng.standard_normal((6000, n)), axis=1)[:, r - 1]
    assert approx["mean"] == pytest.approx(sims.mean(), abs=0.02)
    assert approx["var"] == pytest.approx(sims.var(), rel=0.15)
    with pytest.raises(ValueError):
        gibbons_large_sample_moments(0, 10)


def test_edf_moments_match_the_binomial():
    F, n = 0.3, 40
    out = gibbons_edf_mean_var(F, n)
    assert out["mean"] == pytest.approx(F)
    assert out["var"] == pytest.approx(stats.binom.var(n, F) / n**2)
    jm = gibbons_edf_joint_moment(0.3, 0.7, n)
    assert jm["cov_edf"] == pytest.approx(0.3 * (1 - 0.7) / n)
    # Monte Carlo covariance check
    rng = np.random.default_rng(3)
    u = rng.random((8000, n))
    Sx = (u <= 0.3).mean(axis=1)
    Sy = (u <= 0.7).mean(axis=1)
    assert np.cov(Sx, Sy)[0, 1] == pytest.approx(jm["cov_edf"], rel=0.15)
    with pytest.raises(ValueError):
        gibbons_edf_joint_moment(0.7, 0.3, n)


def test_gc_witness_and_pointwise_clt():
    rng = np.random.default_rng(4)
    d_small = gibbons_glivenko_cantelli(rng.standard_normal(50))["sup_distance"]
    d_large = gibbons_glivenko_cantelli(rng.standard_normal(5000))["sup_distance"]
    assert d_large < d_small  # convergence visible
    out = gibbons_glivenko_cantelli(rng.standard_normal(100))
    assert 0 < out["dkw_bound_at_observed"] <= 1
    z = gibbons_edf_asymp_normal(0.35, 0.3, 100)
    assert z["z"] == pytest.approx(0.05 / np.sqrt(0.3 * 0.7 / 100))
    with pytest.raises(ValueError):
        gibbons_edf_asymp_normal(0.5, 1.0, 100)


def test_binomial_beta_identity_holds_everywhere():
    for t in (0.0, 0.2, 0.5, 0.9, 1.0):
        for r, n in ((1, 5), (3, 8), (8, 8)):
            out = gibbons_binomial_beta_link(t, r, n)
            assert out["agree"] is True
            assert out["binomial_tail"] == pytest.approx(out["incomplete_beta"], abs=1e-12)
    with pytest.raises(ValueError):
        gibbons_binomial_beta_link(0.5, 0, 5)


def test_pit_and_inverse_pit_roundtrip():
    rng = np.random.default_rng(5)
    x = rng.standard_normal(400)
    out = gibbons_pit(x)  # true generator: should NOT reject
    assert out["ks_p"] > 0.01
    # wrong CDF: should reject decisively
    bad = gibbons_pit(x, F=lambda v: stats.norm.cdf(v, loc=2.0))
    assert bad["ks_p"] < 1e-6
    # inverse transform reproduces the target law
    u = rng.random(2000)
    draws = gibbons_pit_rng(u, stats.expon.ppf)["X"]
    assert stats.kstest(draws, "expon").pvalue > 0.01
    with pytest.raises(ValueError):
        gibbons_pit_rng([1.5], stats.expon.ppf)


def test_empirical_quantile_is_a_step_function_of_order_stats():
    data = [3.0, 1.0, 4.0, 1.5, 5.0]
    srt = sorted(data)
    # u in ((i-1)/n, i/n] -> X_(i)
    assert gibbons_emp_quantile(0.2, data)["quantile"] == srt[0]
    assert gibbons_emp_quantile(0.2000001, data)["quantile"] == srt[1]
    assert gibbons_emp_quantile(1.0, data)["quantile"] == srt[4]
    out = gibbons_emp_quantile([0.1, 0.5, 0.9], data)
    assert list(out["quantile"]) == [srt[0], srt[2], srt[4]]
    with pytest.raises(ValueError):
        gibbons_emp_quantile(0.0, data)


def test_rank_identity_and_median_range_distributions():
    x = [3.0, 1.0, 4.0, 1.5, 5.0]
    out = gibbons_rank_def(x)
    assert list(out["ranks"]) == [3, 1, 4, 2, 5]
    assert out["edf_values"] == pytest.approx(np.array([3, 1, 4, 2, 5]) / 5)
    with pytest.raises(ValueError):
        gibbons_rank_def([1.0, 1.0, 2.0])
    # median CDF at the true median of a symmetric law is exactly 1/2
    med = gibbons_median_dist(0.0, 11)
    assert med["cdf"] == pytest.approx(0.5)
    assert med["beta_params"] == (6, 6)
    # and matches Monte Carlo elsewhere
    rng = np.random.default_rng(6)
    sims = np.median(rng.standard_normal((6000, 11)), axis=1)
    assert gibbons_median_dist(0.3, 11)["cdf"] == pytest.approx(
        (sims <= 0.3).mean(), abs=0.02
    )
    with pytest.raises(ValueError):
        gibbons_median_dist(0.0, 10)  # even n refused
    # range CDF vs Monte Carlo
    sims_w = np.ptp(rng.standard_normal((5000, 8)), axis=1)
    assert gibbons_range_dist(3.0, 8)["cdf"] == pytest.approx(
        (sims_w <= 3.0).mean(), abs=0.02
    )
    with pytest.raises(ValueError):
        gibbons_range_dist(-1.0, 8)
