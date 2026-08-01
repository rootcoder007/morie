"""Shared backend for the Brus Spatial Sampling shelf.

Every function implements a numbered equation from Brus, D. J. (2022),
Spatial Sampling with R, The R Series, CRC Press (open-access bookdown
edition at dickbrus.github.io/SpatialSamplingwithR, whose LaTeX source is
the spec).  Print equation numbers cited per function; the two known
print/web numbering drifts in chapter 13 are documented at the functions.
"""

from __future__ import annotations

import math

import numpy as np

__all__: list = []


def _v(x):
    return np.atleast_1d(np.asarray(x, dtype=float))


# ------------------------------------------------- ch 2: Horvitz-Thompson


def ht_total(z, pi):
    """HT total t_hat = sum z_k / pi_k with weights w_k = 1/pi_k, eqs (2.2)-(2.3)."""
    z, pi = _v(z), _v(pi)
    if z.shape != pi.shape or (pi <= 0).any() or (pi > 1).any():
        raise ValueError("need matching shapes and 0 < pi <= 1")
    return float(np.sum(z / pi))


def ht_mean(z, pi, n_population):
    """HT mean zbar_hat = (1/N) sum z_k / pi_k, eq (2.4)."""
    if n_population <= 0:
        raise ValueError("N must be positive")
    return ht_total(z, pi) / n_population


# ------------------------------------------------- ch 3: simple random sampling


def si_proportion(y):
    """Sample proportion p_hat = (1/n) sum y_k of 0/1 indicators, eq (3.6)."""
    y = _v(y)
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("y must be 0/1 indicators")
    return float(y.mean())


def si_proportion_variance(p_hat, n, n_population):
    """V_hat(p_hat) = (1 - n/N) p(1-p)/(n-1), eq (3.14)."""
    if not 0 <= p_hat <= 1 or n < 2 or n > n_population:
        raise ValueError("need 0<=p<=1 and 2 <= n <= N")
    return float((1.0 - n / n_population) * p_hat * (1.0 - p_hat) / (n - 1))


def confidence_interval(estimate, variance, u_crit):
    """Interval estimate -/+ u * sqrt(V), eq (3.15)."""
    if variance < 0:
        raise ValueError("variance must be nonnegative")
    half = u_crit * math.sqrt(variance)
    return {"lower": float(estimate - half), "upper": float(estimate + half)}


def infinite_total(zbar_hat, area, sample_area):
    """t_hat(z) = (A/a) zbar_hat for infinite populations, eq (3.18)."""
    if area <= 0 or sample_area <= 0:
        raise ValueError("areas must be positive")
    return float(area / sample_area * zbar_hat)


def infinite_total_variance(s2_hat, n, area, sample_area):
    """V_hat(t_hat) = (A/a)^2 S2_hat/n, eq (3.21)."""
    if s2_hat < 0 or n <= 0 or area <= 0 or sample_area <= 0:
        raise ValueError("invalid inputs")
    return float((area / sample_area) ** 2 * s2_hat / n)


# ------------------------------------------------- ch 4: stratified SI


def stratified_mean(stratum_means, stratum_weights):
    """zbar_hat = sum w_h zbar_hat_h, eqs (4.1)-(4.2)."""
    m, w = _v(stratum_means), _v(stratum_weights)
    if m.shape != w.shape or (w < 0).any() or abs(w.sum() - 1.0) > 1e-8:
        raise ValueError("weights must be nonnegative and sum to 1")
    return float(np.dot(w, m))


def stratified_variance(stratum_variances, stratum_weights):
    """V_hat = sum w_h^2 V_hat_h, eq (4.4)."""
    v, w = _v(stratum_variances), _v(stratum_weights)
    if v.shape != w.shape or (v < 0).any():
        raise ValueError("invalid inputs")
    return float(np.dot(w ** 2, v))


def stratified_cost(c0, stratum_costs, stratum_sizes):
    """Linear cost C = c0 + sum n_h c_h, eq (4.18)."""
    c, n = _v(stratum_costs), _v(stratum_sizes)
    if c.shape != n.shape or (c < 0).any() or (n < 0).any() or c0 < 0:
        raise ValueError("invalid inputs")
    return float(c0 + np.dot(n, c))


# ------------------------------------------------- ch 6: cluster sampling


def cluster_total_pps(cluster_totals, cluster_sizes, m_population, n):
    """t_hat = (M/n) sum t_j/M_j = (M/n) sum zbar_j (pps clusters), eq (6.4)."""
    t, m = _v(cluster_totals), _v(cluster_sizes)
    if t.shape != m.shape or (m <= 0).any() or n != t.size or m_population <= 0:
        raise ValueError("invalid inputs")
    return float(m_population / n * np.sum(t / m))


def cluster_total_si(cluster_totals, n_clusters_population, n):
    """t_hat = (N/n) sum t_j (SI of clusters), eq (6.9)."""
    t = _v(cluster_totals)
    if n != t.size or n_clusters_population <= 0:
        raise ValueError("invalid inputs")
    return float(n_clusters_population / n * t.sum())


def cluster_mean_from_total(t_hat, m_population):
    """zbarbar_hat = t_hat / M, eq (6.10)."""
    if m_population <= 0:
        raise ValueError("M must be positive")
    return float(t_hat / m_population)


# ------------------------------------------------- ch 7: two-stage sampling


def twostage_mean(primary_unit_means):
    """zbarbar_hat = (1/n) sum zbar_hat_j, eq (7.2)."""
    m = _v(primary_unit_means)
    return float(m.mean())


def twostage_variance_components(s2_between, s2_within, n, m):
    """V = S2_b/n + S2_w/(n m), eq (7.3)."""
    if s2_between < 0 or s2_within < 0 or n <= 0 or m <= 0:
        raise ValueError("invalid inputs")
    return float(s2_between / n + s2_within / (n * m))


def twostage_variance_estimator(primary_unit_means):
    """V_hat = S2_hat(zbar)/n with S2_hat = var of PSU means, eqs (7.7)-(7.8)."""
    m = _v(primary_unit_means)
    n = m.size
    if n < 2:
        raise ValueError("need >= 2 primary units")
    s2 = float(np.sum((m - m.mean()) ** 2) / (n - 1))
    return {"s2_psu": s2, "variance": s2 / n, "n": n}


def twostage_optimal_n_variance(s_w, s_b, c1, c2, v_max):
    """n = (S_w S_b sqrt(c2/c1) + S_b^2)/V_max, eq (7.9)."""
    if min(s_w, s_b, c1, c2, v_max) <= 0:
        raise ValueError("all inputs must be positive")
    return float((s_w * s_b * math.sqrt(c2 / c1) + s_b ** 2) / v_max)


def twostage_optimal_m(s_w, s_b, c1, c2):
    """m = (S_w/S_b) sqrt(c1/c2), eq (7.10)."""
    if min(s_w, s_b, c1, c2) <= 0:
        raise ValueError("all inputs must be positive")
    return float(s_w / s_b * math.sqrt(c1 / c2))


def twostage_optimal_n_budget(s_w, s_b, c1, c2, c_max):
    """n = C_max S_b / (S_w sqrt(c1 c2) + S_b c1), eq (7.11)."""
    if min(s_w, s_b, c1, c2, c_max) <= 0:
        raise ValueError("all inputs must be positive")
    return float(c_max * s_b / (s_w * math.sqrt(c1 * c2) + s_b * c1))


def twostage_total_variance_pps(p, t_j, t_total, m_j, f2_j, s2_j, m_j_sampled, n):
    """True two-stage pps variance, eq (7.12)."""
    p, t_j, m_j = _v(p), _v(t_j), _v(m_j)
    f2_j, s2_j, m_s = _v(f2_j), _v(s2_j), _v(m_j_sampled)
    if not (p.shape == t_j.shape == m_j.shape == f2_j.shape == s2_j.shape
            == m_s.shape) or (p <= 0).any() or n <= 0:
        raise ValueError("invalid inputs")
    first = np.sum(p * (t_j / p - t_total) ** 2) / n
    second = np.sum(m_j ** 2 * (1.0 - f2_j) * s2_j / (m_s * p)) / n
    return float(first + second)


def twostage_total_si(psu_total_estimates, n_psu_population):
    """t_hat = (N/n) sum t_hat_j (SI of PSUs), eq (7.13)."""
    t = _v(psu_total_estimates)
    if n_psu_population <= 0 or t.size == 0:
        raise ValueError("invalid inputs")
    return float(n_psu_population / t.size * t.sum())


# ------------------------------------------------- ch 8: pps sampling


def pps_total_variance(z, p, t_hat):
    """V_hat(t_hat) = (1/(n(n-1))) sum (z_k/p_k - t_hat)^2, eq (8.2)."""
    z, p = _v(z), _v(p)
    n = z.size
    if z.shape != p.shape or (p <= 0).any() or n < 2:
        raise ValueError("invalid inputs")
    return float(np.sum((z / p - t_hat) ** 2) / (n * (n - 1)))


# ------------------------------------------------- ch 9: balanced/spread


def regression_total(t_pi_z, t_x_true, t_pi_x, b_hat):
    """Regression estimator of the total, eq (9.2)."""
    return float(t_pi_z + b_hat * (t_x_true - t_pi_x))


def balanced_variance(e, pi, c, n_population, p):
    """V_hat = (1/N^2)(n/(n-p)) sum c_k (e_k/pi_k)^2, eq (9.3)."""
    e, pi, c = _v(e), _v(pi), _v(c)
    n = e.size
    if not (e.shape == pi.shape == c.shape) or (pi <= 0).any() or n <= p:
        raise ValueError("invalid inputs")
    return float(np.sum(c * (e / pi) ** 2) * n / (n - p) / n_population ** 2)


def local_mean_variance(e, pi, e_local_mean, n, p):
    """V_hat = (n/(n-p))(p/(p+1)) sum (1-pi)(e/pi - ebar)^2, eq (9.10)."""
    e, pi, eb = _v(e), _v(pi), _v(e_local_mean)
    if not (e.shape == pi.shape == eb.shape) or (pi <= 0).any() or n <= p:
        raise ValueError("invalid inputs")
    return float(n / (n - p) * p / (p + 1)
                 * np.sum((1.0 - pi) * (e / pi - eb) ** 2))


# ------------------------------------------------- ch 10: model-assisted


def difference_estimator(m_all, z_sample, m_sample, pi_sample, n_population):
    """zbar_dif = mean of model predictions + HT mean of residuals, eq (10.2).

    Serves the working model of eq (10.1): z = m(x) + eps.
    """
    m_all = _v(m_all)
    z, m_s, pi = _v(z_sample), _v(m_sample), _v(pi_sample)
    if not (z.shape == m_s.shape == pi.shape) or (pi <= 0).any():
        raise ValueError("invalid inputs")
    if m_all.size != n_population:
        raise ValueError("m_all must cover the population")
    return float(m_all.mean() + np.sum((z - m_s) / pi) / n_population)


def gls_population_slope(x, z, sigma2):
    """Population GLS b = (sum x x^T/sig2)^-1 sum x z/sig2, eq (10.4)."""
    x, z, s2 = np.asarray(x, float), _v(z), _v(sigma2)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[0] != z.size or s2.size != z.size or (s2 <= 0).any():
        raise ValueError("invalid inputs")
    xtx = (x / s2[:, None]).T @ x
    xtz = (x / s2[:, None]).T @ z
    return np.linalg.solve(xtx, xtz)


def gls_sample_slope(x, z, sigma2, pi):
    """Sample-weighted GLS b_hat with 1/(sig2 pi) weights, eqs (10.6), (10.15).

    Passing sigma2 = 1 and pi = 1 vectors gives the unweighted OLS form of
    eq (10.15).
    """
    pi = _v(pi)
    if (pi <= 0).any():
        raise ValueError("pi must be positive")
    return gls_population_slope(x, z, _v(sigma2) * pi)


def regression_estimator_general(x_all, b_hat, z_sample, x_sample, pi_sample,
                                 n_population):
    """zbar_regr = mean(x^T b) + HT mean of residuals, eq (10.8)."""
    x_all = np.asarray(x_all, float)
    x_s = np.asarray(x_sample, float)
    if x_all.ndim == 1:
        x_all = x_all[:, None]
    if x_s.ndim == 1:
        x_s = x_s[:, None]
    b = _v(b_hat)
    z, pi = _v(z_sample), _v(pi_sample)
    if x_all.shape[0] != n_population or (pi <= 0).any():
        raise ValueError("invalid inputs")
    resid = z - x_s @ b
    return float((x_all @ b).mean() + np.sum(resid / pi) / n_population)


def regression_estimator_slopes(zbar_pi, b_hats, xbar_true, xbar_pi):
    """zbar_regr = zbar_pi + sum b_j (xbar_j - xbar_hat_j), eqs (10.9), (10.10), (10.21)."""
    b, xt, xp = _v(b_hats), _v(xbar_true), _v(xbar_pi)
    if not (b.shape == xt.shape == xp.shape):
        raise ValueError("shape mismatch")
    return float(zbar_pi + np.dot(b, xt - xp))


def si_regression_variance(e, n, n_population):
    """V_hat = (1 - n/N) S2_hat(e)/n with S2_hat(e) = sum e^2/(n-1), eqs (10.13)-(10.14)."""
    e = _v(e)
    if e.size != n or n < 2 or n > n_population:
        raise ValueError("invalid inputs")
    s2_e = float(np.sum(e ** 2) / (n - 1))
    return {"s2_e": s2_e, "variance": (1.0 - n / n_population) * s2_e / n}


def g_weight_simple(x_k, xbar_true, xbar_sample, s2_x):
    """g_k = 1 + (xbar - xbar_S)(x_k - xbar_S)/S2_hat(x), eq (10.17)."""
    if s2_x <= 0:
        raise ValueError("S2(x) must be positive")
    return float(1.0 + (xbar_true - xbar_sample) * (x_k - xbar_sample) / s2_x)


def g_weighted_variance(g, e, n, n_population):
    """V_hat = (1 - n/N) sum g^2 e^2 / (n(n-1)), eq (10.18)."""
    g, e = _v(g), _v(e)
    if g.shape != e.shape or n < 2:
        raise ValueError("invalid inputs")
    return float((1.0 - n / n_population) * np.sum(g ** 2 * e ** 2)
                 / (n * (n - 1)))


def ratio_total(t_pi_z, t_pi_x, t_x_true):
    """Ratio estimator t_ratio = (t_pi(z)/t_pi(x)) t(x), eq (10.23).

    The heteroscedastic through-the-origin working model of eq (10.24)
    (sigma2 proportional to x) motivates this estimator.
    """
    if t_pi_x == 0:
        raise ValueError("t_pi(x) must be nonzero")
    return float(t_pi_z / t_pi_x * t_x_true)


def ratio_total_variance(e, n, n_population):
    """V_hat(t_ratio) = N^2 S2_hat(e)/n, eq (10.25)."""
    e = _v(e)
    if e.size != n or n < 2:
        raise ValueError("invalid inputs")
    return float(n_population ** 2 * np.sum(e ** 2) / (n - 1) / n)


def ratio_g_weight(t_x_true, t_pi_x):
    """Constant g-weight g = t(x)/t_pi(x), eq (10.27)."""
    if t_pi_x == 0:
        raise ValueError("t_pi(x) must be nonzero")
    return float(t_x_true / t_pi_x)


def poststratified_mean(group_means_sample, group_weights):
    """Poststratified estimator under the ANOVA model of eq (10.32):
    zbar_pst = sum w_g zbar_S,g (group means weighted by population shares)."""
    return stratified_mean(group_means_sample, group_weights)


def mixed_calibration_mean(zbar_pi, a_hat, pi_sample, m_all_mean, m_ht_mean,
                           b_hat, n_population):
    """Mixed-model calibration estimator, eq (10.36)."""
    pi = _v(pi_sample)
    if (pi <= 0).any():
        raise ValueError("pi must be positive")
    term_a = a_hat * (1.0 - np.sum(1.0 / pi) / n_population)
    term_b = b_hat * (m_all_mean - m_ht_mean)
    return float(zbar_pi + term_a + term_b)


def mixed_calibration_intercept(b_hat, z_sample, pi_sample, n_population):
    """a_hat = (1 - b_hat) (1/N) sum z_k/pi_k, eq (10.38)."""
    z, pi = _v(z_sample), _v(pi_sample)
    if z.shape != pi.shape or (pi <= 0).any():
        raise ValueError("invalid inputs")
    return float((1.0 - b_hat) * np.sum(z / pi) / n_population)


def mixed_calibration_si(z_sample, b_si, m_all_mean, m_sample_mean):
    """SI simplification zbar_MC = zbar_S + b_SI (mbar_pop - mbar_S), eq (10.40)."""
    z = _v(z_sample)
    return float(z.mean() + b_si * (m_all_mean - m_sample_mean))


def mc_variance_via_residuals(e, pi, n_population):
    """V_hat(zbar_MC) = V_hat of the HT mean of model residuals, eq (10.42).

    With-replacement pps form on the residual expansion e_k/pi_k: the
    variance of the HT total of e is sum(n e_k/pi_k - t_hat)^2/(n(n-1)),
    divided by N^2 for the mean.
    """
    e, pi = _v(e), _v(pi)
    n = e.size
    if e.shape != pi.shape or (pi <= 0).any() or n < 2 or n_population <= 0:
        raise ValueError("invalid inputs")
    t_hat = float(np.sum(e / pi))
    return float(np.sum((n * e / pi - t_hat) ** 2) / (n * (n - 1))
                 / n_population ** 2)


# ------------------------------------------------- ch 11: two-phase sampling


def twophase_stratified_variance(n1h, n1, s2_2h, n2h, zbar_2h, zbar_hat):
    """V_hat for two-phase sampling for stratification, eq (11.5)."""
    n1h, s2, n2h, zb = _v(n1h), _v(s2_2h), _v(n2h), _v(zbar_2h)
    if not (n1h.shape == s2.shape == n2h.shape == zb.shape) or n1 <= 0:
        raise ValueError("invalid inputs")
    first = np.sum((n1h / n1) ** 2 * s2 / n2h)
    second = np.sum((n1h / n1) * (zb - zbar_hat) ** 2) / n1
    return float(first + second)


def twophase_regression_variance(s2_z, n1, s2_e, n2, n_population):
    """V_hat = (1-n1/N) S2(z)/n1 + (1-n2/n1) S2(e)/n2, eqs (11.7)-(11.8)."""
    if min(s2_z, s2_e) < 0 or not 0 < n2 <= n1 <= n_population:
        raise ValueError("invalid inputs")
    return float((1.0 - n1 / n_population) * s2_z / n1
                 + (1.0 - n2 / n1) * s2_e / n2)


def s2_residuals(e, n):
    """S2_hat(e) = sum e_k^2 / (n-1), eq (11.8)."""
    e = _v(e)
    if e.size != n or n < 2:
        raise ValueError("invalid inputs")
    return float(np.sum(e ** 2) / (n - 1))


# ------------------------------------------------- ch 12: required sample size


def n_for_proportion_se(p_star, se_max):
    """n = (sqrt(p(1-p))/se_max)^2 + 1, eq (12.3)."""
    if not 0 < p_star < 1 or se_max <= 0:
        raise ValueError("invalid inputs")
    return float((math.sqrt(p_star * (1 - p_star)) / se_max) ** 2 + 1)


def n_for_mean_length(u_crit, s_star, l_max):
    """n = (u S*/(l_max/2))^2, eq (12.7)."""
    if min(u_crit, s_star, l_max) <= 0:
        raise ValueError("invalid inputs")
    return float((u_crit * s_star / (l_max / 2.0)) ** 2)


def n_for_cv(u_crit, cv_star, r_max):
    """n = (u cv*/r_max)^2, eq (12.10)."""
    if min(u_crit, cv_star, r_max) <= 0:
        raise ValueError("invalid inputs")
    return float((u_crit * cv_star / r_max) ** 2)


def n_for_proportion_length(u_crit, p_star, l_max):
    """n = (u sqrt(p(1-p))/(l_max/2))^2 + 1, eq (12.11)."""
    if not 0 < p_star < 1 or min(u_crit, l_max) <= 0:
        raise ValueError("invalid inputs")
    return float((u_crit * math.sqrt(p_star * (1 - p_star))
                  / (l_max / 2.0)) ** 2 + 1)


def n_design_effect(design_effect, n_si):
    """n(p, zbar) = sqrt(de) n(SI, pi), eq (12.14)."""
    if design_effect <= 0 or n_si <= 0:
        raise ValueError("invalid inputs")
    return float(math.sqrt(design_effect) * n_si)


def _log_beta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def beta_posterior_pdf(p, z, n, c, d):
    """Beta posterior f(p|z,n,c,d), eq (12.24)."""
    if not 0 < p < 1 or z < 0 or z > n or c <= 0 or d <= 0:
        raise ValueError("invalid inputs")
    a, b = z + c, n - z + d
    return math.exp((a - 1) * math.log(p) + (b - 1) * math.log(1 - p)
                    - _log_beta(a, b))


def _beta_cdf(x, a, b, n_grid=20001):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    # composite trapezoid on the density (smooth here: a, b >= 1);
    # hand-rolled -- numpy 2 removed trapz and the de-numpy campaign is
    # heading this way regardless
    xs = np.linspace(1e-12, x, n_grid)
    f = np.exp((a - 1) * np.log(xs) + (b - 1) * np.log1p(-xs)
               - _log_beta(a, b))
    dx = (xs[-1] - xs[0]) / (n_grid - 1)
    return float((f.sum() - 0.5 * (f[0] + f[-1])) * dx)


def beta_posterior_interval_prob(v, l, z, n, c, d):
    """Pr{p in (v, v+l)} under the Beta posterior, eqs (12.18), (12.27)."""
    if l < 0 or not 0 <= v <= 1:
        raise ValueError("invalid inputs")
    a, b = z + c, n - z + d
    return _beta_cdf(min(v + l, 1.0), a, b) - _beta_cdf(v, a, b)


def average_length_criterion(lengths, probs, l_max):
    """ALC check: sum l(z,n) f(z,n) <= l_max, eqs (12.17), (12.25)."""
    ls, ps = _v(lengths), _v(probs)
    if ls.shape != ps.shape or (ps < 0).any() or abs(ps.sum() - 1.0) > 1e-8:
        raise ValueError("probs must be a distribution")
    expected = float(np.dot(ls, ps))
    return {"expected_length": expected, "satisfied": expected <= l_max}


def average_coverage_criterion(coverages, probs, alpha):
    """ACC check: sum Pr{p in interval} f(z,n) >= 1 - alpha, eqs (12.19), (12.27)."""
    cs, ps = _v(coverages), _v(probs)
    if cs.shape != ps.shape or (ps < 0).any() or abs(ps.sum() - 1.0) > 1e-8:
        raise ValueError("probs must be a distribution")
    expected = float(np.dot(cs, ps))
    return {"expected_coverage": expected, "satisfied": expected >= 1 - alpha}


# --------------------------------- ch 13, 21: model-based / kriging kernels


def exponential_covariance(h, c0, c1, phi):
    """C(h) = c0 + c1 at h = 0; c1 exp(-h/phi) for h > 0 (ch 13/21 model).

    Companion of the exponential semivariogram of print eq (21.13):
    gamma(h) = 0 at h = 0; c0 + c1 exp(-h/phi) for h > 0.  [Note: web and
    print agree; the nugget c0 is discontinuous at the origin.]
    """
    h = _v(h)
    if (h < 0).any() or c0 < 0 or c1 <= 0 or phi <= 0:
        raise ValueError("invalid inputs")
    out = np.where(h == 0, c0 + c1, c1 * np.exp(-h / phi))
    return out if out.size > 1 else float(out[0])


def exponential_semivariogram(h, c0, c1, phi):
    """Exponential semivariogram, eq (21.13): 0 at h = 0, else c0 + c1 (1 - exp(-h/phi)).

    The display in the book prints c0 + c1 exp(-h/phi), but its own prose
    pins the correct form: "the semivariance goes asymptotically to a
    maximum" and "at three times the distance parameter is at 95% of the
    sill" -- 1 - exp(-3) = 0.950, whereas the printed form would decay.
    Display typo (eq 21.12's spherical bracket is sign-flipped the same
    way); implemented per the book's own numbers.
    """
    h = _v(h)
    if (h < 0).any() or c0 < 0 or c1 <= 0 or phi <= 0:
        raise ValueError("invalid inputs")
    out = np.where(h == 0, 0.0, c0 + c1 * (1.0 - np.exp(-h / phi)))
    return out if out.size > 1 else float(out[0])


def gaussian_process_model(mu, cov):
    """Spec container for Z(s) = mu(s) + eps, Cov = C(h), eqs (13.1), (21.1)-(21.2).

    Returns the mean vector and covariance with validity checks
    (symmetry, positive semi-definiteness).
    """
    mu = _v(mu)
    cov = np.asarray(cov, float)
    n = mu.size
    if cov.shape != (n, n) or not np.allclose(cov, cov.T, atol=1e-10):
        raise ValueError("cov must be symmetric n x n")
    eig = np.linalg.eigvalsh(cov)
    if eig.min() < -1e-8 * max(1.0, abs(eig).max()):
        raise ValueError("cov must be positive semi-definite")
    return {"mu": mu, "cov": cov, "n": n}


def mean_semivariance_stsi_variance(gamma_bar_h, weights, n_h):
    """E_xi{V_STSI} = sum w_h^2 gammabar_h / n_h, eq (13.5)."""
    g, w, n = _v(gamma_bar_h), _v(weights), _v(n_h)
    if not (g.shape == w.shape == n.shape) or (n <= 0).any() or (g < 0).any():
        raise ValueError("invalid inputs")
    return float(np.sum(w ** 2 * g / n))


def mean_semivariance_equal_area(gamma_bar_h, n):
    """Equal-area simplification E_xi{V_STSI} = (1/n^2) sum gammabar_h, eq (13.7)."""
    g = _v(gamma_bar_h)
    if n <= 0 or (g < 0).any():
        raise ValueError("invalid inputs")
    return float(g.sum() / n ** 2)


def optimal_allocation_variance(weights, s_h, c_h, n):
    """V = (1/n)(sum w S sqrt(c))(sum w S/sqrt(c)), eq (13.10)."""
    w, s, c = _v(weights), _v(s_h), _v(c_h)
    if not (w.shape == s.shape == c.shape) or (c <= 0).any() or n <= 0:
        raise ValueError("invalid inputs")
    return float(np.sum(w * s * np.sqrt(c)) * np.sum(w * s / np.sqrt(c)) / n)


def ospats_criterion_terms(w_h, s_h):
    """O = (sum w_h S_h)^2, eq (13.12)."""
    w, s = _v(w_h), _v(s_h)
    if w.shape != s.shape or (s < 0).any():
        raise ValueError("invalid inputs")
    return float(np.sum(w * s) ** 2)


def expected_squared_distance(zhat_i, zhat_j, r2, s2_i, s2_j, s2_ij):
    """E_xi[d2_ij] = (zhat_i - zhat_j)^2/R^2 + S2_i + S2_j - 2 S2_ij, eq (13.15)."""
    if r2 <= 0:
        raise ValueError("R^2 must be positive")
    return float((zhat_i - zhat_j) ** 2 / r2 + s2_i + s2_j - 2.0 * s2_ij)


def expected_stratum_variance(d2_matrix_upper_sum, n_h):
    """E_xi[S2_h(z)] = (1/N_h^2) sum_{i<j} E_xi[d2_ij].

    Print eq (13.16); the web edition leaves this display untagged.
    """
    if n_h <= 0:
        raise ValueError("N_h must be positive")
    return float(d2_matrix_upper_sum / n_h ** 2)


def ospats_objective(per_stratum_sums, n_population):
    """E_xi[O] = (1/N) sum_h (sum_{i<j} E_xi[d2_ij])^(1/2).

    Print eq (13.17) = web eq (13.16).
    """
    s = _v(per_stratum_sums)
    if (s < 0).any() or n_population <= 0:
        raise ValueError("invalid inputs")
    return float(np.sum(np.sqrt(s)) / n_population)


def kriging_weights_covariance(cov_ss, cov_s0):
    """Simple-kriging-with-constraint system of eq (21.4): solve for lambda, nu."""
    c = np.asarray(cov_ss, float)
    c0 = _v(cov_s0)
    n = c0.size
    if c.shape != (n, n):
        raise ValueError("shape mismatch")
    a = np.zeros((n + 1, n + 1))
    a[:n, :n] = c
    a[:n, n] = 1.0
    a[n, :n] = 1.0
    rhs = np.concatenate([c0, [1.0]])
    sol = np.linalg.solve(a, rhs)
    return {"lam": sol[:n], "nu": float(sol[n])}


def ok_variance_covariance_form(sigma2, lam, cov_s0, nu):
    """V_OK = sigma^2 - lambda^T c0 - nu, eq (21.8)."""
    lam, c0 = _v(lam), _v(cov_s0)
    if lam.shape != c0.shape:
        raise ValueError("shape mismatch")
    return float(sigma2 - np.dot(lam, c0) - nu)


def ok_variance_semivariance_form(lam, gamma_s0, nu):
    """V_OK = lambda^T gamma0 + nu, eq (21.11)."""
    lam, g0 = _v(lam), _v(gamma_s0)
    if lam.shape != g0.shape:
        raise ValueError("shape mismatch")
    return float(np.dot(lam, g0) + nu)


def gaussian_loglikelihood(z, mu, cov):
    """Multivariate normal density of eq (21.23), returned as log for stability."""
    gp = gaussian_process_model(mu, cov)
    z = _v(z)
    if z.size != gp["n"]:
        raise ValueError("shape mismatch")
    diff = z - gp["mu"]
    sign, logdet = np.linalg.slogdet(gp["cov"])
    if sign <= 0:
        raise ValueError("cov must be positive definite")
    quad = float(diff @ np.linalg.solve(gp["cov"], diff))
    n = gp["n"]
    return float(-0.5 * (n * math.log(2 * math.pi) + logdet + quad))


# --------------------------------- ch 14-16, 20, 24-26: assorted


def small_area_mb_mean(xbar_d, beta_hat, v_d):
    """Model-based small-area mean xbar_d^T beta + v_d, eq (14.15)."""
    x, b = _v(xbar_d), _v(beta_hat)
    if x.shape != b.shape:
        raise ValueError("shape mismatch")
    return float(np.dot(x, b) + v_d)


def trend_weights(times):
    """OLS trend weights w_j = (t_j - tbar)/sum(t - tbar)^2, eq (15.4)."""
    t = _v(times)
    if t.size < 2:
        raise ValueError("need >= 2 time points")
    d = t - t.mean()
    ss = float(np.dot(d, d))
    if ss == 0:
        raise ValueError("times must vary")
    return d / ss


def gls_estimator(x, c, zhat):
    """GLS zhat_GLS = (X^T C^-1 X)^-1 X^T C^-1 zhat, eq (15.10)."""
    x = np.asarray(x, float)
    c = np.asarray(c, float)
    z = _v(zhat)
    if x.ndim == 1:
        x = x[:, None]
    n = z.size
    if x.shape[0] != n or c.shape != (n, n):
        raise ValueError("shape mismatch")
    ci_x = np.linalg.solve(c, x)
    ci_z = np.linalg.solve(c, z)
    return np.linalg.solve(x.T @ ci_x, x.T @ ci_z)


def linear_model_prediction(beta0, beta1, x):
    """Simple linear working model Z = b0 + b1 x, eq (16.1)."""
    return float(beta0 + beta1 * float(x))


def ols_beta(x, z):
    """OLS beta_hat = (X^T X)^-1 X^T z (design matrix passed as-is), eq (20.2)."""
    x = np.asarray(x, float)
    z = _v(z)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[0] != z.size:
        raise ValueError("shape mismatch")
    return np.linalg.solve(x.T @ x, x.T @ z)


def ols_prediction_variance(sigma2_eps, x0, x):
    """V_hat(Z(s0)) = sig2 (1 + x0^T (X^T X)^-1 x0), eq (20.3)."""
    x = np.asarray(x, float)
    x0 = _v(x0)
    if x.ndim == 1:
        x = x[:, None]
    if sigma2_eps < 0 or x.shape[1] != x0.size:
        raise ValueError("invalid inputs")
    quad = float(x0 @ np.linalg.solve(x.T @ x, x0))
    return float(sigma2_eps * (1.0 + quad))


def nested_anova_prediction(mu, a_i, b_ij, c_ijk, eps):
    """Nested random-effects composition Z = mu + A + B + C + eps, eq (24.1)."""
    return float(mu + a_i + b_ij + c_ijk + eps)


def fisher_information_reml(a, da_list):
    """[I]_ij = 0.5 Tr(A^-1 dA_i A^-1 dA_j), eq (24.2)."""
    a = np.asarray(a, float)
    ai = np.linalg.inv(a)
    das = [np.asarray(d, float) for d in da_list]
    p = len(das)
    info = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            info[i, j] = 0.5 * np.trace(ai @ das[i] @ ai @ das[j])
    return info


def variance_of_kriging_variance(cov_theta, dv_dtheta):
    """VKV = sum_ij Cov(th_i,th_j) dV/dth_i dV/dth_j, eq (24.3)."""
    c = np.asarray(cov_theta, float)
    d = _v(dv_dtheta)
    if c.shape != (d.size, d.size):
        raise ValueError("shape mismatch")
    return float(d @ c @ d)


def augmented_kriging_variance(v_ok, e_tau2):
    """AKV = V_OK + E[tau^2], eq (24.4)."""
    if v_ok < 0 or e_tau2 < 0:
        raise ValueError("variances must be nonnegative")
    return float(v_ok + e_tau2)


def expected_tau2(cov_theta, dlam_dtheta, a):
    """E[tau^2] = sum_ij Cov(th_i,th_j) dlam^T/dth_i A dlam/dth_j, eq (24.5)."""
    c = np.asarray(cov_theta, float)
    a = np.asarray(a, float)
    ds = [np.atleast_1d(np.asarray(d, float)) for d in dlam_dtheta]
    p = len(ds)
    if c.shape != (p, p):
        raise ValueError("shape mismatch")
    out = 0.0
    for i in range(p):
        for j in range(p):
            out += c[i, j] * float(ds[i] @ a @ ds[j])
    return float(out)


def estimation_adjusted_criterion(akv, v_ok, vkv):
    """EAC = AKV + VKV/(2 V_OK), eq (24.6)."""
    if v_ok <= 0 or akv < 0 or vkv < 0:
        raise ValueError("invalid inputs")
    return float(akv + vkv / (2.0 * v_ok))


def classification_indicator(c_hat, c_true, u):
    """y_k = 1 if chat_k = c_k = u else 0, eq (25.8)."""
    return 1.0 if (c_hat == c_true == u) else 0.0


def iid_mean_variance(sigma2, n):
    """V(mu_hat) = sigma^2/n under iid, eq (26.2)."""
    if sigma2 < 0 or n <= 0:
        raise ValueError("invalid inputs")
    return float(sigma2 / n)


def autocorrelated_mean_variance(sigma2, n, rho_bar):
    """V(mu_hat) = (sigma^2/n)(1 + (n-1) rhobar), eq (26.3)."""
    base = iid_mean_variance(sigma2, n)
    out = base * (1.0 + (n - 1) * rho_bar)
    if out < 0:
        raise ValueError("rho_bar too negative for this n")
    return float(out)


def effective_sample_size(n, rho_bar):
    """n_eff = n/(1 + (n-1) rhobar), eq (26.4)."""
    den = 1.0 + (n - 1) * rho_bar
    if n <= 0 or den <= 0:
        raise ValueError("invalid inputs")
    return float(n / den)


def fpc_mean_variance(s2, n, n_population):
    """Design-based V(zbar_hat) = (1 - n/N) S2/n, eq (26.5)."""
    if s2 < 0 or not 0 < n <= n_population:
        raise ValueError("invalid inputs")
    return float((1.0 - n / n_population) * s2 / n)
