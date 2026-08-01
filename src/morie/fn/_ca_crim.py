"""Shared backend for the Advanced Statistics in Criminology shelf (ca* modules).

Every function implements a numbered equation from Weisburd, Wilson,
Wooditch & Britt (2022), Advanced Statistics in Criminology and Criminal
Justice, 5th ed., Springer (doi:10.1007/978-3-030-67738-1).  Equation
numbers are cited per function; front modules ca<ch>e<eq> route here.
"""

from __future__ import annotations

import math

import numpy as np

__all__: list = []  # internal backend; fronts re-export


# ---------------------------------------------------------------- ch 1-2: OLS


def linear_predictor(b0, bs, xs):
    """Structural model Yhat = b0 + sum(bk xk) (eqs 1.1, 2.21, 2.22, 4.5)."""
    bs = np.atleast_1d(np.asarray(bs, dtype=float))
    xs = np.atleast_1d(np.asarray(xs, dtype=float))
    if bs.shape != xs.shape:
        raise ValueError("bs and xs must have the same length")
    return float(b0 + np.dot(bs, xs))


def ols_simple(x, y):
    """Simple OLS via eqs (2.2)-(2.6): slope, intercept, r, t (both forms)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1-D and equal length")
    n = x.size
    if n < 3:
        raise ValueError("need n >= 3")
    xd = x - x.mean()
    yd = y - y.mean()
    sxx = float(np.dot(xd, xd))
    b1 = float(np.dot(xd, yd) / sxx)                      # eq (2.2)
    b0 = float(y.mean() - b1 * x.mean())                  # eq (2.3)
    r = float(np.dot(yd, xd) / math.sqrt(np.dot(yd, yd) * sxx))   # eq (2.4)
    resid = y - (b0 + b1 * x)
    se_b1 = math.sqrt(float(np.dot(resid, resid)) / (n - 2) / sxx)
    t_b1 = b1 / se_b1                                     # eq (2.5)
    t_r = r * math.sqrt((n - 2) / (1.0 - r * r))          # eq (2.6)
    return {"b1": b1, "b0": b0, "r": r, "se_b1": se_b1, "t": float(t_b1),
            "t_from_r": float(t_r), "df": n - 2, "n": n}


def ols_two_iv(r_y1, r_y2, r_12, s_y, s_1, s_2):
    """Two-IV OLS slopes from correlations, eqs (2.7)-(2.8)."""
    if not (-1 < r_12 < 1):
        raise ValueError("|r_12| must be < 1")
    den = 1.0 - r_12 * r_12
    b1 = (r_y1 - r_y2 * r_12) / den * (s_y / s_1)         # eq (2.7)
    b2 = (r_y2 - r_y1 * r_12) / den * (s_y / s_2)         # eq (2.8)
    return {"b1": float(b1), "b2": float(b2)}


def coef_t(b, se):
    """t = b / se_b, eq (2.9); also z of eqs (4.16), (6.x), (7.10)."""
    if se <= 0:
        raise ValueError("se must be positive")
    return float(b) / float(se)


def coef_ci(b, se, crit):
    """CI b -/+ se*crit, eqs (2.10), (4.17), (5.5), (11.38)-(11.39)."""
    if se <= 0:
        raise ValueError("se must be positive")
    return {"lower": float(b - se * crit), "upper": float(b + se * crit)}


def variance_partition(y, yhat):
    """Total/model/residual variances and R^2, eqs (2.11)-(2.14)."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    if y.shape != yhat.shape or y.ndim != 1:
        raise ValueError("y and yhat must be 1-D and equal length")
    n = y.size
    ybar = y.mean()
    ss_total = float(np.sum((y - ybar) ** 2))
    ss_model = float(np.sum((yhat - ybar) ** 2))
    ss_resid = float(np.sum((y - yhat) ** 2))
    return {
        "var_total": ss_total / n,        # eq (2.11)
        "var_model": ss_model / n,        # eq (2.12)
        "var_resid": ss_resid / n,        # eq (2.13)
        "ss_total": ss_total, "ss_model": ss_model, "ss_resid": ss_resid,
        "r2": ss_model / ss_total,        # eq (2.14)
        "n": n,
    }


def adjusted_r2(r2, n, k):
    """Adjusted R^2 = 1 - (1 - R^2)(n-1)/(n-k-1), eq (2.15)."""
    if n - k - 1 <= 0:
        raise ValueError("need n > k + 1")
    return float(1.0 - (1.0 - r2) * (n - 1) / (n - k - 1))


def f_overall_ss(ss_model, ss_resid, n, k):
    """Overall F from sums of squares, eq (2.16)."""
    df_model = k - 1
    df_resid = n - k
    if df_model <= 0 or df_resid <= 0:
        raise ValueError("invalid degrees of freedom")
    return {"f": float((ss_model / df_model) / (ss_resid / df_resid)),
            "df1": df_model, "df2": df_resid}


def f_overall_r2(r2, n, k):
    """Overall F from R^2: F = R^2 (n-k-1) / ((1-R^2) k), eq (2.17)."""
    if not 0 <= r2 < 1:
        raise ValueError("need 0 <= r2 < 1")
    if n - k - 1 <= 0:
        raise ValueError("need n > k + 1")
    return float(r2 * (n - k - 1) / ((1.0 - r2) * k))


def f_nested_ss(ss_resid_restricted, ss_resid_full, k_full, k_restricted, n):
    """Nested-model F from residual SS, eq (2.18)."""
    df_num = k_full - k_restricted
    df_den = n - k_full
    if df_num <= 0 or df_den <= 0:
        raise ValueError("invalid degrees of freedom")
    ms_resid_full = ss_resid_full / df_den
    f = (ss_resid_restricted - ss_resid_full) / df_num / ms_resid_full
    return {"f": float(f), "df1": df_num, "df2": df_den}


def f_nested_r2(r2_full, r2_restricted, k_full, k_restricted, n):
    """Nested-model F from the two R^2 values, eq (2.19)."""
    df_num = k_full - k_restricted
    df_den = n - k_full - 1
    if df_num <= 0 or df_den <= 0:
        raise ValueError("invalid degrees of freedom")
    f = ((r2_full - r2_restricted) / df_num) / ((1.0 - r2_full) / df_den)
    return {"f": float(f), "df1": df_num, "df2": df_den}


def beta_standardized(b, s_x, s_y):
    """Standardized regression coefficient Beta = b s_x / s_y, eq (2.20)."""
    if s_x <= 0 or s_y <= 0:
        raise ValueError("standard deviations must be positive")
    return float(b * s_x / s_y)


def dummy_subgroup_equation(b0, bs, dummy_index, dummy_value):
    """Fold a 0/1 dummy into the intercept, eqs (2.21)-(2.22)."""
    bs = list(np.atleast_1d(np.asarray(bs, dtype=float)))
    if not 0 <= dummy_index < len(bs):
        raise ValueError("dummy_index out of range")
    if dummy_value not in (0, 1):
        raise ValueError("dummy_value must be 0 or 1")
    b_d = bs.pop(dummy_index)
    return {"intercept": float(b0 + b_d * dummy_value), "slopes": bs}


# ------------------------------------------------------------- ch 3: collinearity


def tolerance(r2_x):
    """Tolerance = 1 - R^2_x, eq (3.1)."""
    if not 0 <= r2_x < 1:
        raise ValueError("need 0 <= r2_x < 1")
    return float(1.0 - r2_x)


def vif(r2_x):
    """VIF = 1 / (1 - R^2_x), eq (3.2)."""
    return float(1.0 / tolerance(r2_x))


# ------------------------------------------------------- ch 1 & 4: logistic


def logit(p):
    """logit(p) = ln(p / (1-p)), eqs (1.3), (4.1), (4.4), (5.1)."""
    p = float(p)
    if not 0 < p < 1:
        raise ValueError("p must be strictly between 0 and 1")
    return math.log(p / (1.0 - p))


def inv_logit(xb):
    """P(Y=1) = 1 / (1 + e^-Xb) = e^logit/(1+e^logit), eqs (4.2)-(4.3), (4.6)."""
    xb = float(xb)
    if xb >= 0:
        return 1.0 / (1.0 + math.exp(-xb))
    e = math.exp(xb)
    return e / (1.0 + e)


def odds(p):
    """odds = p / (1 - p), eq (4.7)."""
    p = float(p)
    if not 0 < p < 1:
        raise ValueError("p must be strictly between 0 and 1")
    return p / (1.0 - p)


def odds_ratio_unit_change(b):
    """OR for a one-unit change: odds(x+1)/odds(x) = e^b, eq (4.8)."""
    return math.exp(float(b))


def derivative_at_mean(ybar, b):
    """DM = ybar (1 - ybar) b, eq (4.9)."""
    if not 0 < ybar < 1:
        raise ValueError("ybar must be strictly between 0 and 1")
    return float(ybar * (1.0 - ybar) * b)


def beta_logistic(b, s, gelman=False):
    """Standardized logistic coefficient b*s (eq 4.10) or b*2s (eq 4.11)."""
    if s <= 0:
        raise ValueError("s must be positive")
    return float(b * (2.0 * s if gelman else s))


def percent_correct_predictions(n_correct, n_total):
    """Percent correct = 100 n_correct / n_total, eq (4.12)."""
    if n_total <= 0 or n_correct < 0 or n_correct > n_total:
        raise ValueError("need 0 <= n_correct <= n_total, n_total > 0")
    return 100.0 * n_correct / n_total


def cox_snell_r2(neg2ll_null, neg2ll_full, n):
    """Cox & Snell R^2 = 1 - e^[((-2LLnull)-(-2LLfull))/n]... eq (4.13).

    Arguments are the -2LL values as reported by software (positive numbers);
    the exponent is -(chi2)/n so the result lies in [0, 1).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    chi2 = float(neg2ll_null) - float(neg2ll_full)
    return 1.0 - math.exp(-chi2 / n)


def model_chi2(neg2ll_null, neg2ll_full):
    """Model chi^2 = (-2LLnull) - (-2LLfull), eqs (4.14), (6.5), (6.6)."""
    return float(neg2ll_null) - float(neg2ll_full)


def wald_statistic(b, se):
    """W = (b / se)^2, eq (4.15)."""
    return coef_t(b, se) ** 2


def likelihood_ratio_chi2(neg2ll_reduced, neg2ll_full):
    """LR chi^2 = (-2LLreduced) - (-2LLfull), eq (4.18)."""
    return float(neg2ll_reduced) - float(neg2ll_full)


# ------------------------------------------- ch 5: multinomial / ordinal logit


def multinomial_probs(xbs):
    """P(y=m) = e^{xb_m} / sum_j e^{xb_j}, eq (5.3) (softmax over categories)."""
    xbs = np.atleast_1d(np.asarray(xbs, dtype=float))
    z = np.exp(xbs - xbs.max())          # stable softmax, same ratios
    return z / z.sum()


def multinomial_conditional_or(xb_m, xb_n):
    """Conditional OR_{m/n} = e^{xb_m} / e^{xb_n}, eq (5.4)."""
    return math.exp(float(xb_m) - float(xb_n))


def cumulative_probability(probs, m):
    """P(y <= m) = sum_{j<=m} P(y=j), eq (5.6); m is 1-based."""
    probs = np.atleast_1d(np.asarray(probs, dtype=float))
    if not (probs >= 0).all() or abs(probs.sum() - 1.0) > 1e-8:
        raise ValueError("probs must be nonnegative and sum to 1")
    if not 1 <= m <= probs.size - 1:
        raise ValueError("m must be in 1..j-1")
    return float(probs[:m].sum())


def cumulative_logit(probs, m):
    """logit[P(y<=m)] = ln(P(y<=m)/P(y>m)), eq (5.7)."""
    c = cumulative_probability(probs, m)
    return logit(c)


def ordinal_logit(tau_m, bs, xs, parameterization="plus"):
    """Ordinal logit tau_m +/- Xb, eqs (5.8) ('plus') and (5.9) ('minus')."""
    xb = linear_predictor(0.0, bs, xs)
    if parameterization == "plus":
        return float(tau_m + xb)
    if parameterization == "minus":
        return float(tau_m - xb)
    raise ValueError("parameterization must be 'plus' or 'minus'")


# --------------------------------------------------------- ch 6: count models


def poisson_loglink_predict(b0, b1, x1):
    """y = e^{b0 + b1 x1}, eqs (6.1)-(6.4)."""
    return math.exp(float(b0) + float(b1) * float(x1))


def incidence_rate_ratio(b):
    """IRR = e^b (multiplicative change per unit of x; ch 6)."""
    return math.exp(float(b))


def poisson_offset_predict(b0, b1, x1, exposure):
    """ln(y) = b0 + b1 x1 + ln(exposure), eq (6.7): count = rate * exposure."""
    if exposure <= 0:
        raise ValueError("exposure must be positive")
    return math.exp(float(b0) + float(b1) * float(x1) + math.log(exposure))


def quasi_poisson_theta(y, yhat, k):
    """Over-dispersion theta = 1/(n-k-1) sum (y - yhat)^2 / yhat (ch 6)."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    if y.shape != yhat.shape or y.ndim != 1:
        raise ValueError("y and yhat must be 1-D and equal length")
    if (yhat <= 0).any():
        raise ValueError("yhat must be positive")
    n = y.size
    if n - k - 1 <= 0:
        raise ValueError("need n > k + 1")
    return float(np.sum((y - yhat) ** 2 / yhat) / (n - k - 1))


def quasi_poisson_se(se, theta):
    """se_quasi = se * sqrt(theta) (ch 6)."""
    if se <= 0 or theta <= 0:
        raise ValueError("se and theta must be positive")
    return float(se * math.sqrt(theta))


def negative_binomial_variance(mu, alpha):
    """Var(Y) = mu + mu^2 alpha, eq (6.8)."""
    if mu < 0 or alpha < 0:
        raise ValueError("mu and alpha must be nonnegative")
    return float(mu + mu * mu * alpha)


# ------------------------------------------------------ ch 7: multilevel


def grand_mean_model(y):
    """y_i = beta0 + e_i: intercept = mean, error variance, eq (7.1)."""
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or y.size < 2:
        raise ValueError("y must be 1-D with n >= 2")
    return {"intercept": float(y.mean()),
            "var_error": float(y.var(ddof=1)), "n": y.size}


def cluster_means_model(groups):
    """Per-cluster intercepts (means) of eqs (7.3)-(7.4); groups = list of arrays."""
    means = []
    for g in groups:
        g = np.asarray(g, dtype=float)
        if g.size == 0:
            raise ValueError("empty cluster")
        means.append(float(g.mean()))
    allv = np.concatenate([np.asarray(g, dtype=float) for g in groups])
    return {"cluster_means": means, "grand_mean": float(allv.mean()),
            "u_j": [m - float(allv.mean()) for m in means]}


def variance_components_sigma2_u(ms_between, ms_within, n_per_cluster):
    """sigma^2_u = (MSbetween - MSwithin) / n, eq (7.6)."""
    if n_per_cluster <= 0:
        raise ValueError("n_per_cluster must be positive")
    return float((ms_between - ms_within) / n_per_cluster)


def intraclass_correlation(sigma2_u, sigma2_e):
    """rho = sigma^2_u / (sigma^2_u + sigma^2_e), eq (7.7)."""
    if sigma2_u < 0 or sigma2_e <= 0:
        raise ValueError("need sigma2_u >= 0 and sigma2_e > 0")
    return float(sigma2_u / (sigma2_u + sigma2_e))


def lr_test_chi2(ll_null, ll_full):
    """LR chi^2 = -2 (LL1 - LL2) for nested models, eq (7.8)."""
    return float(-2.0 * (float(ll_null) - float(ll_full)))


def grand_mean_center(x):
    """x_ij - grand mean (ch 7 centering)."""
    x = np.asarray(x, dtype=float)
    return x - x.mean()


def cluster_mean_center(groups):
    """x_ij - cluster mean, per cluster (ch 7 centering; eq 7.11 input)."""
    return [np.asarray(g, dtype=float) - np.asarray(g, dtype=float).mean()
            for g in groups]


def multilevel_predict(b0, bs, xs, u_terms, e_ij=0.0):
    """Composed multilevel prediction: fixed part + random effects + error.

    Serves eqs (7.2), (7.3), (7.5), (7.9), (7.12)-(7.16): y = b0 + sum(b x)
    + sum(u) + e.
    """
    fixed = linear_predictor(b0, bs, xs) if len(np.atleast_1d(bs)) else float(b0)
    u_terms = np.atleast_1d(np.asarray(u_terms, dtype=float))
    return float(fixed + u_terms.sum() + e_ij)


# ---------------------------------------------------------- ch 8: power


def noncentrality_delta_generic(mean_pop, mean_null):
    """delta = mean test statistic (population) - mean (null), eq (8.1)."""
    return float(mean_pop) - float(mean_null)


def cohens_d_population(mu1, mu2, sigma):
    """d = (mu1 - mu2) / sigma, eq (8.2)."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return float((mu1 - mu2) / sigma)


def noncentrality_delta_d(d, n1, n2):
    """delta = d sqrt(n1 n2 / (n1 + n2)), ch 8."""
    if n1 <= 0 or n2 <= 0:
        raise ValueError("sample sizes must be positive")
    return float(d * math.sqrt(n1 * n2 / (n1 + n2)))


def _norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _noncentral_t_cdf(t, df, delta):
    """P(T_nc <= t): Johnson-Kramer normal approximation (adequate for df >= 10)."""
    num = t * (1.0 - 1.0 / (4.0 * df)) - delta
    den = math.sqrt(1.0 + t * t / (2.0 * df))
    return _norm_cdf(num / den)


def power_from_delta_t(delta, t_cv, df):
    """t_beta = delta - t_cv (eq 8.3) and power = P(T_nc(delta) > t_cv)."""
    if df <= 0:
        raise ValueError("df must be positive")
    beta = _noncentral_t_cdf(t_cv, df, delta)
    return {"t_beta": float(delta - t_cv), "beta": float(beta),
            "power": float(1.0 - beta)}


def cohens_f(sigma_means, sigma_error):
    """f = sigma_m / sigma_e, eq (8.4)."""
    if sigma_means < 0 or sigma_error <= 0:
        raise ValueError("need sigma_means >= 0 and sigma_error > 0")
    return float(sigma_means / sigma_error)


def noncentrality_lambda_f(f, n_total):
    """lambda = n f^2 for the F distribution, eq (8.5)."""
    if n_total <= 0:
        raise ValueError("n_total must be positive")
    return float(n_total * f * f)


def noncentrality_delta_r(r, n):
    """delta = r sqrt(n-2) / sqrt(1-r^2), eq (8.6)."""
    if not -1 < r < 1:
        raise ValueError("|r| must be < 1")
    if n <= 2:
        raise ValueError("need n > 2")
    return float(r * math.sqrt(n - 2) / math.sqrt(1.0 - r * r))


def r2_from_f2(f2):
    """R^2 = f^2 / (1 + f^2), eq (8.7)."""
    if f2 < 0:
        raise ValueError("f2 must be nonnegative")
    return float(f2 / (1.0 + f2))


# ------------------------------------------------------ ch 9: experiments


def treatment_b_confounded(r_yt, r_yx, r_tx, s_y, s_t):
    """b_t = (r_yt - r_yx r_tx) / (1 - r_tx^2) * (s_y / s_t), eq (9.1)."""
    if not -1 < r_tx < 1:
        raise ValueError("|r_tx| must be < 1")
    if s_y <= 0 or s_t <= 0:
        raise ValueError("standard deviations must be positive")
    return float((r_yt - r_yx * r_tx) / (1.0 - r_tx * r_tx) * (s_y / s_t))


def treatment_b_randomized(r_yt, s_y, s_t):
    """b_t = r_yt s_y / s_t when treatment is unconfounded, eq (9.2)."""
    if s_y <= 0 or s_t <= 0:
        raise ValueError("standard deviations must be positive")
    return float(r_yt * s_y / s_t)


def t_independent(m1, m2, s1, s2, n1, n2):
    """Independent-samples t with pooled variance, eqs (9.3), (9.11), (11.6)."""
    if n1 < 2 or n2 < 2:
        raise ValueError("need n >= 2 per group")
    if s1 < 0 or s2 < 0:
        raise ValueError("standard deviations must be nonnegative")
    df = n1 + n2 - 2
    pooled_var = ((n1 - 1) * s1 * s1 + (n2 - 1) * s2 * s2) / df
    se = math.sqrt(pooled_var * (n1 + n2) / (n1 * n2))
    if se == 0:
        raise ValueError("zero pooled variance")
    return {"t": float((m1 - m2) / se), "df": df, "se": float(se),
            "s_pooled": float(math.sqrt(pooled_var))}


def chi2_2x2(a, b, c, d):
    """chi^2 = (ad-bc)^2 (a+b+c+d) / [(a+b)(c+d)(a+c)(b+d)], eqs (9.4)."""
    for v in (a, b, c, d):
        if v < 0:
            raise ValueError("cell counts must be nonnegative")
    den = (a + b) * (c + d) * (a + c) * (b + d)
    if den == 0:
        raise ValueError("a marginal total is zero")
    num = (a * d - b * c) ** 2 * (a + b + c + d)
    return {"chi2": float(num / den), "df": 1}


def anova_oneway(groups):
    """One-way ANOVA MSbetween, MSwithin, F, eqs (9.5)-(9.7), (9.12)."""
    arrays = [np.asarray(g, dtype=float) for g in groups]
    a = len(arrays)
    if a < 2 or any(g.size < 2 for g in arrays):
        raise ValueError("need >= 2 groups with >= 2 obs each")
    allv = np.concatenate(arrays)
    n_total = allv.size
    grand = allv.mean()
    ss_between = sum(g.size * (g.mean() - grand) ** 2 for g in arrays)
    ss_within = sum(float(np.sum((g - g.mean()) ** 2)) for g in arrays)
    ms_between = ss_between / (a - 1)                       # eq (9.5)
    ms_within = ss_within / (n_total - a)                   # eq (9.6)
    return {"ms_between": float(ms_between), "ms_within": float(ms_within),
            "f": float(ms_between / ms_within),             # eq (9.7)
            "df1": a - 1, "df2": n_total - a,
            "ss_between": float(ss_between), "ss_within": float(ss_within)}


def repeated_measures_ms(data):
    """MSsubjects and MSBsubjects for two-way repeated measures, eqs (9.8)-(9.9).

    data: array (n_subjects, b_levels); all subjects belong to one A level here
    stacked per A group as a list -> pass list of (n_j, b) arrays.
    """
    groups = [np.asarray(g, dtype=float) for g in data]
    if any(g.ndim != 2 for g in groups):
        raise ValueError("each A group must be a 2-D (subjects x B) array")
    b = groups[0].shape[1]
    if any(g.shape[1] != b for g in groups) or b < 2:
        raise ValueError("all groups need the same b >= 2 B-levels")
    a = len(groups)
    n_total = sum(g.shape[0] for g in groups)
    # subject means and A-group means
    ss_subjects = 0.0
    ss_b_subjects = 0.0
    for g in groups:
        subj_mean = g.mean(axis=1)                       # ybar_ij
        a_mean = g.mean()                                # ybar_.j
        ss_subjects += b * float(np.sum((subj_mean - a_mean) ** 2))
        k_mean = g.mean(axis=0)                          # ybar_k within group
        resid = g - subj_mean[:, None] - k_mean[None, :] + a_mean
        ss_b_subjects += float(np.sum(resid ** 2))
    ms_subjects = ss_subjects / (n_total - a)               # eq (9.8)
    ms_b_subjects = ss_b_subjects / ((n_total - a) * (b - 1))   # eq (9.9)
    return {"ms_subjects": float(ms_subjects),
            "ms_b_subjects": float(ms_b_subjects),
            "df_subjects": n_total - a,
            "df_b_subjects": (n_total - a) * (b - 1)}


def t_paired(differences):
    """Paired t = dbar / sqrt(s_d^2 / df), eq (9.10); df = n_pairs - 1."""
    d = np.asarray(differences, dtype=float)
    if d.ndim != 1 or d.size < 2:
        raise ValueError("need >= 2 paired differences")
    df = d.size - 1
    var_d = d.var(ddof=1)
    if var_d == 0:
        raise ValueError("zero variance of differences")
    return {"t": float(d.mean() / math.sqrt(var_d / d.size)), "df": df}


def anova_randomized_block(y, treatment, block):
    """Two-way additive ANOVA y = mu + alpha_j + beta_k + e, eq (9.13).

    Returns treatment F with block variability removed from the error term.
    """
    y = np.asarray(y, dtype=float)
    treatment = np.asarray(treatment)
    block = np.asarray(block)
    if not (y.shape == treatment.shape == block.shape) or y.ndim != 1:
        raise ValueError("y, treatment, block must be 1-D and equal length")
    t_levels = np.unique(treatment)
    b_levels = np.unique(block)
    a, b = t_levels.size, b_levels.size
    if a < 2 or b < 2:
        raise ValueError("need >= 2 treatment levels and >= 2 blocks")
    grand = y.mean()
    ss_treat = sum((y[treatment == t]).size * (y[treatment == t].mean() - grand) ** 2
                   for t in t_levels)
    ss_block = sum((y[block == k]).size * (y[block == k].mean() - grand) ** 2
                   for k in b_levels)
    ss_total = float(np.sum((y - grand) ** 2))
    ss_resid = ss_total - ss_treat - ss_block
    df_resid = y.size - a - b + 1
    if df_resid <= 0:
        raise ValueError("insufficient residual degrees of freedom")
    ms_treat = ss_treat / (a - 1)
    ms_resid = ss_resid / df_resid
    return {"f_treatment": float(ms_treat / ms_resid),
            "ms_treatment": float(ms_treat), "ms_resid": float(ms_resid),
            "ss_block": float(ss_block), "df1": a - 1, "df2": df_resid}


# ---------------------------------------------------------- ch 10: PSM


def psm_standardized_bias(mean_t, mean_c, s_t, s_c):
    """Bias = 100 (xbar_t - xbar_c) / sqrt((s_t^2 + s_c^2)/2), eq (10.1)."""
    if s_t < 0 or s_c < 0 or (s_t == 0 and s_c == 0):
        raise ValueError("need nonnegative s with at least one positive")
    return float(100.0 * (mean_t - mean_c)
                 / math.sqrt((s_t * s_t + s_c * s_c) / 2.0))


# ------------------------------------------------------ ch 11: meta-analysis


def pooled_sd(s1, s2, n1, n2):
    """s_pooled = sqrt(((n1-1)s1^2 + (n2-1)s2^2)/(n1+n2-2)), eq (11.2)."""
    if n1 < 2 or n2 < 2:
        raise ValueError("need n >= 2 per group")
    return math.sqrt(((n1 - 1) * s1 * s1 + (n2 - 1) * s2 * s2)
                     / (n1 + n2 - 2))


def cohens_d_sample(m1, m2, s1, s2, n1, n2):
    """d = (x1 - x2) / s_pooled, eq (11.1)."""
    sp = pooled_sd(s1, s2, n1, n2)
    if sp == 0:
        raise ValueError("zero pooled SD")
    return float((m1 - m2) / sp)


def hedges_j(n1, n2):
    """Small-sample correction J = 1 - 3/(4(n1+n2)-9), eq (11.3)."""
    den = 4 * (n1 + n2) - 9
    if den <= 0:
        raise ValueError("samples too small")
    return float(1.0 - 3.0 / den)


def hedges_g(d, n1, n2):
    """g = J d, eq (11.4)."""
    return float(hedges_j(n1, n2) * d)


def d_from_t(t, n1, n2):
    """d = t sqrt((n1+n2)/(n1 n2)), eq (11.5)."""
    if n1 <= 0 or n2 <= 0:
        raise ValueError("sample sizes must be positive")
    return float(t * math.sqrt((n1 + n2) / (n1 * n2)))


def se_g(g, n1, n2):
    """se_g = sqrt((n1+n2)/(n1 n2) + g^2/(2(n1+n2))), eq (11.7)."""
    if n1 <= 0 or n2 <= 0:
        raise ValueError("sample sizes must be positive")
    return math.sqrt((n1 + n2) / (n1 * n2) + g * g / (2.0 * (n1 + n2)))


def risk_ratio(a, b, c, d):
    """RR = [a/(a+b)] / [c/(c+d)], eq (11.8)."""
    if min(a, b, c, d) < 0 or a + b == 0 or c + d == 0 or c == 0:
        raise ValueError("invalid cell counts")
    return float((a / (a + b)) / (c / (c + d)))


def se_log_rr(p1, p2, n1, n2):
    """se_ln(RR) = sqrt((1-p1)/(n1 p1) + (1-p2)/(n2 p2)), eq (11.9)."""
    if not (0 < p1 < 1 and 0 < p2 < 1) or n1 <= 0 or n2 <= 0:
        raise ValueError("need 0 < p < 1 and positive n")
    return math.sqrt((1 - p1) / (n1 * p1) + (1 - p2) / (n2 * p2))


def odds_ratio_2x2(a, b, c, d):
    """OR = ad / bc, eq (11.10)."""
    if min(a, b, c, d) < 0 or b * c == 0:
        raise ValueError("invalid cell counts")
    return float(a * d / (b * c))


def se_log_or(a, b, c, d):
    """se_ln(OR) = sqrt(1/a + 1/b + 1/c + 1/d), eq (11.11)."""
    if min(a, b, c, d) <= 0:
        raise ValueError("all cells must be positive")
    return math.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)


def fisher_z(r):
    """Zr = 0.5 ln((1+r)/(1-r)), eq (11.12)."""
    if not -1 < r < 1:
        raise ValueError("|r| must be < 1")
    return 0.5 * math.log((1.0 + r) / (1.0 - r))


def se_fisher_z(n):
    """se_Zr = 1 / sqrt(n - 3), eq (11.13)."""
    if n <= 3:
        raise ValueError("need n > 3")
    return 1.0 / math.sqrt(n - 3)


def r_from_fisher_z(z):
    """r = (e^{2Zr} - 1) / (e^{2Zr} + 1), eq (11.14)."""
    return math.tanh(float(z))


LOGISTIC_SD = math.sqrt(math.pi ** 2 / 3.0)   # eq (11.15)


def d_from_log_or(ln_or, method="logit"):
    """d from ln(OR): /sqrt(pi^2/3) (eq 11.16) or Cox /1.65 (eq 11.18)."""
    if method == "logit":
        return float(ln_or / LOGISTIC_SD)
    if method == "cox":
        return float(ln_or / 1.65)
    raise ValueError("method must be 'logit' or 'cox'")


def se_d_from_se_log_or(se_ln_or, method="logit"):
    """se_d from se_ln(OR), eqs (11.17) and (11.19)."""
    if se_ln_or <= 0:
        raise ValueError("se must be positive")
    div = LOGISTIC_SD if method == "logit" else 1.65 if method == "cox" else None
    if div is None:
        raise ValueError("method must be 'logit' or 'cox'")
    return float(math.sqrt(se_ln_or ** 2 / div ** 2))


def _probit(p):
    """Inverse standard normal CDF (Acklam rational approximation, |err|<1.2e-9)."""
    if not 0 < p < 1:
        raise ValueError("p must be strictly between 0 and 1")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) \
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q \
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) \
        / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def d_probit(p1, p2):
    """d = probit(p1) - probit(p2), eq (11.20)."""
    return float(_probit(p1) - _probit(p2))


def se_d_probit(p1, p2, n1, n2):
    """se_d for the probit method, eq (11.21)."""
    if n1 <= 0 or n2 <= 0:
        raise ValueError("sample sizes must be positive")
    z1, z2 = _probit(p1), _probit(p2)
    term1 = 2.0 * math.pi * p1 * (1 - p1) * math.exp(z1 * z1) / n1
    term2 = 2.0 * math.pi * p2 * (1 - p2) * math.exp(z2 * z2) / n2
    return math.sqrt(term1 + term2)


def d_from_r_pointbiserial(r):
    """d = 2r / sqrt(1 - r^2), eq (11.22)."""
    if not -1 < r < 1:
        raise ValueError("|r| must be < 1")
    return float(2.0 * r / math.sqrt(1.0 - r * r))


def se_d_from_se_r(r, se_r):
    """se_d = sqrt(4 se_r^2 / (1 - r^2)^3), eq (11.23)."""
    if not -1 < r < 1 or se_r <= 0:
        raise ValueError("need |r| < 1 and se_r > 0")
    return math.sqrt(4.0 * se_r ** 2 / (1.0 - r * r) ** 3)


def log_or_from_d(d, method="logit"):
    """ln(OR) = d/0.551 (eq 11.24, logit) or d/0.606 (eq 11.26, Cox)."""
    if method == "logit":
        return float(d / 0.551)
    if method == "cox":
        return float(d / 0.606)
    raise ValueError("method must be 'logit' or 'cox'")


def se_log_or_from_se_d(se_d, method="logit"):
    """se_ln(OR) = sqrt(se_d^2/0.551^2) (eq 11.25) or /0.606^2 (eq 11.27)."""
    if se_d <= 0:
        raise ValueError("se must be positive")
    div = 0.551 if method == "logit" else 0.606 if method == "cox" else None
    if div is None:
        raise ValueError("method must be 'logit' or 'cox'")
    return float(math.sqrt(se_d ** 2 / div ** 2))


def or_from_rr(rr, p2):
    """OR = RR p2 (1-p2) / [p2 (1 - RR p2)], eq (11.28)."""
    if not 0 < p2 < 1 or rr <= 0 or rr * p2 >= 1:
        raise ValueError("need 0 < p2 < 1, rr > 0, rr*p2 < 1")
    return float(rr * p2 * (1 - p2) / (p2 * (1 - rr * p2)))


def rr_from_or(or_, p2):
    """RR = OR / (1 - p2 + p2 OR), eq (11.29)."""
    if not 0 < p2 < 1 or or_ <= 0:
        raise ValueError("need 0 < p2 < 1 and OR > 0")
    return float(or_ / (1 - p2 + p2 * or_))


def r_from_d(d, n1=None, n2=None):
    """r = d / sqrt(d^2 + (n1+n2)^2/(n1 n2)) (eq 11.30); equal-n: /sqrt(d^2+4) (eq 11.31)."""
    if n1 is None or n2 is None:
        return float(d / math.sqrt(d * d + 4.0))
    if n1 <= 0 or n2 <= 0:
        raise ValueError("sample sizes must be positive")
    h = (n1 + n2) ** 2 / (n1 * n2)
    return float(d / math.sqrt(d * d + h))


def se_r_from_se_d(d, se_d, n1=None, n2=None):
    """se_r from se_d, eq (11.32); equal-n simplification eq (11.33)."""
    if se_d <= 0:
        raise ValueError("se must be positive")
    h = 4.0 if (n1 is None or n2 is None) else (n1 + n2) ** 2 / (n1 * n2)
    return math.sqrt(h * se_d ** 2 / (d * d + h) ** 3)


def fixed_effect_weight(se):
    """w = 1 / se^2, eq (11.34)."""
    if se <= 0:
        raise ValueError("se must be positive")
    return 1.0 / (se * se)


def random_effects_weight(se, tau2):
    """w = 1 / (se^2 + tau^2), eq (11.43)."""
    if se <= 0 or tau2 < 0:
        raise ValueError("need se > 0 and tau2 >= 0")
    return 1.0 / (se * se + tau2)


def mean_effect_size(ys, ws):
    """ybar = sum(w y) / sum(w), eq (11.35); se and z, eqs (11.36)-(11.37)."""
    ys = np.asarray(ys, dtype=float)
    ws = np.asarray(ws, dtype=float)
    if ys.shape != ws.shape or ys.ndim != 1 or ys.size == 0:
        raise ValueError("ys and ws must be 1-D and equal length")
    if (ws <= 0).any():
        raise ValueError("weights must be positive")
    wsum = ws.sum()
    ybar = float(np.dot(ws, ys) / wsum)
    se = float(math.sqrt(1.0 / wsum))                      # eq (11.36)
    return {"mean": ybar, "se": se, "z": ybar / se, "k": ys.size}


def q_statistic(ys, ws):
    """Q = sum w (y - ybar)^2, eq (11.40) == computational eq (11.41)."""
    ys = np.asarray(ys, dtype=float)
    ws = np.asarray(ws, dtype=float)
    m = mean_effect_size(ys, ws)["mean"]
    q_def = float(np.dot(ws, (ys - m) ** 2))               # eq (11.40)
    q_comp = float(np.dot(ws, ys ** 2) - np.dot(ws, ys) ** 2 / ws.sum())  # eq (11.41)
    return {"q": q_def, "q_computational": q_comp, "df": ys.size - 1}


def i_squared(q, df):
    """I^2 = (Q - df)/Q * 100 (floored at 0), eq (11.42)."""
    if q <= 0 or df < 0:
        raise ValueError("need Q > 0 and df >= 0")
    return float(max(0.0, (q - df) / q * 100.0))


def tau2_dersimonian_laird(ys, ws_fixed):
    """DerSimonian-Laird tau^2 = (Q - df)/(sum w - sum w^2/sum w), eq (11.44)."""
    ws = np.asarray(ws_fixed, dtype=float)
    qres = q_statistic(ys, ws)
    c = ws.sum() - float(np.dot(ws, ws)) / ws.sum()
    if c <= 0:
        raise ValueError("degenerate weights")
    return float(max(0.0, (qres["q"] - qres["df"]) / c))


def q_within_between(ys_by_group, ws_by_group):
    """Analog-to-the-ANOVA: Qwithin (eq 11.45) and Qbetween = Q - Qwithin (eq 11.46)."""
    if len(ys_by_group) != len(ws_by_group) or len(ys_by_group) < 2:
        raise ValueError("need >= 2 groups with matching weights")
    q_within = 0.0
    all_y, all_w = [], []
    for ys, ws in zip(ys_by_group, ws_by_group):
        q_within += q_statistic(ys, ws)["q"]
        all_y.extend(np.asarray(ys, dtype=float))
        all_w.extend(np.asarray(ws, dtype=float))
    q_total = q_statistic(all_y, all_w)["q"]
    return {"q_within": float(q_within),
            "q_between": float(q_total - q_within),
            "q_total": float(q_total),
            "df_within": len(all_y) - len(ys_by_group),
            "df_between": len(ys_by_group) - 1}


# ---------------------------------------------------------- ch 12: spatial


def morans_i(x, w):
    """Moran's I = n sum_ij w_ij (x_i - xbar)(x_j - xbar) / (W sum (x-xbar)^2), eq (12.1)."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    n = x.size
    if x.ndim != 1 or w.shape != (n, n):
        raise ValueError("x must be 1-D with w an n x n matrix")
    if n < 2:
        raise ValueError("need n >= 2")
    xd = x - x.mean()
    denom_ss = float(np.dot(xd, xd))
    w_sum = float(w.sum())
    if denom_ss == 0 or w_sum == 0:
        raise ValueError("degenerate input (constant x or zero weights)")
    num = float(xd @ w @ xd)
    return float(n * num / (w_sum * denom_ss))


def morans_i_expected(n):
    """E(I) = -1/(n-1), eq (12.2)."""
    if n < 2:
        raise ValueError("need n >= 2")
    return -1.0 / (n - 1)


def ols_matrix(x, y):
    """OLS y = X beta + e via normal equations, eq (12.3); X gains an intercept."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim != 1 or x.shape[0] != y.size:
        raise ValueError("x rows must match y length")
    design = np.column_stack([np.ones(y.size), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    return {"beta": beta, "residuals": resid}


def spatial_lag_reduced_form(rho, w, xb, e):
    """SAR y = rho W y + X beta + e solved as y = (I - rho W)^{-1}(Xb + e), eq (12.4)."""
    w = np.asarray(w, dtype=float)
    xb = np.asarray(xb, dtype=float)
    e = np.asarray(e, dtype=float)
    n = xb.size
    if w.shape != (n, n) or e.size != n:
        raise ValueError("shape mismatch")
    a = np.eye(n) - rho * w
    cond = np.linalg.cond(a)
    if not np.isfinite(cond) or cond > 1e12:
        raise ValueError("(I - rho W) is singular or near-singular")
    return np.linalg.solve(a, xb + e)
