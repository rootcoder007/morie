"""Shared backend for the Analysis of Categorical Data with R shelf.

Every function implements a numbered equation from Bilder, C. R. &
Loughin, T. M. (2025), Analysis of Categorical Data with R, 2nd ed.,
Chapman & Hall/CRC (the PDF in the project library is the spec; equation
numbers cited per function).
"""

from __future__ import annotations

import itertools
import math

import numpy as np

__all__: list = []


def _v(x):
    return np.atleast_1d(np.asarray(x, dtype=float))


def _logit(p):
    if not 0 < p < 1:
        raise ValueError("p must be strictly between 0 and 1")
    return math.log(p / (1.0 - p))


def _expit(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


# ------------------------------------------------------------- ch 1: binomial


def binomial_pmf(w, n, p):
    """P(W = w) = C(n, w) p^w (1-p)^(n-w), eq (1.1)."""
    if not 0 <= w <= n or not 0 <= p <= 1:
        raise ValueError("need 0 <= w <= n and 0 <= p <= 1")
    return float(math.comb(n, w) * p ** w * (1.0 - p) ** (n - w))


def mle_variance_pi(pi_hat, n):
    """Var_hat(pi_hat) = pi_hat (1 - pi_hat) / n, eq (1.3)."""
    if not 0 <= pi_hat <= 1 or n <= 0:
        raise ValueError("invalid inputs")
    return float(pi_hat * (1.0 - pi_hat) / n)


def wilson_interval(w, n, z):
    """Wilson (score) interval, eq (1.4): pi_tilde +/- (z sqrt(n)/(n+z^2)) sqrt(...)."""
    if not 0 <= w <= n or n <= 0 or z <= 0:
        raise ValueError("invalid inputs")
    p_hat = w / n
    p_tilde = (w + z * z / 2.0) / (n + z * z)
    half = (z * math.sqrt(n) / (n + z * z)) \
        * math.sqrt(p_hat * (1 - p_hat) + z * z / (4.0 * n))
    return {"estimate": p_tilde, "lower": p_tilde - half,
            "upper": p_tilde + half}


def beta_pdf(v, a, b):
    """Beta density Gamma(a+b)/(Gamma(a)Gamma(b)) v^(a-1)(1-v)^(b-1), eq (1.5)."""
    if not 0 < v < 1 or a <= 0 or b <= 0:
        raise ValueError("invalid inputs")
    ln = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
          + (a - 1) * math.log(v) + (b - 1) * math.log(1 - v))
    return math.exp(ln)


def true_confidence_level(n, p, interval_fn):
    """C(pi) = sum_w I(w) C(n,w) p^w (1-p)^(n-w), eq (1.6).

    interval_fn(w, n) must return (lower, upper); I(w) checks containment.
    """
    if n <= 0 or not 0 < p < 1:
        raise ValueError("invalid inputs")
    c = 0.0
    for w in range(n + 1):
        lo, hi = interval_fn(w, n)
        if lo <= p <= hi:
            c += binomial_pmf(w, n, p)
    return float(c)


def pearson_chi2_two_groups(w1, n1, w2, n2):
    """Two-group Pearson chi-square with pooled pi_bar, eq (1.7)."""
    if min(w1, w2) < 0 or w1 > n1 or w2 > n2 or min(n1, n2) <= 0:
        raise ValueError("invalid inputs")
    p_bar = (w1 + w2) / (n1 + n2)
    if p_bar in (0.0, 1.0):
        raise ValueError("degenerate pooled proportion")
    x2 = 0.0
    for w, n in ((w1, n1), (w2, n2)):
        x2 += (w - n * p_bar) ** 2 / (n * p_bar)
        x2 += (n - w - n * (1 - p_bar)) ** 2 / (n * (1 - p_bar))
    return {"x2": float(x2), "df": 1, "pi_bar": p_bar}


def lrt_two_groups(w1, n1, w2, n2):
    """-2 log(Lambda) for two binomial proportions, eq (1.8)."""
    p_bar = (w1 + w2) / (n1 + n2)
    p1 = w1 / n1
    p2 = w2 / n2

    def term(w, n, p_hat):
        out = 0.0
        if w > 0:
            out += w * math.log(p_bar / p_hat)
        if n - w > 0:
            out += (n - w) * math.log((1 - p_bar) / (1 - p_hat))
        return out

    stat = -2.0 * (term(w1, n1, p1) + term(w2, n2, p2))
    return {"stat": float(stat), "df": 1}


def or_wald_interval(w1, n1, w2, n2, z):
    """Wald CI for the odds ratio, eq (1.10)."""
    if min(w1, w2) <= 0 or w1 >= n1 or w2 >= n2:
        raise ValueError("need interior cell counts")
    or_hat = (w1 / (n1 - w1)) / (w2 / (n2 - w2))
    se = math.sqrt(1 / w1 + 1 / (n1 - w1) + 1 / w2 + 1 / (n2 - w2))
    return {"or": float(or_hat),
            "lower": float(or_hat * math.exp(-z * se)),
            "upper": float(or_hat * math.exp(z * se)), "se_log": se}


# --------------------------------------------------------- ch 2: logistic


def bernoulli_likelihood(pis, ys):
    """L = prod pi_i^y_i (1 - pi_i)^(1-y_i), eq (2.1) (returned as log)."""
    pis, ys = _v(pis), _v(ys)
    if pis.shape != ys.shape or not np.isin(ys, (0.0, 1.0)).all() \
            or ((pis <= 0) | (pis >= 1)).any():
        raise ValueError("invalid inputs")
    return float(np.sum(ys * np.log(pis) + (1 - ys) * np.log(1 - pis)))


def logistic_pi(b0, bs, xs):
    """pi = exp(b0 + sum b x)/(1 + exp(...)), eq (2.2)."""
    bs, xs = _v(bs), _v(xs)
    if bs.shape != xs.shape:
        raise ValueError("shape mismatch")
    return _expit(b0 + float(np.dot(bs, xs)))


def logit_form(p):
    """log(pi/(1-pi)) = b0 + b1 x1 + ..., eq (2.3): returns the logit."""
    return _logit(p)


def logistic_loglik(b, x, y):
    """Log-likelihood sum y xb - log(1 + exp(xb)), eqs (2.4)-(2.5)."""
    x = np.asarray(x, float)
    y = _v(y)
    b = _v(b)
    if x.ndim == 1:
        x = x[:, None]
    xb = x @ b
    return float(np.sum(y * xb - np.logaddexp(0.0, xb)))


def logistic_mle(x, y, max_iter=100, tol=1e-10):
    """Newton-Raphson MLE for logistic regression (design x incl. intercept)."""
    x = np.asarray(x, float)
    y = _v(y)
    if x.ndim == 1:
        x = x[:, None]
    n, p = x.shape
    b = np.zeros(p)
    for _ in range(max_iter):
        pi = 1.0 / (1.0 + np.exp(-(x @ b)))
        wdiag = pi * (1 - pi)
        grad = x.T @ (y - pi)
        hess = (x * wdiag[:, None]).T @ x
        step = np.linalg.solve(hess, grad)
        b = b + step
        if np.abs(step).max() < tol:
            break
    pi = 1.0 / (1.0 + np.exp(-(x @ b)))
    cov = np.linalg.inv((x * (pi * (1 - pi))[:, None]).T @ x)
    return {"beta": b, "cov": cov, "loglik": logistic_loglik(b, x, y)}


def lrt_statistic(loglik_null, loglik_full):
    """-2 log(Lambda) = -2 (LL0 - LLa), eqs (2.6)-(2.7)."""
    stat = -2.0 * (float(loglik_null) - float(loglik_full))
    return float(stat)


def residual_deviance(pis, ys):
    """-2 sum y log(pi/y) + (1-y) log((1-pi)/(1-y)) vs saturated, eqs (2.8)-(2.9)."""
    pis, ys = _v(pis), _v(ys)
    if pis.shape != ys.shape:
        raise ValueError("shape mismatch")
    dev = 0.0
    for p, y in zip(pis, ys):
        if not 0 < p < 1:
            raise ValueError("pi must be in (0, 1)")
        if y > 0:
            dev += y * math.log(p / y)
        if 1 - y > 0:
            dev += (1 - y) * math.log((1 - p) / (1 - y))
    return float(-2.0 * dev)


def or_ci_logistic(b1, var_b1, c, z):
    """Wald CI for OR = exp(c b1): exp(c b1 -/+ c z sqrt(var)), eq (2.11).

    Also serves the ordinal odds-ratio interval whose OCR fragment named
    stub 3e50 (a printed confidence bound "3.50", not an equation).
    """
    if var_b1 < 0 or z <= 0:
        raise ValueError("invalid inputs")
    half = abs(c) * z * math.sqrt(var_b1)
    return {"or": math.exp(c * b1), "lower": math.exp(c * b1 - half),
            "upper": math.exp(c * b1 + half)}


def linear_predictor_variance(xs, cov):
    """Var(b0 + b1 x1 + ... + bp xp) = sum x_i x_j Cov(b_i, b_j), eqs (2.14), (2.16).

    xs includes the leading 1 for the intercept.
    """
    xs = _v(xs)
    cov = np.asarray(cov, float)
    if cov.shape != (xs.size, xs.size):
        raise ValueError("shape mismatch")
    return float(xs @ cov @ xs)


def pi_wald_interval(xb, var_xb, z):
    """Wald CI for pi via the logit scale, eq (2.15)."""
    if var_xb < 0 or z <= 0:
        raise ValueError("invalid inputs")
    half = z * math.sqrt(var_xb)
    return {"pi": _expit(xb), "lower": _expit(xb - half),
            "upper": _expit(xb + half)}


def interaction_logit(b, x1, x2, z1, z2):
    """Two-categorical interaction logit of eq (2.22).

    b = (b0, b1, b2, b3, b4, b5, b6, b7, b8) for logit = b0 + b1 x1 +
    b2 x2 + b3 z1 + b4 z2 + b5 x1 z1 + b6 x1 z2 + b7 x2 z1 + b8 x2 z2.
    """
    b = _v(b)
    if b.size != 9:
        raise ValueError("need 9 coefficients")
    terms = np.array([1, x1, x2, z1, z2, x1 * z1, x1 * z2, x2 * z1,
                      x2 * z2], float)
    return float(np.dot(b, terms))


# ------------------------------------------------- ch 3: multicategory


def multinomial_pmf(counts, probs):
    """P(N1 = n1, ..., NJ = nJ) = n!/(prod n_j!) prod p_j^n_j, eq (3.1)."""
    counts = np.asarray(counts)
    probs = _v(probs)
    if counts.shape != probs.shape or (counts < 0).any() \
            or abs(probs.sum() - 1.0) > 1e-8 or (probs < 0).any():
        raise ValueError("invalid inputs")
    n = int(counts.sum())
    ln = math.lgamma(n + 1)
    for c, p in zip(counts, probs):
        ln -= math.lgamma(int(c) + 1)
        if c > 0:
            if p == 0:
                return 0.0
            ln += c * math.log(p)
    return math.exp(ln)


def contingency_pmf(count_table, prob_table):
    """One-multinomial contingency PMF over I x J cells, eq (3.2)."""
    c = np.asarray(count_table)
    p = np.asarray(prob_table, float)
    if c.shape != p.shape or c.ndim != 2:
        raise ValueError("need matching 2-D tables")
    return multinomial_pmf(c.ravel(), p.ravel())


def product_multinomial_pmf(count_table, cond_prob_table):
    """Product-multinomial PMF: prod over rows of row-multinomials, eq (3.3)."""
    c = np.asarray(count_table)
    p = np.asarray(cond_prob_table, float)
    if c.shape != p.shape or c.ndim != 2:
        raise ValueError("need matching 2-D tables")
    out = 1.0
    for i in range(c.shape[0]):
        out *= multinomial_pmf(c[i], p[i])
    return float(out)


def baseline_logit(bj0, bjs, xs):
    """log(pi_j / pi_1) = bj0 + bj1 x1 + ... + bjp xp, eqs (3.4), (3.10)."""
    bjs, xs = _v(bjs), _v(xs)
    if bjs.shape != xs.shape:
        raise ValueError("shape mismatch")
    return float(bj0 + np.dot(bjs, xs))


def baseline_probs(logits_2_to_J):
    """pi_j from baseline logits (pi_1 reference), companion of eq (3.4)."""
    ls = _v(logits_2_to_J)
    z = np.concatenate([[0.0], ls])
    e = np.exp(z - z.max())
    return e / e.sum()


def pi_j_wald_interval(pi_hat, var_pi, z):
    """One-at-a-time Wald interval pi_j +/- z sqrt(Var), eq (3.8)."""
    if var_pi < 0 or z <= 0:
        raise ValueError("invalid inputs")
    half = z * math.sqrt(var_pi)
    return {"lower": pi_hat - half, "upper": pi_hat + half}


def proportional_odds_logit(bj0, bs, xs):
    """logit(P(Y <= j)) = bj0 + b1 x1 + ... + bp xp, eq (3.11)."""
    bs, xs = _v(bs), _v(xs)
    if bs.shape != xs.shape:
        raise ValueError("shape mismatch")
    return float(bj0 + np.dot(bs, xs))


def category_prob_from_cumulative(cum_probs, j):
    """pi_j = P(Y <= j) - P(Y <= j-1), eq (3.12); cum_probs excludes P(Y<=J)=1."""
    cp = np.concatenate([[0.0], _v(cum_probs), [1.0]])
    if (np.diff(cp) < -1e-12).any():
        raise ValueError("cumulative probabilities must be nondecreasing")
    if not 1 <= j <= cp.size - 1:
        raise ValueError("j out of range")
    return float(cp[j] - cp[j - 1])


def polr_parameterization(bj0, etas, xs):
    """polr() form logit(P(Y <= j)) = bj0 - eta1 x1 - ... - etap xp, eq (3.13)."""
    etas, xs = _v(etas), _v(xs)
    if etas.shape != xs.shape:
        raise ValueError("shape mismatch")
    return float(bj0 - np.dot(etas, xs))


def nonproportional_odds_logit(bj0, bjs, xs):
    """Non-proportional odds logit(P(Y <= j)) = bj0 + bj1 x1 + ..., eq (3.16)."""
    return baseline_logit(bj0, bjs, xs)


# ---------------------------------------------------------- ch 4: counts


def poisson_score_interval(mu_hat, n, z):
    """Score interval mu_hat + z^2/2n +/- z sqrt((mu_hat + z^2/4n)/n), eq (4.1)."""
    if mu_hat < 0 or n <= 0 or z <= 0:
        raise ValueError("invalid inputs")
    centre = mu_hat + z * z / (2.0 * n)
    half = z * math.sqrt((mu_hat + z * z / (4.0 * n)) / n)
    return {"lower": centre - half, "upper": centre + half}


def poisson_log_link(b0, bs, xs):
    """log(mu) = b0 + b1 x1 + ... + bp xp -> mu, eq (4.2)."""
    bs, xs = _v(bs), _v(xs)
    if bs.shape != xs.shape:
        raise ValueError("shape mismatch")
    return math.exp(b0 + float(np.dot(bs, xs)))


def poisson_loglik(b, x, y):
    """Poisson log-likelihood sum(-mu + y xb - log y!), eq (4.3)."""
    x = np.asarray(x, float)
    y = _v(y)
    b = _v(b)
    if x.ndim == 1:
        x = x[:, None]
    xb = x @ b
    return float(np.sum(-np.exp(xb) + y * xb
                        - np.array([math.lgamma(v + 1) for v in y])))


def loglinear_independence_mean(b0, beta_x_i, beta_z_j):
    """log(mu_ij) = b0 + bX_i + bZ_j -> mu_ij, eqs (4.4)-(4.5)."""
    return math.exp(b0 + beta_x_i + beta_z_j)


def loglinear_saturated_mean(b0, beta_x_i, beta_z_j, beta_xz_ij):
    """log(mu_ij) = b0 + bX_i + bZ_j + bXZ_ij -> mu_ij, eq (4.6)."""
    return math.exp(b0 + beta_x_i + beta_z_j + beta_xz_ij)


def loglinear_odds_ratio(bxz_ij, bxz_ipjp, bxz_ipj, bxz_ijp):
    """OR_{ii',jj'} = exp(bXZ_ij + bXZ_i'j' - bXZ_i'j - bXZ_ij'), eq (4.7)."""
    return math.exp(bxz_ij + bxz_ipjp - bxz_ipj - bxz_ijp)


def ordinal_score_mean_ratio(beta_z_j, beta_z_jp, beta_xz_i, s_j, s_jp):
    """mu_ij/mu_ij' = exp((bZ_j - bZ_j') + bXZ_i (s_j - s_j')), eqs (4.11)-(4.12)."""
    return math.exp((beta_z_j - beta_z_jp) + beta_xz_i * (s_j - s_jp))


def poisson_rate_mean(b0, bs, xs, exposure):
    """log(mu) = log(t) + b0 + sum b x -> mu = t exp(...), eq (4.15)."""
    if exposure <= 0:
        raise ValueError("exposure must be positive")
    return exposure * poisson_log_link(b0, bs, xs)


# ------------------------------------------------- ch 5: model selection


def bic_posterior_probs(bics):
    """tau_m = exp(-Delta_m/2) / sum exp(-Delta_a/2), eq (5.2)."""
    b = _v(bics)
    d = b - b.min()
    e = np.exp(-d / 2.0)
    return e / e.sum()


def model_averaged_estimate(taus, thetas):
    """theta_MA = sum tau_m theta_m, eq (5.3)."""
    t, th = _v(taus), _v(thetas)
    if t.shape != th.shape or abs(t.sum() - 1.0) > 1e-8 or (t < 0).any():
        raise ValueError("taus must be a distribution")
    return float(np.dot(t, th))


def model_averaged_variance(taus, thetas, variances):
    """Var(theta_MA) = sum tau [(theta_m - theta_MA)^2 + Var(theta_m)], eq (5.4)."""
    t, th, v = _v(taus), _v(thetas), _v(variances)
    if not (t.shape == th.shape == v.shape) or (v < 0).any():
        raise ValueError("invalid inputs")
    ma = model_averaged_estimate(t, th)
    return float(np.dot(t, (th - ma) ** 2 + v))


# ------------------------------------------------- ch 6: additional topics


def prevalence_from_apparent(pi, se, sp):
    """pi_tilde = (pi + Sp - 1)/(Se + Sp - 1), eqs (6.1), (6.3)."""
    if not 0 <= pi <= 1 or not 0 < se <= 1 or not 0 < sp <= 1 \
            or se + sp <= 1:
        raise ValueError("need Se + Sp > 1 and probabilities in range")
    return float((pi + sp - 1.0) / (se + sp - 1.0))


def misclassified_binomial_loglik(pi_tilde, se, sp, w, n):
    """log-likelihood of eq (6.2): binomial in pi = Se pt + (1-Sp)(1-pt)."""
    p = se * pi_tilde + (1 - sp) * (1 - pi_tilde)
    if not 0 < p < 1 or not 0 <= w <= n:
        raise ValueError("invalid inputs")
    return float(w * math.log(p) + (n - w) * math.log(1 - p))


def logistic_joint_probability(b, x, y):
    """Joint P(Y1..Yn) = exp(sum y xb)/prod(1+exp(xb)), eq (6.4) (as log)."""
    return logistic_loglik(b, x, y)


def exact_conditional_pmf(t_values, counts, beta, t_obs):
    """Exact conditional PMF P(T = t_u | I) = c(t_u) e^{b t_u}/sum, eqs (6.5)-(6.6)."""
    ts, cs = _v(t_values), _v(counts)
    if ts.shape != cs.shape or (cs <= 0).any():
        raise ValueError("invalid inputs")
    ln = beta * ts + np.log(cs)
    ln -= ln.max()
    probs = np.exp(ln)
    probs /= probs.sum()
    idx = np.where(np.isclose(ts, t_obs))[0]
    if idx.size == 0:
        raise ValueError("t_obs not among t_values")
    return {"probs": probs, "p_at_t": float(probs[idx[0]])}


def weighted_category_total(weights, ys, category):
    """N_hat_i = sum w_s I(y_s = i), eq (6.7)."""
    w = _v(weights)
    ys = np.asarray(ys)
    if w.size != ys.size or (w < 0).any():
        raise ValueError("invalid inputs")
    return float(w[ys == category].sum())


def jackknife_variance(replicate_estimates, full_estimate):
    """Var_hat = ((R-1)/R) sum (est_r - est)^2, eqs (6.8), (6.10)."""
    r = _v(replicate_estimates)
    n = r.size
    if n < 2:
        raise ValueError("need >= 2 replicates")
    return float((n - 1) / n * np.sum((r - full_estimate) ** 2))


def survey_proportion_variance(var_ni, var_n, cov_ni_n, pi_hat, n_hat):
    """Var(pi_hat_i) = (Var(N_i) + pi^2 Var(N) - 2 pi Cov)/N^2, eq (6.9)."""
    if n_hat <= 0:
        raise ValueError("N_hat must be positive")
    out = (var_ni + pi_hat ** 2 * var_n - 2 * pi_hat * cov_ni_n) / n_hat ** 2
    return float(out)


def kott_carr_interval(pi_hat, var_pi, t_crit):
    """Effective-n Wilson-type survey interval, eq (6.11)."""
    if not 0 < pi_hat < 1 or var_pi <= 0 or t_crit <= 0:
        raise ValueError("invalid inputs")
    n_eff = pi_hat * (1 - pi_hat) / var_pi
    t2 = t_crit * t_crit
    centre = 2 * n_eff * pi_hat + t2
    half = t_crit * math.sqrt(t2 + 4 * n_eff * pi_hat * (1 - pi_hat))
    den = 2 * (n_eff + t2)
    return {"n_effective": n_eff, "lower": (centre - half) / den,
            "upper": (centre + half) / den}


def spmi_loglinear_mean(b0, beta_w_a, beta_y_b):
    """SPMI independence model log(mu_ab(ij)) = b0 + bW_a + bY_b, eqs (6.14)-(6.15)."""
    return math.exp(b0 + beta_w_a + beta_y_b)


def three_mrcv_mean(b0, beta_w_a, beta_y_b, beta_z_c):
    """Complete independence for three MRCVs, eq (6.16)."""
    return math.exp(b0 + beta_w_a + beta_y_b + beta_z_c)


def glmm_linear_predictor(b0, b1, x, random_intercept):
    """GLMM linear predictors of eqs (6.17), (6.18), (6.20).

    b1 = 0, x = 0 gives the random-intercept-only model of eq (6.17).
    """
    return float(b0 + b1 * x + random_intercept)


def bayes_rule(p_a_given_b, p_b, p_a_given_notb):
    """P(B|A) = P(A|B)P(B)/(P(A|B)P(B) + P(A|~B)P(~B)), eq (6.21)."""
    for p in (p_a_given_b, p_b, p_a_given_notb):
        if not 0 <= p <= 1:
            raise ValueError("probabilities must be in [0, 1]")
    num = p_a_given_b * p_b
    den = num + p_a_given_notb * (1 - p_b)
    if den == 0:
        raise ValueError("zero marginal probability")
    return float(num / den)


def posterior_density_binomial(pi, w, n, a, b):
    """Beta(w+a, n+b-w) posterior density, eqs (6.22)-(6.23)."""
    if not 0 <= w <= n:
        raise ValueError("invalid w")
    return beta_pdf(pi, w + a, n - w + b)


def bayes_estimate_binomial(w, n, a, b):
    """pi_B = (n/(n+a+b)) pi_hat + ((a+b)/(n+a+b)) E(pi), eq (6.24)."""
    if not 0 <= w <= n or a <= 0 or b <= 0 or n <= 0:
        raise ValueError("invalid inputs")
    direct = (w + a) / (n + a + b)
    weighted = (n / (n + a + b)) * (w / n) \
        + ((a + b) / (n + a + b)) * (a / (a + b))
    if abs(direct - weighted) > 1e-12:
        raise AssertionError("decomposition identity violated")
    return float(direct)


def posterior_kernel_regression(logliks, log_priors):
    """Unnormalized log posterior sum(loglik) + sum(log prior), eq (6.25).

    Returns normalized weights over a parameter grid (the integral in the
    denominator is approximated by the grid sum).
    """
    ll, lp = _v(logliks), _v(log_priors)
    if ll.shape != lp.shape:
        raise ValueError("shape mismatch")
    ln = ll + lp
    ln -= ln.max()
    w = np.exp(ln)
    return w / w.sum()


def group_testing_expected_tests(i_size, se, sp, pi_tilde):
    """E(T_k) = 1 + I [Se + (1 - Se - Sp)(1 - pi_tilde)^I], eq (6.26)."""
    if i_size < 1 or not 0 <= pi_tilde <= 1 or not 0 < se <= 1 \
            or not 0 < sp <= 1:
        raise ValueError("invalid inputs")
    return float(1.0 + i_size * (se + (1 - se - sp)
                                 * (1 - pi_tilde) ** i_size))


def group_testing_logit(b0, bs, xs):
    """Group-testing regression logit(pi_tilde) = b0 + sum b x, eq (6.32)."""
    bs, xs = _v(bs), _v(xs)
    if bs.shape != xs.shape:
        raise ValueError("shape mismatch")
    return _expit(b0 + float(np.dot(bs, xs)))


def piecewise_cubic(x, knot, coef_left, coef_right):
    """Piecewise cubic f(x) of eq (6.34): two separate cubics around a knot."""
    c = _v(coef_left if x <= knot else coef_right)
    if c.size != 4:
        raise ValueError("each piece needs 4 coefficients")
    return float(c[0] + c[1] * x + c[2] * x ** 2 + c[3] * x ** 3)


def truncated_power_spline(x, betas, knots):
    """Cubic truncated power spline, eq (6.35): cubic + sum b_d (x-k_d)^3_+."""
    b = _v(betas)
    k = _v(knots)
    if b.size != 4 + k.size:
        raise ValueError("need 4 + D coefficients")
    out = b[0] + b[1] * x + b[2] * x ** 2 + b[3] * x ** 3
    for d, kd in enumerate(k):
        if x > kd:
            out += b[4 + d] * (x - kd) ** 3
    return float(out)


def spline_basis_eval(x, betas, basis_fns):
    """f(x) = sum beta_j h_j(x) over basis functions, eq (6.36)."""
    b = _v(betas)
    if b.size != len(basis_fns):
        raise ValueError("coefficients must match basis functions")
    return float(sum(bj * h(x) for bj, h in zip(b, basis_fns)))


def spline_odds_ratio(betas, basis_fns, a, b_pt):
    """OR between x = a and x = b: exp(sum beta_j (h_j(a) - h_j(b))), eq (6.37)."""
    fa = spline_basis_eval(a, betas, basis_fns)
    fb = spline_basis_eval(b_pt, betas, basis_fns)
    return math.exp(fa - fb)
