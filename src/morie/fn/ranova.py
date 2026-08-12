"""ANOVA estimation of variance components (Searle, Casella & McCulloch 1992)."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["ranova", "anova_variance_components"]


def _groups(y, group):
    g = {}
    for v, k in zip(y, group):
        g.setdefault(k, []).append(float(v))
    keys = sorted(g, key=lambda k: (str(type(k)), k))
    return keys, [g[k] for k in keys]


def ranova(y, group):
    r"""
    ANOVA (method-of-moments) estimation of variance components.

    One-way random model y_ij = mu + a_i + e_ij with a_i ~ (0,
    sigma_a^2) and e_ij ~ (0, sigma_e^2), independent.  Forming the
    usual sums of squares (Searle et al. Sec. 2.2, Eqs. 18-19)

        SSA = sum_i n_i (ybar_i. - ybar..)^2,
        SSE = sum_ij (y_ij - ybar_i.)^2,
        MSA = SSA / (a - 1),   MSE = SSE / (N - a),

    the ANOVA estimators equate mean squares to their expectations.
    For BALANCED data (a classes of n each) this is their Eq. (21):

        sigma_e^2 = SSE / [a(n - 1)] = MSE,
        sigma_a^2 = (MSA - MSE) / n.

    For UNBALANCED data (their Ch. 5, Henderson's Method I applied to
    the one-way classification) E[MSA] = sigma_e^2 + n_0 sigma_a^2
    with the standard coefficient

        n_0 = ( N - sum_i n_i^2 / N ) / (a - 1),

    so sigma_a^2 = (MSA - MSE) / n_0, which reduces to (MSA - MSE)/n
    when all n_i = n.  These are unbiased but can go negative; the
    raw solution and the truncated-at-zero estimate are both
    returned, since Searle et al. treat non-negativity as a separate
    step (their Sec. 3.3, and Tables 4.9-4.15).

    Sources
    -------
    Searle, S. R., Casella, G. & McCulloch, C. E. (1992). *Variance
    Components*. Wiley.  One-way ANOVA estimators Eq. (21), Sec. 2.2
    (pp. 26-27); balanced-data ANOVA estimation Ch. 4; unbalanced
    data and Henderson's Method I Ch. 5 (local copy fetched-wave3/
    Variance_components_FULL.pdf).

    Parameters
    ----------
    y : sequence of float
        Observations.
    group : sequence
        Class label per observation.

    Returns
    -------
    RichResult
        Keys: sigma2_a, sigma2_e, sigma2_a_raw (may be negative),
        msa, mse, ssa, sse, n0, a (number of classes), N, balanced,
        icc (sigma2_a / (sigma2_a + sigma2_e)).
    """
    y = [float(v) for v in y]
    if len(y) != len(group):
        raise ValueError("y and group must have equal length")
    keys, gs = _groups(y, group)
    a = len(keys)
    if a < 2:
        raise ValueError("need at least two classes")
    ns = [len(g) for g in gs]
    N = sum(ns)
    if min(ns) < 1:
        raise ValueError("every class needs an observation")
    grand = sum(y) / N
    means = [sum(g) / len(g) for g in gs]
    ssa = sum(ns[i] * (means[i] - grand) ** 2 for i in range(a))
    sse = sum((v - means[i]) ** 2 for i in range(a) for v in gs[i])
    if N == a:
        raise ValueError("need replication within classes")
    msa = ssa / (a - 1)
    mse = sse / (N - a)
    balanced = len(set(ns)) == 1
    # n_0 coefficient; equals n exactly when balanced
    n0 = (N - sum(n * n for n in ns) / N) / (a - 1)
    s2a_raw = (msa - mse) / n0
    s2a = s2a_raw if s2a_raw > 0.0 else 0.0
    s2e = mse
    denom = s2a + s2e
    return RichResult(payload={
        "sigma2_a": s2a,
        "sigma2_e": s2e,
        "sigma2_a_raw": s2a_raw,
        "msa": msa,
        "mse": mse,
        "ssa": ssa,
        "sse": sse,
        "n0": n0,
        "a": a,
        "N": N,
        "n_i": ns,
        "balanced": balanced,
        "icc": (s2a / denom) if denom > 0 else 0.0,
        "method": "ANOVA variance components (Searle et al. 1992, Eq. 21)",
    })


# long descriptive alias (stub-era name)
anova_variance_components = ranova


def cheatsheet():
    return ("ranova: sigma2_e = MSE; sigma2_a = (MSA - MSE)/n0, "
            "n0 = (N - sum n_i^2/N)/(a-1)")
