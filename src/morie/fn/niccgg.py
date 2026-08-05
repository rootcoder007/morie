# morie.fn -- function file (rootcoder007/morie)
"""Nakagawa-Schielzeth marginal R^2 for a random-intercept LMM."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["nakagawa_marginal_r2"]


def nakagawa_marginal_r2(y, X, Z=None, cluster=None):
    """Marginal (fixed-effects-only) R-squared of a random-intercept LMM.

    Formula (Nakagawa & Schielzeth 2013, eq. 26):

        R2_marginal = sigma2_f / (sigma2_f + sigma2_l + sigma2_e)

    where ``sigma2_f`` is the variance of the fixed-effect fitted values
    ``X beta``, ``sigma2_l`` the between-cluster (random-intercept)
    variance and ``sigma2_e`` the residual variance.  The conditional
    R-squared, which the same paper defines, adds ``sigma2_l`` to the
    numerator and is returned alongside.

    Determinism: the variance components are the one-way random-effects
    ANOVA moment estimators computed on the OLS residuals, not an
    iterative REML fit, so both language arms land on the same numbers
    exactly.  ``sigma2_l`` is truncated at zero, as a variance must be.

        sigma2_e = MSW,
        sigma2_l = max(0, (MSB - MSW) / n0),
        n0       = (N - sum_i n_i^2 / N) / (k - 1).

    Parameters
    ----------
    y : array-like, shape (N,)
        Response.
    X : array-like, shape (N, p)
        Fixed-effects design WITHOUT an intercept column; one is added.
    cluster : array-like, shape (N,)
        Grouping label per observation; any hashable values.
    Z : ignored
        Accepted for signature compatibility; a random-intercept model
        has ``Z`` equal to the cluster indicator, which ``cluster``
        already supplies.

    Returns
    -------
    RichResult
        ``estimate`` (marginal R-squared), ``r2_marginal``,
        ``r2_conditional``, ``sigma2_f``, ``sigma2_l``, ``sigma2_e``,
        ``icc``, ``n``, ``n_clusters``.

    References
    ----------
    Nakagawa, S. & Schielzeth, H. (2013).  A general and simple method
    for obtaining R^2 from generalized linear mixed-effects models.
    Methods in Ecology and Evolution, 4(2), 133--142.
    doi:10.1111/j.2041-210x.2012.00261.x
    """
    yv = C.vec(y)
    N = len(yv)
    if N == 0:
        raise ValueError("nakagawa_marginal_r2: y is empty")
    Xm = C.cbind1(C.mat(X))
    if len(Xm) != N:
        raise ValueError("nakagawa_marginal_r2: X and y have different lengths")
    if cluster is None:
        raise ValueError("nakagawa_marginal_r2: cluster labels are required")
    lab = list(cluster)
    if len(lab) != N:
        raise ValueError("nakagawa_marginal_r2: cluster and y have different lengths")
    _, fitted, resid, _ = C.lstsq(Xm, yv)
    s2f = C.var(fitted, ddof=1)

    groups = []
    for g in lab:
        if g not in groups:
            groups.append(g)
    k = len(groups)
    if k < 2:
        raise ValueError("nakagawa_marginal_r2: need at least two clusters")
    if k >= N:
        raise ValueError("nakagawa_marginal_r2: need more observations than clusters")
    idx = {g: j for j, g in enumerate(groups)}
    ni = [0] * k
    si = [0.0] * k
    for i in range(N):
        j = idx[lab[i]]
        ni[j] += 1
        si[j] += resid[i]
    gm = sum(resid) / N
    msb = sum(ni[j] * (si[j] / ni[j] - gm) ** 2 for j in range(k)) / (k - 1)
    msw = 0.0
    for i in range(N):
        j = idx[lab[i]]
        msw += (resid[i] - si[j] / ni[j]) ** 2
    msw /= (N - k)
    n0 = (N - sum(v * v for v in ni) / N) / (k - 1)
    s2l = (msb - msw) / n0
    if s2l < 0.0:
        s2l = 0.0
    s2e = msw
    tot = s2f + s2l + s2e
    if tot <= 0.0:
        raise ValueError("nakagawa_marginal_r2: total variance is zero")
    return RichResult(payload={
        "estimate": s2f / tot, "r2_marginal": s2f / tot,
        "r2_conditional": (s2f + s2l) / tot,
        "sigma2_f": s2f, "sigma2_l": s2l, "sigma2_e": s2e,
        "icc": s2l / (s2l + s2e) if (s2l + s2e) > 0.0 else float("nan"),
        "n": N, "n_clusters": k,
        "method": "Nakagawa-Schielzeth marginal R^2 (random intercept)"})


def cheatsheet():
    return "niccgg: Nakagawa-Schielzeth marginal R^2 for an LMM"


nakagawamarginalr2 = nakagawa_marginal_r2
