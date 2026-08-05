# morie.fn -- function file (rootcoder007/morie)
"""Sex-stratified Mendelian randomisation."""

import math

from . import _s03core as core

from ._richresult import RichResult

__all__ = ["sex_specific_mr"]


def _ratio(g, x, y):
    n = len(g)
    gm = sum(g) / n
    xm = sum(x) / n
    ym = sum(y) / n
    sgg = sum((g[i] - gm) ** 2 for i in range(n))
    if sgg <= 0.0:
        raise ValueError("the instrument does not vary in one stratum")
    gx = sum((g[i] - gm) * (x[i] - xm) for i in range(n)) / sgg
    gy = sum((g[i] - gm) * (y[i] - ym) for i in range(n)) / sgg
    if abs(gx) < 1e-12:
        raise ValueError("the instrument does not predict the exposure in "
                         "one stratum")
    ay = ym - gy * gm
    resid = [y[i] - ay - gy * g[i] for i in range(n)]
    s2 = sum(t * t for t in resid) / (n - 2) if n > 2 else float("nan")
    se_gy = math.sqrt(s2 / sgg) if s2 == s2 else float("nan")
    beta = gy / gx
    return beta, abs(se_gy / gx), gx, gy, n


def sex_specific_mr(y, exposure, instrument, sex):
    """Run the instrumental-variable estimate separately in each sex.

    A genetic instrument can act on the outcome through different biology
    in men and in women, and pooling then estimates a weighted average of
    two different causal effects -- a quantity that answers no question.
    Stratifying costs power but keeps the estimand interpretable, and the
    difference between the strata is itself the interesting result: a
    large one is evidence of effect modification, not of a bad instrument.

    Formula: within each stratum the Wald ratio ``beta = Gamma/gamma``
    with ``Gamma`` the instrument-outcome slope and ``gamma`` the
    instrument-exposure slope; ``se(beta) = se(Gamma)/|gamma|`` to first
    order.  The strata are compared by ``z = (b1 - b2)/sqrt(se1^2 +
    se2^2)`` -- Burgess, Small & Thompson (2017) Sections 2 and 5.

    Parameters
    ----------
    y : array-like
        Outcome.
    exposure : array-like
        Exposure the instrument is meant to move.
    instrument : array-like
        Genetic instrument (e.g. an allele count).
    sex : array-like
        Stratum label per observation; exactly two distinct values.

    Returns
    -------
    RichResult
        ``estimate`` (the inverse-variance pool of the two strata),
        ``se``, ``strata``, ``beta_by_stratum``, ``se_by_stratum``,
        ``n_by_stratum``, ``z_het``, ``p_het``.

    References
    ----------
    Burgess, S., Small, D. S. and Thompson, S. G. (2017).  A review of
    instrumental variable estimators for Mendelian randomization.
    Statistical Methods in Medical Research 26(5):2333-2355.
    doi:10.1177/0962280215597579.
    """
    Y = [float(t) for t in core.vec(y)]
    X = [float(t) for t in core.vec(exposure)]
    G = [float(t) for t in core.vec(instrument)]
    S = [t for t in core.vec(sex)]
    n = len(Y)
    if n == 0:
        raise ValueError("no observations")
    if not (len(X) == len(G) == len(S) == n):
        raise ValueError("all inputs must have the same length")
    labels = sorted({float(t) for t in S})
    if len(labels) != 2:
        raise ValueError("sex must take exactly two distinct values")
    betas = []
    ses = []
    ns = []
    for lab in labels:
        idx = [i for i in range(n) if float(S[i]) == lab]
        if len(idx) < 3:
            raise ValueError("each stratum needs at least three observations")
        b, s, _, _, k = _ratio([G[i] for i in idx], [X[i] for i in idx],
                               [Y[i] for i in idx])
        betas.append(b)
        ses.append(s)
        ns.append(k)
    w = [1.0 / (t * t) for t in ses]
    pooled = sum(w[j] * betas[j] for j in range(2)) / sum(w)
    se_p = math.sqrt(1.0 / sum(w))
    sd = math.sqrt(ses[0] ** 2 + ses[1] ** 2)
    z = (betas[0] - betas[1]) / sd
    return RichResult(payload={
        "estimate": pooled, "se": se_p, "strata": labels,
        "beta_by_stratum": betas, "se_by_stratum": ses,
        "n_by_stratum": ns, "z_het": z,
        "p_het": 2.0 * (1.0 - core.pnorm(abs(z))),
        "method": "Sex-stratified Mendelian randomisation"})


def cheatsheet():
    return "mtr2sx: sex-stratified Mendelian randomisation with a heterogeneity test"


# compact alias per ledger/NAMING.md
sexspecificmr = sex_specific_mr
