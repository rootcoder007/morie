# morie.fn -- function file (rootcoder007/morie)
"""EM algorithm imputation for missing data."""

from __future__ import annotations

import math

from ._containers import DescriptiveResult


def em_imputation(
    data,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> DescriptiveResult:
    """EM imputation under a multivariate normal model.

    The mean and covariance are the maximum-likelihood estimates from the
    EM algorithm for incomplete multivariate normal data (Little and Rubin
    2002, sec. 11.2): the E-step fills each row's missing coordinates with
    their conditional mean given the observed ones AND carries their
    conditional covariance, which the M-step adds back,

        Sigma = (1/n) sum_i [ (x_i - mu)(x_i - mu)' + C_i ].

    Dropping ``C_i`` -- iterating regression imputation, which is what this
    function used to do -- shrinks the covariance and is not EM. The EM
    iterations are those of :func:`morie.fn.eslmem.esl_mvn_em_missing`; the
    returned data are the conditional means at the final estimates.

    Parameters
    ----------
    data : (n, p) array with NaN (or None) for missing values
    max_iter, tol : EM controls (largest change in mu or Sigma)

    Returns
    -------
    DescriptiveResult
        ``value`` = number of missing cells; ``extra``: ``n_missing``,
        ``n``, ``p``, ``imputed_means``, ``mean``, ``cov``, ``imputed``
        (the completed data), ``loglik`` (observed-data), ``iterations``,
        ``converged``.

    References
    ----------
    Dempster, A. P., Laird, N. M. and Rubin, D. B. (1977). Maximum
    likelihood from incomplete data via the EM algorithm. JRSS B 39, 1-38.
    Little, R. J. A. and Rubin, D. B. (2002). Statistical Analysis with
    Missing Data, 2nd ed. Wiley, sec. 11.2.

    Examples
    --------
    >>> r = em_imputation([[1.0, 2.1], [2.0, 2.9], [3.5, 4.2], [4.0, float("nan")]], tol=1e-12, max_iter=1000)
    >>> [round(v, 10) for v in r.extra["mean"]]
    [2.625, 3.4526315789]
    """
    from .eslmem import esl_mvn_em_missing
    from .nlsgn import _inverse

    raw = data.tolist() if hasattr(data, "tolist") else [list(r) if isinstance(r, (list, tuple)) else r for r in data]
    if raw and not isinstance(raw[0], (list, tuple)):
        raw = [[v] for v in raw]

    def miss(v):
        return v is None or (isinstance(v, float) and math.isnan(v))

    rows = [[None if miss(v) else float(v) for v in r] for r in raw]
    n, p = len(rows), len(rows[0])
    n_missing = sum(1 for r in rows for v in r if v is None)
    fit = esl_mvn_em_missing(rows, max_iter=max_iter, tol=tol)
    mu, S = fit["mean"], fit["cov"]
    imputed = []
    for r in rows:
        o = [j for j in range(p) if r[j] is not None]
        m = [j for j in range(p) if r[j] is None]
        x = [r[j] if r[j] is not None else mu[j] for j in range(p)]
        if m and o:
            Soo = _inverse([[S[a][b] for b in o] for a in o])
            dev = [r[j] - mu[j] for j in o]
            for a in m:
                reg = [sum(S[a][o[k]] * Soo[k][q] for k in range(len(o))) for q in range(len(o))]
                x[a] = mu[a] + sum(g * d for g, d in zip(reg, dev))
        imputed.append(x)
    means = []
    for j in range(p):
        s = 0.0
        for x in imputed:
            s += x[j]
        means.append(s / n)
    return DescriptiveResult(
        name="em_imputation",
        value=float(n_missing),
        extra={
            "n_missing": n_missing,
            "n": n,
            "p": p,
            "imputed_means": means,
            "mean": list(mu),
            "cov": [list(r) for r in S],
            "imputed": imputed,
            "loglik": fit["loglik"],
            "iterations": fit["iterations"],
            "converged": fit["converged"],
        },
    )


emimq = em_imputation


def cheatsheet() -> str:
    return "em_imputation({}) -> EM (with conditional covariance) MVN estimates and imputation."


# compact alias per ledger/NAMING.md
emimputation = em_imputation
