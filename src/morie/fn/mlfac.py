# morie.fn -- function file (rootcoder007/morie)
"""Maximum Likelihood Factor Analysis.

Fits the factor model S = Lambda Lambda' + Psi by maximum likelihood (the objective of
stats::factanal) and returns loadings, uniquenesses and fit statistics.

References
----------
Joreskog, K. G. (1967). Some contributions to maximum likelihood factor analysis.
    Psychometrika, 32(4), 443-482.
Lawley, D. N. & Maxwell, A. E. (1971). Factor Analysis as a Statistical Method, 2nd ed.
"""

import math

from ._mlfa import corr_matrix, eigh_desc, fa_statistic, mlfa_fit, to_corr


def _ssum(it):
    # plain left-to-right summation: sum() of floats is compensated from Python 3.12 on, which
    # would make results depend on the Python version and differ from the R arm
    s = 0.0
    for v in it:
        s += v
    return s


__all__ = ["mlfac"]


def mlfac(X, n_factors=None, max_iter=500, tol=1e-12, scale=True):
    r"""Maximum Likelihood Factor Analysis.

    Minimises F(Psi) = sum_{j>m} (theta_j - log theta_j) - (p - m), theta the
    eigenvalues of Psi^{-1/2} S Psi^{-1/2}, over uniquenesses in [0.005, 1] by
    projected BFGS with the analytic gradient (the criterion minimised by ``factanal``); the loadings
    are Psi^{1/2} U diag(sqrt(theta - 1)) (unrotated). The likelihood-ratio
    statistic is Bartlett's (n - 1 - (2p + 5)/6 - 2m/3) F on
    ((p - m)^2 - p - m)/2 degrees of freedom; AIC = chi2 - 2 df and
    BIC = chi2 - df log n are on that scale.

    Parameters
    ----------
    X : n x p data
    n_factors : int, optional
        Number of factors (default: Kaiser count of correlation eigenvalues > 1).
    max_iter : int
        BFGS iterations.
    tol : float
        Gradient tolerance.
    scale : bool
        Analyse the correlation matrix (True) or the covariance matrix.

    Returns
    -------
    dict
        loadings (p x m), communalities, uniqueness (standardised), variance_explained,
        objective, log_likelihood, statistic, dof, p_value, aic, bic.

    Examples
    --------
    >>> X = [[1, 2, 1, 3], [2, 3, 2, 4], [3, 3, 4, 4], [4, 5, 4, 6], [5, 5, 6, 5], [6, 7, 5, 8], [7, 8, 7, 8], [8, 8, 9, 9]]
    >>> r = mlfac(X, n_factors=1)
    >>> all(0 < u <= 1 for u in r["uniqueness"])
    True
    """
    from ._rrng_core import pchisq

    if hasattr(X, "to_numpy"):
        X = X.to_numpy()
    if hasattr(X, "tolist"):
        X = X.tolist()
    X = [[float(v) for v in row] for row in X]
    n, p = len(X), len(X[0])
    C = corr_matrix(X)
    R = to_corr(C)
    if n_factors is None:
        ev, _ = eigh_desc(R)
        n_factors = max(1, sum(v > 1 for v in ev))
    m = int(n_factors)
    fit = mlfa_fit(R, m, max_iter=max_iter, gtol=tol)
    L = fit["loadings"]
    if not scale:
        sd = [math.sqrt(C[j][j]) for j in range(p)]
        L = [[L[j][k] * sd[j] for k in range(m)] for j in range(p)]
    h2 = [_ssum(v * v for v in row) for row in fit["loadings"]]
    stat, dof = fa_statistic(fit["objective"], n, p, m)
    # at the MLE, log|Sigma| + tr(Sigma^{-1} S) = F + log|S| + p
    loglik = -n / 2 * (p * math.log(2 * math.pi) + fit["objective"] + _ssum(math.log(e) for e in eigh_desc(R)[0]) + p)
    return {
        "loadings": L,
        "communalities": h2,
        "uniqueness": fit["uniquenesses"],
        "variance_explained": [_ssum(L[j][k] ** 2 for j in range(p)) for k in range(m)],
        "objective": fit["objective"],
        "log_likelihood": loglik,
        "statistic": stat,
        "dof": dof,
        "p_value": pchisq(stat, dof, lower_tail=False) if dof > 0 else float("nan"),
        "aic": stat - 2 * dof,
        "bic": stat - dof * math.log(n),
        "n_factors": m,
    }


def cheatsheet():
    return "mlfac: maximum likelihood factor analysis (factanal objective), loadings, uniquenesses, chi2, AIC, BIC."
