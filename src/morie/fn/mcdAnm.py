# morie.fn -- function file (rootcoder007/morie)
"""Minimum covariance determinant outlier detection."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["mcd_outlier"]


def mcd_outlier(X, support_fraction=None, n_trials=50, alpha=0.025, seed=0):
    r"""Robust Mahalanobis distances from a minimum-covariance-determinant fit.

    Finds the subset of :math:`h` points whose covariance has the smallest
    determinant, and measures distance from *that* centre and scatter:

    .. math::
        d_i^2 = (x_i - \hat\mu_{\text{MCD}})^\top
                \hat\Sigma_{\text{MCD}}^{-1}
                (x_i - \hat\mu_{\text{MCD}}).

    The reason to bother is **masking**. Classical Mahalanobis distance uses
    the sample mean and covariance, which the outliers themselves inflate, so
    a cluster of outliers hides itself -- each one looks unremarkable against
    a covariance it helped create. MCD estimates location and scatter from a
    clean majority, so the outliers stand out. The doctest demonstrates
    exactly that failure and its repair.

    The breakdown point is :math:`(n-h)/n`, maximised at :math:`h \approx n/2`,
    at the cost of efficiency when the data is in fact clean. The default
    :math:`h = 0.75n` is the usual compromise. Cutoffs use the chi-squared
    quantile, which is exact only asymptotically and under normality.

    Parameters
    ----------
    X : array-like
        Data ``(n, p)``.
    support_fraction : float, optional
        Fraction of points in the clean subset, in (0.5, 1]. Default 0.75.
    n_trials : int
        Random starts for the C-step search.
    alpha : float
        Tail probability for the chi-squared cutoff.
    seed : int
        Seed.

    Returns
    -------
    RichResult
        ``distance`` (robust), ``classical_distance``, ``outlier``,
        ``location``, ``covariance``, ``cutoff``, ``n_outliers``.

    References
    ----------
    Rousseeuw, P. J., & Van Driessen, K. (1999). A fast algorithm for the
        minimum covariance determinant estimator. *Technometrics*, 41(3),
        212-223.

    Examples
    --------
    Masking: a cluster of outliers inflates the classical covariance until it
    hides itself, while the robust distance still finds it.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = np.r_[rng.normal(0, 1, (200, 2)), rng.normal(7, 0.3, (25, 2))]
    >>> r = mcd_outlier(X, seed=1)
    >>> bool(r["distance"][200:].min() > r["distance"][:200].max())
    True
    >>> bool(r["classical_distance"][200:].min() < r["classical_distance"][:200].max())
    True

    Nearly all of the contaminating cluster is flagged.

    >>> bool(r["outlier"][200:].mean() > 0.9)
    True

    The robust location is near the clean centre, not dragged toward the
    contamination.

    >>> bool(np.max(np.abs(r["location"])) < 1.0)
    True
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n, p = X.shape
    if n <= p:
        raise ValueError(f"need more observations than dimensions (n={n}, p={p})")
    frac = 0.75 if support_fraction is None else float(support_fraction)
    if not 0.5 < frac <= 1.0:
        raise ValueError("support_fraction must be in (0.5, 1]")
    h = max(int(np.ceil(frac * n)), p + 1)
    rng = np.random.default_rng(seed)

    best = (np.inf, None, None)
    for _ in range(int(n_trials)):
        idx = rng.choice(n, h, replace=False)
        for _ in range(20):                      # C-steps
            mu = X[idx].mean(axis=0)
            S = np.cov(X[idx], rowvar=False).reshape(p, p) + 1e-9 * np.eye(p)
            try:
                Si = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                break
            d = np.einsum("ij,jk,ik->i", X - mu, Si, X - mu)
            new = np.argsort(d, kind="stable")[:h]
            if np.array_equal(np.sort(new), np.sort(idx)):
                break
            idx = new
        det = float(np.linalg.det(np.cov(X[idx], rowvar=False).reshape(p, p)))
        if 0 <= det < best[0]:
            best = (det, X[idx].mean(axis=0),
                    np.cov(X[idx], rowvar=False).reshape(p, p) + 1e-9 * np.eye(p))
    _, mu, S = best
    if mu is None:
        mu, S = X.mean(axis=0), np.cov(X, rowvar=False).reshape(p, p)

    from ._stats_core import chi2

    Si = np.linalg.pinv(S)
    d2 = np.einsum("ij,jk,ik->i", X - mu, Si, X - mu)
    cmu = X.mean(axis=0)
    cS = np.cov(X, rowvar=False).reshape(p, p)
    cd2 = np.einsum("ij,jk,ik->i", X - cmu, np.linalg.pinv(cS), X - cmu)
    cut = float(chi2.ppf(1 - alpha, p))
    out = d2 > cut
    return RichResult(
        title="MCD outlier detection",
        summary_lines=[("n", n), ("h", h), ("outliers", int(out.sum())),
                       ("cutoff", cut)],
        warnings=["the chi-squared cutoff is exact only asymptotically and "
                  "under normality"],
        payload={
            "distance": np.sqrt(np.maximum(d2, 0)),
            "classical_distance": np.sqrt(np.maximum(cd2, 0)),
            "outlier": out, "location": mu, "covariance": S,
            "cutoff": float(np.sqrt(cut)), "n_outliers": int(out.sum()),
            "h": int(h), "support_fraction": frac, "method": "mcd_outlier",
        },
    )


def cheatsheet():
    return "mcdAnm: defeats MASKING -- classical covariance is inflated by the outliers so they hide themselves"
