# morie.fn -- function file (rootcoder007/morie)
"""Gaussian copula fitted to spatial/multivariate data."""

from . import _array_core as np
from . import _stats_core as stats

from ._copula import copula_tau, tau_to_theta
from ._richresult import RichResult

__all__ = ["copula_gauss_sp"]


def copula_gauss_sp(data):
    r"""Fit a gaussian copula to multivariate data by inversion of Kendall's tau.

    Ranks each column to pseudo-observations, then fits the gaussian
    dependence structure by matching Kendall's tau pairwise (Czado
    2019 Table 3.2, p. 54).
    
    
    The Gaussian copula is elliptical, so the pairwise map rho = sin(pi tau / 2) yields a full correlation matrix, projected to the nearest positive-definite matrix if the pairwise estimates are inconsistent.

    Parameters
    ----------
    data : array-like, shape (n, d)
        Observations, one column per variable.

    Returns
    -------
    RichResult
        keys: ``tau_matrix`` (d, d), ``pseudo_obs`` (n, d), ``correlation``, ``positive_definite``, ``n``, ``d``, ``method``.

    References
    ----------
    Czado, C. (2019). *Analyzing Dependent Data with Vine Copulas*.
    Springer. Ch. 3 (bivariate families and tau), Table 3.2 p. 54.
    """
    X = np.asarray(data, dtype=float)
    if X.ndim != 2:
        raise ValueError("data must be 2-D (n observations x d variables).")
    n, d = X.shape
    if d < 2:
        raise ValueError("need at least 2 variables.")
    if n < 5:
        raise ValueError(f"need at least 5 observations, got {n}.")
    if not np.all(np.isfinite(X)):
        raise ValueError("data must be finite.")

    U = np.column_stack([stats.rankdata(X[:, j]) / (n + 1) for j in range(d)])
    tau = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            t = float(stats.kendalltau(X[:, i], X[:, j]).statistic)
            tau[i, j] = tau[j, i] = 0.0 if not np.isfinite(t) else t

    R = np.sin(np.pi * tau / 2.0)
    np.fill_diagonal(R, 1.0)
    vals = np.linalg.eigvalsh(R)
    pd = bool(vals.min() > 0)
    if not pd:  # nearest PD by eigenvalue clipping, then rescale to unit diagonal
        w, V = np.linalg.eigh(R)
        R = V @ np.diag(np.maximum(w, 1e-8)) @ V.T
        dg = np.sqrt(np.diag(R))
        R = R / np.outer(dg, dg)

    return RichResult(
        payload={
            "tau_matrix": tau,
            "pseudo_obs": U,
            "correlation": R,
            "positive_definite": pd,
            "n": int(n),
            "d": int(d),
            "method": "Gaussian copula by pairwise tau inversion rho = sin(pi tau / 2)",
        }
    )


def cheatsheet():
    return "zxcpg: gaussian copula fitted by pairwise Kendall-tau inversion"


# compact alias per ledger/NAMING.md
copulagausssp = copula_gauss_sp
