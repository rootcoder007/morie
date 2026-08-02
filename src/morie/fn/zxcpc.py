# morie.fn -- function file (rootcoder007/morie)
"""Clayton copula fitted to spatial/multivariate data."""

from . import _array_core as np
from scipy import stats

from ._copula import copula_tau, tau_to_theta
from ._richresult import RichResult

__all__ = ["copula_clayton_sp"]


def copula_clayton_sp(data):
    r"""Fit a clayton copula to multivariate data by inversion of Kendall's tau.

    Ranks each column to pseudo-observations, then fits the clayton
    dependence structure by matching Kendall's tau pairwise (Czado
    2019 Table 3.2, p. 54).
    
    Pairwise thetas are reported rather than a single global parameter, because a one-parameter Archimedean cannot represent an arbitrary correlation pattern -- averaging them would hide that.
    

    Parameters
    ----------
    data : array-like, shape (n, d)
        Observations, one column per variable.

    Returns
    -------
    RichResult
        keys: ``tau_matrix`` (d, d), ``pseudo_obs`` (n, d), ``theta_matrix``, ``n``, ``d``, ``method``.

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

    theta = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            t = tau[i, j]
            # Clayton admits only positive dependence; NaN marks the pairs
            # it cannot represent rather than silently clamping them
            theta[i, j] = theta[j, i] = tau_to_theta("clayton", t) if t > 1e-6 else np.nan

    return RichResult(
        payload={
            "tau_matrix": tau,
            "pseudo_obs": U,
            "theta_matrix": theta,
            "n": int(n),
            "d": int(d),
            "method": "Clayton copula by pairwise tau inversion theta = 2 tau / (1 - tau)",
        }
    )


def cheatsheet():
    return "zxcpc: clayton copula fitted by pairwise Kendall-tau inversion"
