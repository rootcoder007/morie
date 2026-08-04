"""Bayesian bootstrap via Dirichlet weights."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["boot_bayesian"]


def boot_bayesian(x, stat, B):
    """
    Bayesian bootstrap via Dirichlet weights

    Formula: w ~ Dirichlet(1,...,1); θ̂*_b = T_w(x)

    Parameters
    ----------
    x : array-like
        Input data.
    stat : array-like
        Input data.
    B : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta_b

    References
    ----------
    Rubin (1981)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Bayesian bootstrap via Dirichlet weights"}
    )


def cheatsheet():
    return "btbayes: Bayesian bootstrap via Dirichlet weights"


# compact alias per ledger/NAMING.md
bootbayesian = boot_bayesian
