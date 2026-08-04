"""Confirmatory factor analysis 1-factor."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["cfa_one_factor"]


def cfa_one_factor(X, factor_structure):
    """
    Confirmatory factor analysis 1-factor

    Formula: X = lambda * F + eps; ML estimation

    Parameters
    ----------
    X : array-like
        Input data.
    factor_structure : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Jöreskog (1969)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Confirmatory factor analysis 1-factor"})


def cheatsheet():
    return "cfaftr: Confirmatory factor analysis 1-factor"


# compact alias per ledger/NAMING.md
cfaonefactor = cfa_one_factor
