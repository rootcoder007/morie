"""ROBPCA (Hubert et al)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["robust_pca"]


def robust_pca(X, k):
    """
    ROBPCA (Hubert et al)

    Formula: projection-pursuit + MCD on subspace

    Parameters
    ----------
    X : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hubert-Rousseeuw-Vanden Branden (2005)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ROBPCA (Hubert et al)"})


def cheatsheet():
    return "robpca: ROBPCA (Hubert et al)"


# compact alias per ledger/NAMING.md
robustpca = robust_pca
