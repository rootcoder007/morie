# morie.fn -- function file (rootcoder007/morie)
"""Classical Glivenko-Cantelli."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kosorok_ch2_glivenko_cantelli_classical"]


def kosorok_ch2_glivenko_cantelli_classical(X, F=None, n_grid=None):
    r"""Classical Glivenko-Cantelli:

    .. math:: \sup_{t \in \mathbb R} |F_n(t) - F(t)| \to 0
              \quad \text{almost surely.}

    Returns the sup distance along growing subsamples. The supremum is
    computed EXACTLY at the order statistics, on both sides of each
    jump -- evaluating on a fixed grid would understate it and make
    the convergence look better than it is.

    Parameters
    ----------
    X : array-like
        Sample.
    F : callable, optional
        True CDF; uniform on [0, 1] if omitted.
    n_grid : sequence of int, optional
        Subsample sizes.

    Returns
    -------
    RichResult
        keys: ``n_grid``, ``sup_distance``, ``dkw_bound``, ``n``,
        ``method``.
    References
    ----------
    Kosorok, M. R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Ch. 2 (Glivenko-Cantelli).
    """
    X = np.asarray(X, dtype=float).ravel()
    N = X.size
    if N < 8:
        raise ValueError(f"need at least 8 observations, got {N}.")
    if n_grid is None:
        n_grid = [max(4, int(N * f)) for f in (0.1, 0.25, 0.5, 1.0)]
    n_grid = [int(g) for g in n_grid]

    def sup_at(sub):
        xs = np.sort(sub)
        m = xs.size
        Ft = np.clip(xs, 0, 1) if F is None else np.array([F(v) for v in xs])
        return float(max((np.arange(1, m + 1) / m - Ft).max(),
                         (Ft - np.arange(0, m) / m).max()))

    sup = np.array([sup_at(X[:g]) for g in n_grid])
    dkw = np.array([2 * np.exp(-2 * g * s**2) for g, s in zip(n_grid, sup)])
    return RichResult(
        payload={"n_grid": np.array(n_grid), "sup_distance": sup,
                 "dkw_bound": np.minimum(dkw, 1.0), "n": int(N),
                 "method": "sup_t |F_n - F| at the order statistics (Kosorok Ch. 2)"}
    )


def cheatsheet():
    return "ksr028: uniform GC; sup taken exactly at the jumps"
