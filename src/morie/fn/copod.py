# morie.fn -- function file (rootcoder007/morie)
"""COPOD: copula-based outlier detection."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["copod"]


def copod(X, skew_correction=True):
    r"""Empirical-copula outlier scores.

    For each feature, COPOD builds the empirical CDF and its mirror
    and converts both to tail probabilities; the outlier score is the
    negative log-tail-probability summed over dimensions,

    .. math:: s_i = \max\Big(
              -\sum_d \log \hat F_d(x_{id}),\;
              -\sum_d \log \hat{\bar F}_d(x_{id}) \Big),

    with the skewness correction picking, per dimension, the tail the
    feature's skew implicates. Being rank-based it needs no distance
    metric and no scaling -- the property that makes it robust to
    heterogeneous units.

    Parameters
    ----------
    X : array-like, shape (n, d)
        Feature matrix.
    skew_correction : bool, default True
        Use the per-dimension skewness-selected tail.

    Returns
    -------
    RichResult
        keys: ``scores`` (n, higher = more outlying), ``left_tail``,
        ``right_tail``, ``skewness`` (per dimension), ``n``, ``d``,
        ``method``.

    References
    ----------
    Li, Z., Zhao, Y., Botta, N., Ionescu, C. & Hu, X. (2020). COPOD:
    copula-based outlier detection. *Proceedings of the IEEE
    International Conference on Data Mining (ICDM)*, 1118-1123.
    """
    from scipy import stats

    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2:
        raise ValueError("X must be 2-D (n samples x d features).")
    n, d = X.shape
    if n < 3:
        raise ValueError(f"need at least 3 samples, got {n}.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must be finite.")

    # empirical CDFs (left) and their mirrors (right), never 0 or 1
    left = np.empty((n, d))
    right = np.empty((n, d))
    for j in range(d):
        r = stats.rankdata(X[:, j], method="average")
        left[:, j] = r / (n + 1)
        rr = stats.rankdata(-X[:, j], method="average")
        right[:, j] = rr / (n + 1)

    nl = -np.log(left)
    nr = -np.log(right)
    skew = stats.skew(X, axis=0)
    if skew_correction:
        pick = np.where(skew[None, :] < 0, nl, nr)
        scores = np.maximum(np.maximum(nl.sum(axis=1), nr.sum(axis=1)), pick.sum(axis=1))
    else:
        scores = np.maximum(nl.sum(axis=1), nr.sum(axis=1))

    return RichResult(
        payload={
            "scores": scores,
            "left_tail": left,
            "right_tail": right,
            "skewness": skew,
            "n": int(n),
            "d": int(d),
            "method": "COPOD empirical-copula outlier scores",
        }
    )


def cheatsheet():
    return "copod: score = max over tails of -sum log ECDF; rank-based, scale-free"
