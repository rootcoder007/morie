# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""OLS via SVD pseudoinverse (robust to singular X^T X)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_svd_pseudoinverse"]


def geron_svd_pseudoinverse(X, y, rcond=None, fit_intercept=False):
    """
    OLS via SVD pseudoinverse (robust to singular X^T X).

    Formula: theta_hat = X^+ y = V Sigma^+ U^T y

    The normal equations need ``X^T X`` to be invertible and square the
    condition number; the pseudoinverse needs neither. Singular values
    below ``rcond * sigma_max`` are treated as zero, which is what makes
    the answer the *minimum-norm* least-squares solution instead of an
    arbitrary one -- for a rank-deficient design, that choice is the only
    well-defined one.

    Parameters
    ----------
    X : array-like
        Design matrix (n, d).
    y : array-like
        Targets, length n.
    rcond : float, optional
        Relative singular-value cutoff; default ``max(n, d) * eps``.
    fit_intercept : bool, default False
        Prepend a column of ones (the intercept is then ``theta[0]``).

    Returns
    -------
    result : RichResult
        Keys: theta, singular_values, rank, condition_number, residuals,
        rss, estimate, n, method.

    Examples
    --------
    Exactly determined, full rank:

    >>> r = geron_svd_pseudoinverse([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [1.0, 1.0, 2.0])
    >>> [round(float(v), 12) for v in r["theta"]]
    [1.0, 1.0]
    >>> int(r["rank"])
    2

    Rank deficient: duplicate columns leave a one-dimensional family of
    exact fits (every theta with ``theta_0 + theta_1 = 2``), and the
    pseudoinverse returns the shortest member of that family, (1, 1),
    rather than failing:

    >>> r2 = geron_svd_pseudoinverse([[1.0, 1.0], [1.0, 1.0]], [2.0, 2.0])
    >>> [round(float(v), 12) for v in r2["theta"]]
    [1.0, 1.0]
    >>> int(r2["rank"])
    1
    >>> round(float(r2["rss"]), 12)
    0.0

    References
    ----------
    Géron Ch 4
    """
    Xa = np.asarray(X, dtype=float)
    if Xa.ndim == 1:
        Xa = Xa.reshape(-1, 1)
    if Xa.ndim != 2 or Xa.size == 0:
        raise ValueError("geron_svd_pseudoinverse: X must be a non-empty (n, d) design matrix")
    ya = np.asarray(y, dtype=float).ravel()
    if ya.size != Xa.shape[0]:
        raise ValueError(f"geron_svd_pseudoinverse: X has {Xa.shape[0]} rows but y has {ya.size} targets")
    if not (np.all(np.isfinite(Xa)) and np.all(np.isfinite(ya))):
        raise ValueError("geron_svd_pseudoinverse: X and y must be finite")
    if fit_intercept:
        Xa = np.hstack([np.ones((Xa.shape[0], 1)), Xa])

    U, sv, Vt = np.linalg.svd(Xa, full_matrices=False)
    cut = (max(Xa.shape) * np.finfo(float).eps) if rcond is None else float(rcond)
    if cut < 0:
        raise ValueError(f"geron_svd_pseudoinverse: rcond must be non-negative, got {cut}")
    if sv.size == 0 or sv[0] == 0:
        raise ValueError("geron_svd_pseudoinverse: X is the zero matrix; no direction to project onto")
    keep = sv > cut * sv[0]
    rank = int(np.sum(keep))
    s_inv = np.zeros_like(sv)
    s_inv[keep] = 1.0 / sv[keep]
    theta = Vt.T @ (s_inv * (U.T @ ya))
    resid = Xa @ theta - ya
    rss = float(resid @ resid)
    cond = float(sv[0] / sv[keep][-1])

    return RichResult(
        title="Least squares via SVD pseudoinverse",
        summary_lines=[
            ("Rank", rank),
            ("Columns", int(Xa.shape[1])),
            ("Condition number", cond),
            ("Residual sum of squares", rss),
        ],
        interpretation=(
            "The pseudoinverse always returns something: when the design is rank deficient it returns "
            "the minimum-norm solution, and the rank/condition number tell you that happened."
        ),
        payload={
            "theta": theta,
            "singular_values": sv,
            "rank": rank,
            "condition_number": cond,
            "residuals": resid,
            "rss": rss,
            "deficient": bool(rank < Xa.shape[1]),
            "estimate": rss,
            "n": int(Xa.shape[0]),
            "method": "Minimum-norm least squares theta = V Sigma^+ U^T y",
        },
    )


def cheatsheet():
    return "hmsvdp: OLS via SVD pseudoinverse (robust to singular X^T X)"
