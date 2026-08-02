# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Effective degrees of freedom of a linear smoother (ESL Ch 5.4)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["esl_effective_dof"]


def esl_effective_dof(S):
    """
    Effective degrees of freedom df(S) = trace(S).

    For a linear smoother y_hat = S y, the trace of the smoother
    matrix is the effective parameter count. Two related traces come
    along because they coincide only for a PROJECTION (S = S' = S^2):
    trace(S S') and trace(2S - S S'), the latter being the variance-
    correct df used in C_p for non-projection smoothers. The payload
    reports whether S is idempotent so the caller can tell which case
    they are in.

    Parameters
    ----------
    S : array-like, shape (n, n)
        Smoother matrix.

    Returns
    -------
    result : dict
        Keys: estimate (trace S), trace_ssT, df_variance
        (trace(2S - S S')), is_projection, n, method.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), Ch 5.4.1 and Ch 7.6.

    Examples
    --------
    A projection onto a 2-dimensional subspace has df 2, and all three
    traces agree:

    >>> import numpy as np
    >>> X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    >>> H = X @ np.linalg.inv(X.T @ X) @ X.T
    >>> out = esl_effective_dof(H)
    >>> round(out["estimate"], 12)
    2.0
    >>> out["is_projection"]
    True
    >>> round(out["df_variance"], 12)
    2.0
    >>> shrunk = esl_effective_dof(0.5 * H)
    >>> shrunk["is_projection"]
    False
    >>> round(shrunk["estimate"], 12)
    1.0
    """
    S = np.atleast_2d(np.asarray(S, dtype=float))
    if S.shape[0] != S.shape[1]:
        raise ValueError(f"the smoother matrix must be square; got shape {S.shape}.")
    tr = float(np.trace(S))
    tr_ssT = float(np.trace(S @ S.T))
    return RichResult(payload={
        "estimate": tr, "trace_ssT": tr_ssT,
        "df_variance": float(np.trace(2.0 * S - S @ S.T)),
        "is_projection": bool(np.allclose(S, S.T) and np.allclose(S @ S, S)),
        "n": int(S.shape[0]),
        "method": "df(S) = trace(S); trace(SS') and 2S-SS' alongside"})


def cheatsheet():
    return "esleff: df = tr(S); tr(SS') and tr(2S-SS') differ unless S projects"
