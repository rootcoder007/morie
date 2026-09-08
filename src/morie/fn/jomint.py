# morie.fn -- function file (rootcoder007/morie)
"""MinT hierarchical forecast reconciliation."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["joseph_mint_reconciliation"]


def joseph_mint_reconciliation(y_hat, S, W=None, method="ols"):
    r"""Reconcile incoherent hierarchical forecasts by minimum trace.

    Independent forecasts of a hierarchy do not add up: the sum of regional
    forecasts differs from the national forecast. Reconciliation projects them
    onto the coherent subspace,

    .. math::
        \tilde y = S\left(S^\top W^{-1} S\right)^{-1} S^\top W^{-1} \hat y,

    where :math:`S` is the summing matrix. MinT chooses :math:`W` as the
    forecast-error covariance, which minimises the trace of the reconciled
    error covariance -- hence the name.

    The result that makes this worth doing: reconciliation **cannot increase**
    total expected error, because it is an orthogonal projection under the
    right metric. Adding up correctly is free, and often better than free,
    since the projection pools information across the hierarchy -- a noisy
    bottom-level forecast is improved by the aggregate that constrains it.

    Bottom-up and top-down are both special cases with degenerate weight
    matrices, and both throw away information that MinT keeps.

    Parameters
    ----------
    y_hat : array-like
        Base forecasts for every node, ordered to match ``S``.
    S : array-like
        Summing matrix ``(n_total, n_bottom)``.
    W : array-like, optional
        Error covariance. Defaults per ``method``.
    method : {"ols", "wls", "mint"}
        ``"ols"`` uses the identity, ``"wls"`` the diagonal of ``W``,
        ``"mint"`` the full ``W``.

    Returns
    -------
    RichResult
        ``reconciled``, ``bottom``, ``coherent``, ``adjustment``,
        ``incoherence_before``.

    References
    ----------
    Wickramasuriya, S. L., Athanasopoulos, G., & Hyndman, R. J. (2019).
        Optimal forecast reconciliation for hierarchical and grouped time
        series through trace minimization. *JASA*, 114(526), 804-819.

    Examples
    --------
    A two-level hierarchy whose base forecasts do not add up is made coherent.

    >>> import numpy as np
    >>> S = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    >>> r = joseph_mint_reconciliation([10.0, 4.0, 5.0], S)
    >>> bool(r["coherent"])
    True
    >>> rec = r["reconciled"]
    >>> bool(abs(rec[0] - (rec[1] + rec[2])) < 1e-9)
    True

    Incoherence before reconciliation is reported, since it measures how much
    the base forecasts disagreed.

    >>> float(round(r["incoherence_before"], 6))
    0.57735

    Already-coherent forecasts are left alone -- the projection is idempotent.

    >>> c = joseph_mint_reconciliation([9.0, 4.0, 5.0], S)
    >>> bool(np.allclose(c["reconciled"], [9.0, 4.0, 5.0]))
    True

    >>> joseph_mint_reconciliation([1.0, 2.0], S)
    Traceback (most recent call last):
        ...
    ValueError: y_hat has 2 entries but S has 3 rows
    """
    y = np.atleast_1d(np.asarray(y_hat, dtype=float)).ravel()
    S = np.atleast_2d(np.asarray(S, dtype=float))
    if y.size != S.shape[0]:
        raise ValueError(f"y_hat has {y.size} entries but S has {S.shape[0]} rows")
    n, m = S.shape
    if method not in ("ols", "wls", "mint"):
        raise ValueError('method must be "ols", "wls" or "mint"')
    if W is None:
        Wm = np.eye(n)
    else:
        Wm = np.atleast_2d(np.asarray(W, dtype=float))
        if Wm.shape != (n, n):
            raise ValueError(f"W must be ({n}, {n})")
    if method == "ols":
        Wm = np.eye(n)
    elif method == "wls":
        Wm = np.diag(np.diag(Wm))
    Wi = np.linalg.pinv(Wm)
    G = np.linalg.pinv(S.T @ Wi @ S) @ S.T @ Wi
    bottom = G @ y
    rec = S @ bottom
    # Incoherence: how far the base forecasts were from the coherent subspace.
    resid = y - S @ np.linalg.pinv(S) @ y
    return RichResult(
        title=f"MinT reconciliation ({method})",
        summary_lines=[("nodes", int(n)), ("bottom", int(m)),
                       ("incoherence before", float(np.linalg.norm(resid)))],
        payload={
            "reconciled": rec, "bottom": bottom,
            "coherent": bool(np.allclose(rec, S @ bottom)),
            "adjustment": rec - y,
            "incoherence_before": float(np.linalg.norm(resid)),
            "G": G, "method_used": method,
            "method": "joseph_mint_reconciliation",
        },
    )


def cheatsheet():
    return "jomint: orthogonal projection, so reconciling CANNOT increase error; bottom-up/top-down discard information"
