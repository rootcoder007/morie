# morie.fn -- function file (rootcoder007/morie)
"""Bottom-up reconciliation of hierarchical forecasts."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["joseph_bottom_up_reconciliation"]


def joseph_bottom_up_reconciliation(y_hat_bottom, S, base=None, method="bottom_up",
                                    residuals=None):
    r"""Reconcile hierarchical forecasts so they add up.

    Bottom-up sets :math:`\tilde y = S \hat y_{bottom}`: every
    aggregate is the sum of its own leaves, so coherence is exact by
    construction. With ``method="ols"`` or ``"wls"`` the reconciliation
    is the projection

    .. math:: \tilde y = S (S'W^{-1}S)^{-1} S' W^{-1} \hat y,

    applied to the *full* base forecast vector, which lets information
    from the aggregate levels flow back down -- something bottom-up
    cannot do. ``"wls"`` weights by the variance of the base residuals,
    so noisier series pull less.

    Parameters
    ----------
    y_hat_bottom : array-like, shape (m,)
        Base forecasts for the bottom level.
    S : array-like, shape (n, m)
        Summing matrix; its bottom m rows should be the identity.
    base : array-like, shape (n,), optional
        Base forecasts for every series; required for the projection
        methods.
    method : {"bottom_up", "ols", "wls"}
        Reconciliation approach.
    residuals : array-like, shape (T, n), optional
        In-sample base residuals; required for ``"wls"``.

    Returns
    -------
    RichResult
        keys: ``reconciled`` (n,), ``bottom`` (m,), ``coherent``
        (bool), ``P`` (the mapping matrix for projection methods),
        ``method``, ``n``, ``m``.

    References
    ----------
    Hyndman, R. J., Ahmed, R. A., Athanasopoulos, G. & Shang, H. L.
    (2011). Optimal combination forecasts for hierarchical time
    series. *Computational Statistics & Data Analysis*, 55(9),
    2579-2589.

    Wickramasuriya, S. L., Athanasopoulos, G. & Hyndman, R. J. (2019).
    Optimal forecast reconciliation for hierarchical and grouped time
    series through trace minimization. *Journal of the American
    Statistical Association*, 114(526), 804-819.
    """
    S = np.asarray(S, dtype=float)
    if S.ndim != 2:
        raise ValueError("S must be 2-D (n total series x m bottom series).")
    n, m = S.shape
    if method not in ("bottom_up", "ols", "wls"):
        raise ValueError("method must be 'bottom_up', 'ols' or 'wls'.")

    if method == "bottom_up":
        yb = np.asarray(y_hat_bottom, dtype=float).ravel()
        if yb.size != m:
            raise ValueError(f"y_hat_bottom must have {m} entries, got {yb.size}.")
        rec = S @ yb
        return RichResult(
            payload={
                "reconciled": rec, "bottom": yb, "coherent": True, "P": None,
                "n": int(n), "m": int(m),
                "method": "Bottom-up reconciliation (coherent by construction)",
            }
        )

    if base is None:
        raise ValueError(f"method='{method}' needs the full base forecast vector.")
    yh = np.asarray(base, dtype=float).ravel()
    if yh.size != n:
        raise ValueError(f"base must have {n} entries, got {yh.size}.")

    if method == "ols":
        Winv = np.eye(n)
    else:
        if residuals is None:
            raise ValueError("method='wls' needs in-sample residuals.")
        R = np.asarray(residuals, dtype=float)
        if R.ndim != 2 or R.shape[1] != n:
            raise ValueError(f"residuals must be (T, {n}).")
        v = np.maximum(R.var(axis=0), 1e-12)
        Winv = np.diag(1.0 / v)

    P = np.linalg.pinv(S.T @ Winv @ S) @ S.T @ Winv
    rec = S @ (P @ yh)
    return RichResult(
        payload={
            "reconciled": rec, "bottom": P @ yh, "coherent": True, "P": P,
            "n": int(n), "m": int(m),
            "method": f"{method.upper()} optimal-combination reconciliation",
        }
    )


def cheatsheet():
    return "johbu: bottom-up S y_b; ols/wls project the full vector so levels inform each other"
