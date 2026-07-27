# morie.fn -- function file (rootcoder007/morie)
"""Proximal causal inference via a two-stage proxy bridge function."""

import numpy as np

from ._richresult import RichResult

__all__ = ["causal_proximal_proxy"]


def causal_proximal_proxy(y, A, Z_proxy, W_proxy, X=None):
    r"""Two-stage-least-squares proximal (bridge-function) estimator.

    When an unmeasured confounder U is not observed but two proxies
    are -- a treatment-inducing proxy Z (independent of Y given U, A)
    and an outcome-inducing proxy W (independent of A given U) --
    the outcome bridge function :math:`h(W, A, X)` solves

    .. math:: E[Y \mid A, Z, X] = E[h(W, A, X) \mid A, Z, X],

    and :math:`E[h(W, a, X)]` identifies :math:`E[Y(a)]` even though U
    is never observed. In the linear case this is exactly two-stage
    least squares with W instrumented by Z: regress W on (A, Z, X),
    then Y on (A, Ŵ, X). The coefficient on A is the proximal effect.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    A : array-like, shape (n,)
        Treatment.
    Z_proxy : array-like, shape (n,) or (n, q)
        Treatment-inducing proxies.
    W_proxy : array-like, shape (n,) or (n, q)
        Outcome-inducing proxies (same count as Z).
    X : array-like, optional
        Measured covariates.

    Returns
    -------
    RichResult
        keys: ``estimate`` (coefficient on A), ``bridge_coefficients``,
        ``first_stage_r2``, ``naive`` (the unadjusted A coefficient,
        for contrast), ``n``, ``method``.

    References
    ----------
    Miao, W., Geng, Z. & Tchetgen Tchetgen, E. J. (2018).
    Identifying causal effects with proxy variables of an unmeasured
    confounder. *Biometrika*, 105(4), 987-993.

    Tchetgen Tchetgen, E. J., Ying, A., Cui, Y., Shi, X. & Miao, W.
    (2024). An introduction to proximal causal inference. *Statistical
    Science*, 39(3), 375-390.
    """
    y = np.asarray(y, dtype=float).ravel()
    A = np.asarray(A, dtype=float).ravel()
    Z = np.asarray(Z_proxy, dtype=float)
    W = np.asarray(W_proxy, dtype=float)
    if Z.ndim == 1:
        Z = Z[:, None]
    if W.ndim == 1:
        W = W[:, None]
    n = y.size
    if not (A.size == n and Z.shape[0] == n and W.shape[0] == n):
        raise ValueError("y, A, Z_proxy, W_proxy must share their first dimension.")
    if Z.shape[1] < W.shape[1]:
        raise ValueError("need at least as many Z proxies as W proxies for identification.")
    if X is None:
        Xa = np.empty((n, 0))
    else:
        Xa = np.asarray(X, dtype=float)
        if Xa.ndim == 1:
            Xa = Xa[:, None]
        if Xa.shape[0] != n:
            raise ValueError(f"X has {Xa.shape[0]} rows but y has {n}.")
    if n < W.shape[1] + Z.shape[1] + Xa.shape[1] + 4:
        raise ValueError("too few observations for the two stages.")

    one = np.ones(n)
    D1 = np.column_stack([one, A, Z, Xa])
    What = np.empty_like(W)
    r2 = []
    for j in range(W.shape[1]):
        b, *_ = np.linalg.lstsq(D1, W[:, j], rcond=None)
        fit = D1 @ b
        What[:, j] = fit
        tss = float(((W[:, j] - W[:, j].mean()) ** 2).sum())
        r2.append(1 - float(((W[:, j] - fit) ** 2).sum()) / tss if tss > 0 else float("nan"))

    D2 = np.column_stack([one, A, What, Xa])
    b2, *_ = np.linalg.lstsq(D2, y, rcond=None)

    Dn = np.column_stack([one, A, Xa])
    bn, *_ = np.linalg.lstsq(Dn, y, rcond=None)

    return RichResult(
        payload={
            "estimate": float(b2[1]),
            "bridge_coefficients": b2.astype(float),
            "first_stage_r2": np.array(r2),
            "naive": float(bn[1]),
            "n": int(n),
            "method": "Proximal causal inference via a linear outcome bridge (2SLS in W)",
        }
    )


def cheatsheet():
    return "causrho: stage 1 W ~ A + Z + X; stage 2 Y ~ A + What + X; A coefficient is the effect"
