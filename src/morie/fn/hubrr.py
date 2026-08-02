# morie.fn -- function file (rootcoder007/morie)
"""Huber M-estimator regression."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["huber_regression"]


def huber_regression(X, y, c=None, max_iter=100):
    r"""Huber's M-estimator for regression (Huber 1973), solving

    .. math:: \sum_i \psi_c\!\left(\frac{y_i - x_i'\beta}{\hat\sigma}
              \right) x_i = 0, \qquad \psi_c(u) = \max(-c, \min(u, c)),

    by iteratively reweighted least squares with the MAD of the
    residuals as the (re-estimated) scale.

    :math:`c = 1.345` is not folklore: it is the solution of the
    efficiency equation
    :math:`(\int \psi_c'\,d\Phi)^2 / \int \psi_c^2\,d\Phi = 0.95`, so
    the estimator gives up exactly 5% efficiency at the normal in
    exchange for a bounded influence function. What Huber's psi does
    NOT buy is protection against bad LEVERAGE points: it bounds the
    influence of a residual, not of a design row, and its breakdown
    point in regression is 0. A cluster of outliers at a leverage
    position still ruins it -- that failure is demonstrated in the
    tests, and it is what the S/MM estimators exist to fix.

    Parameters
    ----------
    x : array-like, shape (n, p)
        Design; a constant column is added when absent.
    y : array-like, shape (n,)
        Response.
    c : float, optional
        Tuning constant; 1.345 (95% normal efficiency) when omitted.
    max_iter : int, default 100
        IRLS iterations.

    Returns
    -------
    RichResult
        keys: ``beta``, ``scale``, ``se``, ``residuals``, ``weights``,
        ``c``, ``efficiency_at_c``, ``converged``,
        ``bounded_influence_in``, ``breakdown``, ``n``, ``p``,
        ``method``.

    References
    ----------
    Huber, P. J. (1973), "Robust regression: asymptotics,
    conjectures and Monte Carlo", *Annals of Statistics* 1:799-821.
    Huber, P. J. (1964), *Annals of Mathematical Statistics*
    35:73-101, for the psi function.
    """
    from ._robust import HUBER_C_95, mad_scale, prepare_design

    A, yv = prepare_design(X, y)
    n, p = A.shape
    if n <= p:
        raise ValueError(f"need more observations than parameters, "
                         f"got n = {n}, p = {p}.")
    cc = HUBER_C_95 if c is None else float(c)
    if cc <= 0:
        raise ValueError(f"c must be positive, got {cc}.")
    beta = np.linalg.lstsq(A, yv, rcond=None)[0]
    conv = False
    scale = 1.0
    for _ in range(int(max_iter)):
        r = yv - A @ beta
        scale = mad_scale(r)
        if scale <= 0:
            conv = True
            break
        u = r / scale
        w = np.where(np.abs(u) <= cc, 1.0, cc / np.abs(u))
        Anew = A * w[:, None]
        beta_new = np.linalg.lstsq(Anew.T @ A, Anew.T @ yv, rcond=None)[0]
        if np.max(np.abs(beta_new - beta)) < 1e-10 * (1 + np.max(np.abs(beta))):
            beta = beta_new
            conv = True
            break
        beta = beta_new
    r = yv - A @ beta
    u = r / scale if scale > 0 else r
    w = np.where(np.abs(u) <= cc, 1.0, cc / np.maximum(np.abs(u), 1e-300))
    # asymptotic sandwich: sigma^2 * E[psi^2]/E[psi']^2 * (X'X)^-1
    psi = np.clip(u, -cc, cc)
    dpsi = (np.abs(u) <= cc).astype(float)
    denom = float(dpsi.mean())
    kappa = float(np.mean(psi ** 2)) / max(denom ** 2, 1e-12)
    XtX_inv = np.linalg.pinv(A.T @ A)
    se = np.sqrt(np.maximum(np.diag(XtX_inv) * kappa * scale ** 2, 0.0))
    return RichResult(payload={
        "beta": beta, "scale": float(scale), "se": se,
        "residuals": r, "weights": w, "c": cc,
        "efficiency_at_c": 0.95 if c is None else None,
        "converged": bool(conv),
        "bounded_influence_in": "the residual only -- NOT the design; a bad "
                                "leverage cluster still breaks it, which is "
                                "what the S/MM estimators fix",
        "breakdown": 0.0,
        "n": int(n), "p": int(p),
        "method": "Huber M-regression by IRLS, c = 1.345 for 95% normal efficiency"})


def cheatsheet():
    return "hubrr: bounds the residual's influence, not the design's -- leverage still kills it"
