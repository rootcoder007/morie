# morie.fn -- function file (rootcoder007/morie)
"""General M-estimator regression by IRLS."""

import numpy as np

from ._richresult import RichResult

__all__ = ["m_regression"]


def m_regression(X, y, psi="huber", c=None, max_iter=100):
    r"""M-estimator regression with a choice of psi function,
    computed by iteratively reweighted least squares (Huber 1973).

    ``psi="huber"`` is the monotone :math:`\psi_c(u) = \max(-c,
    \min(u, c))`, default :math:`c = 1.345`; ``psi="bisquare"`` is
    Tukey's redescending biweight, default :math:`c = 4.685`. The
    distinction matters: a MONOTONE psi has a unique solution and
    IRLS converges from anywhere, but grants every observation a
    non-zero (if bounded) vote; a REDESCENDING psi zeroes out gross
    outliers entirely, but the objective is non-convex, so the answer
    depends on the start -- IRLS from least squares finds a local
    solution, and for genuine high-breakdown behaviour the S/MM
    chain (``morie.fn.mmreg``) is the correct tool, not this one.
    The output says which situation it is in.

    Parameters
    ----------
    x, y : array-like
        Design (constant added when absent) and response.
    psi : {"huber", "bisquare"}
        Loss family.
    c : float, optional
        Tuning constant; the 95%-efficiency value of the chosen
        family when omitted.
    max_iter : int, default 100
        IRLS iterations.

    Returns
    -------
    RichResult
        keys: ``beta``, ``scale``, ``residuals``, ``weights``, ``psi``,
        ``c``, ``monotone``, ``unique_solution``, ``converged``,
        ``start_dependent_warning``, ``n``, ``p``, ``method``.

    References
    ----------
    Huber, P. J. (1973), *Annals of Statistics* 1:799-821. Beaton,
    A. E. and Tukey, J. W. (1974), *Technometrics* 16:147-185, for
    the biweight.
    """
    from ._robust import (HUBER_C_95, TUKEY_C_95, mad_scale, prepare_design,
                          tukey_weight)

    A, yv = prepare_design(X, y)
    n, p = A.shape
    if n <= p:
        raise ValueError(f"need more observations than parameters, "
                         f"got n = {n}, p = {p}.")
    if psi not in ("huber", "bisquare"):
        raise ValueError("psi must be 'huber' or 'bisquare'.")
    cc = float(c) if c is not None else (
        HUBER_C_95 if psi == "huber" else TUKEY_C_95)
    if cc <= 0:
        raise ValueError(f"c must be positive, got {cc}.")

    def wfun(u):
        if psi == "huber":
            au = np.maximum(np.abs(u), 1e-300)
            return np.where(au <= cc, 1.0, cc / au)
        return tukey_weight(u, cc)

    beta = np.linalg.lstsq(A, yv, rcond=None)[0]
    conv = False
    scale = 1.0
    for _ in range(int(max_iter)):
        r = yv - A @ beta
        scale = mad_scale(r)
        if scale <= 0:
            conv = True
            break
        w = wfun(r / scale)
        if not np.any(w > 0):
            break
        Aw = A * w[:, None]
        beta_new = np.linalg.lstsq(Aw.T @ A, Aw.T @ yv, rcond=None)[0]
        if np.max(np.abs(beta_new - beta)) < 1e-10 * (1 + np.max(np.abs(beta))):
            beta = beta_new
            conv = True
            break
        beta = beta_new
    r = yv - A @ beta
    return RichResult(payload={
        "beta": beta, "scale": float(scale), "residuals": r,
        "weights": wfun(r / scale) if scale > 0 else np.ones(n),
        "psi": psi, "c": cc,
        "monotone": psi == "huber",
        "unique_solution": psi == "huber",
        "converged": bool(conv),
        "start_dependent_warning": None if psi == "huber" else (
            "the biweight objective is non-convex: IRLS from least squares "
            "finds a LOCAL solution, and for high-breakdown behaviour use "
            "the S/MM chain (morie.fn.mmreg) instead"),
        "n": int(n), "p": int(p),
        "method": f"M-regression by IRLS, {psi} psi at c = {cc}"})


def cheatsheet():
    return "mestrg: monotone psi = unique solution; redescending psi = start-dependent, use MM"
