# morie.fn -- function file (rootcoder007/morie)
"""GREG-type calibration estimator."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["calibration_estimator"]


def calibration_estimator(y, X, weights, totals, max_iter=100, tol=1e-10):
    r"""Calibration estimator: find weights closest to the design
    weights that reproduce known population margins exactly,

    .. math:: \min_{w} \sum_i \frac{(w_i - d_i)^2}{d_i}
              \quad\text{subject to}\quad
              \sum_i w_i \mathbf x_i = \mathbf T_x .

    The chi-square distance gives a closed-form solution and
    reproduces GREG exactly, which is why calibration and GREG are
    often described as the same estimator seen from two directions:
    one adjusts the ESTIMATE, the other adjusts the WEIGHTS.

    Calibrating on margins is what makes published tables internally
    consistent -- a survey's estimated age distribution then matches
    the census exactly, by construction. The cost is that chi-square
    calibration can produce NEGATIVE weights, which are awkward to
    defend and are the reason bounded distance functions (raking,
    logit) exist. ``n_negative`` reports them rather than silently
    clipping.

    Parameters
    ----------
    y : array-like, shape (n,)
        Study variable.
    X : array-like, shape (n, p)
        Calibration variables.
    weights : array-like, shape (n,)
        Design weights.
    totals : array-like, shape (p,)
        Population totals to reproduce.
    max_iter, tol
        Retained for interface symmetry; the chi-square solution is
        closed form.

    Returns
    -------
    RichResult
        keys: ``total``, ``calibrated_weights``, ``margins_reproduced``,
        ``max_margin_error``, ``n_negative``, ``weight_ratio_range``,
        ``distance``, ``equals_greg`` (True), ``n``, ``p``,
        ``method``.
    """
    from ._survey import check_weights

    yv = np.asarray(y, dtype=float).ravel()
    Xm = np.atleast_2d(np.asarray(X, dtype=float))
    if Xm.shape[0] != yv.size:
        Xm = Xm.T
    if Xm.shape[0] != yv.size:
        raise ValueError("X must have one row per entry of y.")
    n, p = Xm.shape
    d = check_weights(weights, n, "design weights")
    T = np.atleast_1d(np.asarray(totals, dtype=float)).ravel()
    if T.size != p:
        raise ValueError(f"totals has {T.size} entries for {p} columns.")
    # chi-square calibration: w = d(1 + x'lambda), lambda solving the
    # margin constraints exactly
    A = (d[:, None] * Xm).T @ Xm
    lam = np.linalg.pinv(A) @ (T - (d[:, None] * Xm).sum(axis=0))
    w = d * (1.0 + Xm @ lam)
    achieved = (w[:, None] * Xm).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(d > 0, w / d, np.nan)
    return RichResult(payload={
        "total": float(np.sum(w * yv)), "calibrated_weights": w,
        "margins_reproduced": bool(np.allclose(achieved, T, rtol=1e-8,
                                               atol=1e-8)),
        "max_margin_error": float(np.max(np.abs(achieved - T))),
        "n_negative": int(np.sum(w < 0)),
        "weight_ratio_range": (float(np.nanmin(ratio)), float(np.nanmax(ratio))),
        "distance": "chi-square, which reproduces GREG exactly",
        "equals_greg": True,
        "negative_weight_note": "chi-square calibration can produce negative "
                                "weights; bounded distances (raking, logit) exist for that",
        "n": int(n), "p": int(p),
        "method": "Calibration to known margins; adjusts the WEIGHTS where GREG adjusts the estimate"})


def cheatsheet():
    return "calibr: same estimator as GREG from the other side -- and it can hand you negative weights"
