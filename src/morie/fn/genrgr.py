# morie.fn -- function file (rootcoder007/morie)
"""Generalised regression (calibration) estimator."""

import numpy as np

from ._richresult import RichResult

__all__ = ["calibration_greg"]


def calibration_greg(y, x, weights, totals):
    r"""Generalised regression (GREG) estimator:

    .. math:: \hat T_{GREG} = \hat T_{HT}
              + \big(\mathbf T_x - \hat{\mathbf T}_x\big)'\mathbf B,

    the Horvitz-Thompson total corrected by how far the SAMPLE
    totals of the auxiliaries fall from their known population
    totals.

    GREG is where design-based and model-based thinking meet. The
    coefficient :math:`\mathbf B` comes from a working model, but
    the estimator stays design-consistent whether or not that model
    is right: a wrong model costs efficiency, not validity. That
    property -- ``design_consistent_regardless_of_model`` -- is the
    reason GREG is the standard production estimator in national
    statistics, and it is worth stating because the usual intuition
    about regression adjustment does not carry it.

    Parameters
    ----------
    y : array-like, shape (n,)
        Study variable.
    x : array-like, shape (n, p)
        Auxiliaries.
    weights : array-like, shape (n,)
        Design weights :math:`1/\pi_i`.
    totals : array-like, shape (p,)
        Known population totals of the auxiliaries.

    Returns
    -------
    RichResult
        keys: ``total``, ``ht_total``, ``correction``, ``B``,
        ``residual_totals``,
        ``design_consistent_regardless_of_model`` (True), ``n``,
        ``p``, ``method``.
    """
    from ._survey import check_weights

    yv = np.asarray(y, dtype=float).ravel()
    X = np.atleast_2d(np.asarray(x, dtype=float))
    if X.shape[0] != yv.size:
        X = X.T
    if X.shape[0] != yv.size:
        raise ValueError("x must have one row per entry of y.")
    n, p = X.shape
    w = check_weights(weights, n)
    T = np.atleast_1d(np.asarray(totals, dtype=float)).ravel()
    if T.size != p:
        raise ValueError(f"totals has {T.size} entries for {p} auxiliaries.")
    ht = float(np.sum(w * yv))
    Tx_hat = (w[:, None] * X).sum(axis=0)
    A = (w[:, None] * X).T @ X
    B = np.linalg.pinv(A) @ ((w[:, None] * X).T @ yv)
    corr = float((T - Tx_hat) @ B)
    return RichResult(payload={
        "total": ht + corr, "ht_total": ht, "correction": corr, "B": B,
        "residual_totals": T - Tx_hat,
        "design_consistent_regardless_of_model": True,
        "model_role": "the working model sets B and therefore the EFFICIENCY; "
                      "it does not affect design consistency",
        "n": int(n), "p": int(p),
        "method": "GREG (Sarndal); design-consistent whether or not the working model holds"})


def cheatsheet():
    return "genrgr: a wrong working model costs GREG efficiency, never validity"
