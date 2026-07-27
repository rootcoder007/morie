# morie.fn -- function file (rootcoder007/morie)
"""Synthetic Control Method (Abadie-Diamond-Hainmueller)."""

import numpy as np

from ._richresult import RichResult
from .caussc import causal_synthetic_control

__all__ = ["synthetic_control_method"]


def synthetic_control_method(y_treated, y_controls, treat_time, V=None):
    """Full SCM: weights from the pre-period, effect path after it.

    Fits simplex weights so the donor combination tracks the treated
    unit's pre-treatment outcome path, then reports the post-treatment
    gap between the observed and synthetic series. The pre-period RMSE
    is the fit diagnostic Abadie et al. require before interpreting
    the post gap as an effect.

    Parameters
    ----------
    y_treated : array-like, shape (T,)
        Treated unit's outcome series.
    y_controls : array-like, shape (T, J)
        Donor outcomes, one column per donor.
    treat_time : int
        First post-treatment index (pre-period is ``[:treat_time]``).
    V : array-like, optional
        Predictor weights over the pre-period observations.

    Returns
    -------
    RichResult
        keys: ``weights``, ``att`` (mean post gap), ``gap`` (T,),
        ``synthetic`` (T,), ``rmse_pre``, ``treat_time``, ``method``.

    References
    ----------
    Abadie, A., Diamond, A. & Hainmueller, J. (2010). Synthetic
    control methods for comparative case studies. *Journal of the
    American Statistical Association*, 105(490), 493-505.
    """
    y1 = np.asarray(y_treated, dtype=float).ravel()
    Y0 = np.asarray(y_controls, dtype=float)
    if Y0.ndim != 2 or Y0.shape[0] != y1.size:
        raise ValueError("y_controls must be (T, J) matching y_treated.")
    t0 = int(treat_time)
    if not 2 <= t0 < y1.size:
        raise ValueError(f"treat_time must lie in [2, T), got {t0}.")

    fit = causal_synthetic_control(y1[:t0], Y0[:t0], V=V)
    w = fit["weights"]
    synth = Y0 @ w
    gap = y1 - synth

    return RichResult(
        payload={
            "weights": w,
            "att": float(gap[t0:].mean()),
            "gap": gap,
            "synthetic": synth,
            "rmse_pre": fit["rmse_pre"],
            "treat_time": t0,
            "method": "Synthetic Control Method (Abadie-Diamond-Hainmueller)",
        }
    )


def cheatsheet():
    return "scmaba: SCM -- pre-period simplex weights, post-period gap = effect path"
