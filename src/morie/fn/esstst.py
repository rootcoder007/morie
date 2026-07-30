# morie.fn -- function file (rootcoder007/morie)
"""Kish effective sample size from weights."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["effective_sample_size_weights"]


def effective_sample_size_weights(weights):
    r"""Kish's effective sample size for an arbitrary weight vector.

    .. math::
        n_{\text{eff}} = \frac{\left(\sum_i w_i\right)^2}{\sum_i w_i^2}.

    Equal weights give :math:`n_{\text{eff}} = n`; any variation costs
    information. The quantity is the same one that appears in importance
    sampling, propensity weighting and survey estimation, because in every
    case the question is how many equally-weighted observations carry the same
    precision.

    The single most useful reading is ``max_share`` -- one unit's fraction of
    the total weight. A weighted estimate where one observation carries 30% of
    the weight is, for practical purposes, an estimate from a handful of
    points however large :math:`n` is, and no confidence interval computed
    from :math:`n` will reflect that.

    Parameters
    ----------
    weights : array-like
        Non-negative weights.

    Returns
    -------
    RichResult
        ``ess``, ``n``, ``efficiency``, ``max_share``, ``cv``.

    References
    ----------
    Kish, L. (1965). *Survey Sampling*. Wiley.

    Examples
    --------
    Equal weights lose nothing.

    >>> import numpy as np
    >>> float(effective_sample_size_weights(np.ones(100))["ess"])
    100.0

    One dominant weight collapses the effective sample regardless of n.

    >>> w = np.r_[np.ones(999), 1000.0]
    >>> r = effective_sample_size_weights(w)
    >>> bool(r["ess"] < 5 and r["max_share"] > 0.4)
    True

    Efficiency is the fraction of the nominal sample that survives.

    >>> bool(0 < r["efficiency"] < 0.01)
    True

    >>> effective_sample_size_weights([1.0, -1.0])
    Traceback (most recent call last):
        ...
    ValueError: weights must be non-negative
    """
    w = np.atleast_1d(np.asarray(weights, dtype=float)).ravel()
    if w.size == 0:
        raise ValueError("weights must be non-empty")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    s = float(w.sum())
    if s <= 0:
        raise ValueError("weights must not all be zero")
    ess = float(s**2 / float(np.sum(w**2)))
    share = float(w.max() / s)
    cv = float(np.std(w, ddof=1) / np.mean(w)) if w.size > 1 else 0.0
    return RichResult(
        title="Effective sample size (Kish)",
        summary_lines=[("n", int(w.size)), ("ESS", ess),
                       ("efficiency", ess / w.size), ("max share", share)],
        warnings=([f"one observation carries {share:.1%} of the total weight; "
                   "the estimate rests on a handful of points however large n is"]
                  if share > 0.1 else []),
        payload={
            "ess": ess, "n": int(w.size), "efficiency": float(ess / w.size),
            "max_share": share, "cv": cv,
            "method": "effective_sample_size_weights",
        },
    )


def cheatsheet():
    return "esstst: (sum w)^2 / sum w^2; read max_share -- one dominant weight collapses ESS whatever n is"
