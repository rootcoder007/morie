# morie.fn -- function file (rootcoder007/morie)
"""Bootstrap bias estimate and correction."""

import numpy as np

from ._richresult import RichResult

__all__ = ["boot_bias_estimator"]


def boot_bias_estimator(theta_hat, theta_b):
    r"""The bootstrap bias estimate (Efron 1979; Efron and Tibshirani
    1993, Ch. 10),

    .. math:: \widehat{\mathrm{bias}} = \bar\theta^* - \hat\theta,

    and the corrected estimate
    :math:`\hat\theta - \widehat{\mathrm{bias}} = 2\hat\theta -
    \bar\theta^*`.

    The direction trips people: the correction SUBTRACTS the bias,
    so the corrected value is :math:`2\hat\theta - \bar\theta^*`, on
    the OPPOSITE side of :math:`\hat\theta` from the replicate mean.
    Adding it -- using :math:`\bar\theta^*` itself -- doubles the
    bias instead of removing it, and the test asserts the direction
    on a statistic whose bias is known in closed form (the MLE
    variance, biased by exactly :math:`-\sigma^2/n`).

    Efron and Tibshirani's own warning is carried in the output:
    bias correction is dangerous in practice, because the correction
    adds variance and the corrected estimator can easily have larger
    mean squared error than the raw one. Report the bias; correct
    only when the bias genuinely dominates.

    Parameters
    ----------
    theta_hat : float
        The statistic on the original data.
    theta_b : array-like
        Bootstrap replicates.

    Returns
    -------
    RichResult
        keys: ``bias``, ``corrected``, ``estimate``,
        ``mean_replicate``, ``relative_bias``, ``B``,
        ``correction_warning``, ``method``.

    References
    ----------
    Efron, B. (1979), *Annals of Statistics* 7:1-26. Efron, B. and
    Tibshirani, R. J. (1993), *An Introduction to the Bootstrap*,
    Ch. 10.
    """
    th = float(theta_hat)
    r = np.asarray(theta_b, dtype=float).ravel()
    if r.size < 2:
        raise ValueError(f"need at least 2 replicates, got {r.size}.")
    bias = float(r.mean() - th)
    return RichResult(payload={
        "bias": bias, "corrected": th - bias, "estimate": th,
        "mean_replicate": float(r.mean()),
        "relative_bias": bias / th if th != 0 else np.inf,
        "B": int(r.size),
        "direction_note": "the corrected value is 2 theta_hat - mean(reps), "
                          "on the OPPOSITE side of theta_hat from the "
                          "replicate mean; using the replicate mean itself "
                          "doubles the bias",
        "correction_warning": "correction adds variance and can raise the "
                              "MSE (Efron-Tibshirani Ch. 10); report the "
                              "bias, correct only when it dominates",
        "method": "Bootstrap bias = mean(replicates) - estimate, corrected = 2*estimate - mean"})


def cheatsheet():
    return "btbias: corrected = 2 theta - mean(reps) -- the correction points AWAY from the replicate mean"
