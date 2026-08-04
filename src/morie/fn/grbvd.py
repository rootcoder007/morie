# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bias/variance/noise decomposition of squared error.

Geron, A. (2026). Hands-On Machine Learning with Scikit-Learn and PyTorch. O'Reilly, ch. 4 p. 158 (stated in words there; no formula printed)
"""

from . import _geron as _core

from ._richresult import RichResult

__all__ = ["bvdecomp", "geron_bias_variance_decomposition"]

_METHOD = "Bias/variance/noise decomposition of squared error"


def bvdecomp(preds, truth, noisevar=0.0):
    """Bias/variance/noise decomposition of squared error.

    Squared-error decomposition into bias, variance and noise.

    p. 158 states IN WORDS that the generalization error is the sum of
    bias, variance and irreducible error; NO formula is printed there.
    The decomposition computed here is therefore stated explicitly
    rather than attributed to the book:

    MSE = mean_i (mean_b f_b(x_i) - y_i)^2      (bias^2)
    + mean_i var_b(f_b(x_i))               (variance)
    + noisevar                             (irreducible)

    ``preds`` is a B-by-n matrix: one row per model trained on a
    different training set, one column per test point.  Supplying the
    predictions rather than fitting anything keeps the routine free of
    the resampling randomness the book's discussion assumes.

    Parameters
    ----------
    preds : as documented for the shelf core
        See ``morie.fn._geron.bvdecomp``.
    truth : as documented for the shelf core
        See ``morie.fn._geron.bvdecomp``.
    noisevar : as documented for the shelf core
        See ``morie.fn._geron.bvdecomp``.

    Returns
    -------
    result : RichResult
        Payload keys: bias2, variance, total, mse.

    References
    ----------
    Geron, A. (2026). Hands-On Machine Learning with Scikit-Learn and PyTorch. O'Reilly, ch. 4 p. 158 (stated in words there; no formula printed)
    """
    res = _core.bvdecomp(preds=preds, truth=truth, noisevar=noisevar)
    return RichResult(
        title=_METHOD,
        summary_lines=[("bias2", res["bias2"]), ("variance", res["variance"]), ("total", res["total"]), ("mse", res["mse"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
geron_bias_variance_decomposition = bvdecomp


def cheatsheet():
    return "bvdecomp: Bias/variance/noise decomposition of squared error"
