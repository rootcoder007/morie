# SPDX-License-Identifier: AGPL-3.0-or-later
"""Lognormal kriging."""

from . import _array_core as np

from ._richresult import RichResult
from ._schab_krig import simple_kriging

__all__ = ["schabenberger_lognormal_kriging"]


def schabenberger_lognormal_kriging(coords, z, target, cov_model=None, mu=None):
    """Predict Z(s0) when log Z is a Gaussian random field, Sec. 5.6.1.

    With Y(s) = log Z(s) Gaussian, the obvious predictor exp{p_sk(Y; s0)} is
    BIASED for Z(s0). Applying the Aitchison and Brown (1957) lognormal
    moments twice -- once to the predictor and once inversely to mu_Y --
    gives the bias-corrected form, eq (5.54):

        p_slk(Z; s0) = exp{ p_sk(Y; s0) + sigma^2_sk(Y; s0) / 2 }

    and the text notes this equals E[Z(s0) | Z], the conditional mean.

    The correction is the SIMPLE KRIGING variance, not the process variance.
    The two differ by c' Sigma^-1 c, which is exactly the variance of the
    predictor: the book derives the correction as
    sigma^2_Y/2 - Var[p_sk(Y; s0)]/2 and then identifies that difference as
    sigma^2_sk/2. A previous docstring of this module gave the correction as
    sigma^2_Y/2, which over-corrects everywhere the prediction is informed by
    data, and by the most at the locations where kriging is most confident.

    Parameters
    ----------
    coords : array-like, shape (n, d)
        Sampling locations.
    z : array-like, shape (n,)
        Observed values on the ORIGINAL scale; must be strictly positive,
        since the model is that their logarithm is Gaussian.
    target : array-like, shape (d,)
        Prediction location.
    cov_model : callable, optional
        Covariance as a function of lag, for the Y (log) scale.
    mu : float, optional
        Known mean of Y. Defaults to the mean of log z.

    Returns
    -------
    RichResult
        Keys: ``prediction``, ``naive_prediction``, ``log_prediction``,
        ``log_variance``, ``bias_factor``.

    References
    ----------
    Schabenberger Ch 5, Sec 5.6.1
    """
    z = np.asarray(z, dtype=float).ravel()
    if np.any(z <= 0.0):
        raise ValueError("`z` must be strictly positive for a lognormal model")
    y = np.log(z)
    # simple_kriging is vectorised over targets and returns arrays; this
    # module predicts at one location.
    target = np.atleast_2d(np.asarray(target, dtype=float))
    pred_arr, var_arr = simple_kriging(coords, y, target, cov_model=cov_model,
                                       mu=mu)[:2]
    pred_y = float(np.asarray(pred_arr).ravel()[0])
    var_y = float(np.asarray(var_arr).ravel()[0])
    naive = float(np.exp(pred_y))
    corrected = float(np.exp(pred_y + 0.5 * var_y))
    return RichResult(
        title="Lognormal kriging",
        summary_lines=[("prediction", corrected),
                       ("uncorrected exp(p_sk)", naive),
                       ("log-scale kriging variance", float(var_y))],
        payload={"prediction": corrected, "naive_prediction": naive,
                 "log_prediction": float(pred_y), "log_variance": float(var_y),
                 "bias_factor": float(np.exp(0.5 * var_y)),
                 "method": "lognormal (simple) kriging"},
    )


def cheatsheet():
    return "splgk: lognormal kriging, eq (5.54) (Schabenberger Sec 5.6.1)"
