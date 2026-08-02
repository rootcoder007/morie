# SPDX-License-Identifier: AGPL-3.0-or-later
"""Trans-Gaussian kriging, and the anamorphosis / normal-scores transform."""

from . import _array_core as np

from ._richresult import RichResult
from ._rng import normal_quantile
from ._schab_krig import ordinary_kriging

__all__ = ["schabenberger_trans_gaussian_kriging", "normal_scores",
           "anamorphosis"]


def normal_scores(z):
    """phi^-1(z) = Phi^-1(F(z)), the normal-scores transform, eq (5.60).

    "matching the percentiles of the data to those of a standard Gaussian
    distribution". F is the empirical CDF; the (i - 1/2)/n plotting position
    is used so no observation maps to an infinite score, which the plain i/n
    would do for the largest value.
    """
    z = np.asarray(z, dtype=float).ravel()
    n = z.size
    if n == 0:
        raise ValueError("`z` must be non-empty")
    order = np.argsort(z, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    return normal_quantile((ranks - 0.5) / n)


def anamorphosis(z, y_new):
    """phi(y) = F^-1(Phi(y)), the inverse of `normal_scores`, eq (5.60).

    Maps a Gaussian score back onto the data scale by matching quantiles.
    Values outside the observed range are clamped to it, since an empirical
    CDF carries no information beyond its support.
    """
    z = np.sort(np.asarray(z, dtype=float).ravel())
    y_new = np.atleast_1d(np.asarray(y_new, dtype=float))
    n = z.size
    scores = normal_quantile((np.arange(1, n + 1) - 0.5) / n)
    return np.interp(y_new, scores, z)


def schabenberger_trans_gaussian_kriging(coords, z, target, phi, dphi, d2phi,
                                         semivariogram_fn):
    """Trans-Gaussian kriging, Sec. 5.6.2.

    With Z(s) = phi(Y(s)) and Y Gaussian, the natural predictor
    phi(p_ok(Y; s0)) is biased. Expanding to second order about mu_Y and
    matching E[p(Z; s0)] to E[Z(s0)] gives eq (5.58):

        p_tg(Z; s0) = phi(p_ok(Y; s0))
                      + phi''(mu_Y)/2 * (sigma^2_ok(Y; s0) - 2 m_Y)

    where m_Y is the Lagrange multiplier from the ordinary kriging of Y. The
    mean squared prediction error, from a first-order expansion, is eq
    (5.59):

        E[(p_tg - Z(s0))^2] ~ [phi'(mu_Y)]^2 sigma^2_ok(Y; s0).

    The multiplier enters with the book's sign convention from (5.20),
    m = -(1 - 1'Gamma^-1 gamma(s0))/(1'Gamma^-1 1); a flipped sign would
    move every prediction by 2 m phi''/2 without any other symptom.

    Parameters
    ----------
    coords : array-like, shape (n, d)
        Sampling locations.
    z : array-like, shape (n,)
        Observed values on the Y (Gaussian) scale.
    target : array-like, shape (d,)
        Prediction location.
    phi, dphi, d2phi : callable
        The transformation and its first two derivatives, evaluated at mu_Y.
    semivariogram_fn : callable
        gamma(h) for the Y scale.

    Returns
    -------
    RichResult
        Keys: ``prediction``, ``naive_prediction``, ``correction``,
        ``mspe``, ``kriging_variance``, ``lagrange``, ``mu_y``.

    References
    ----------
    Schabenberger Ch 5, Sec 5.6.2
    """
    z = np.asarray(z, dtype=float).ravel()
    pred_y, var_ok, _, m = ordinary_kriging(coords, z, target, semivariogram_fn)
    mu_y = float(z.mean())
    naive = float(phi(pred_y))
    correction = 0.5 * float(d2phi(mu_y)) * (var_ok - 2.0 * m)
    mspe = float(dphi(mu_y)) ** 2 * var_ok
    return RichResult(
        title="Trans-Gaussian kriging",
        summary_lines=[("prediction", naive + correction),
                       ("uncorrected phi(p_ok)", naive),
                       ("bias correction", correction),
                       ("MSPE", mspe)],
        payload={"prediction": float(naive + correction),
                 "naive_prediction": naive, "correction": float(correction),
                 "mspe": mspe, "kriging_variance": float(var_ok),
                 "lagrange": float(m), "mu_y": mu_y,
                 "method": "trans-Gaussian (ordinary) kriging"},
    )


def cheatsheet():
    return "sptgk: trans-Gaussian kriging, eqs (5.58)-(5.60) (Schabenberger Sec 5.6.2)"
