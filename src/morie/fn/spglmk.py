# SPDX-License-Identifier: AGPL-3.0-or-later
"""Spatial prediction in generalized linear models."""

from . import _array_core as np

from ._richresult import RichResult
from ._schab_glmm import canonical_link, link, predict_glm

__all__ = ["schabenberger_glm_kriging"]


def schabenberger_glm_kriging(nu, Sigma_nu, sigma_0, X, x0, mu0,
                              link_kind="log", beta=None):
    """Predict a new observation in a spatial GLM, Sec. 6.3.6.

    Prediction happens on the PSEUDO-data scale, where the model is linear
    and universal kriging applies, and is then carried back. The kriging
    predictor of nu(s0) and its mean-squared prediction error come from
    Sec. 5.3.3 eq (5.30); everything specific to the GLM is in the return
    trip.

    WHICH ERROR GOES WITH WHICH PREDICTOR. There are two predictors of the
    original data here and the book is emphatic that their error measures
    are not interchangeable.

    Eq (6.87) applies the inverse link to the pseudo-scale prediction,
    Z_hat(s0) = g^-1(p(nu; nu(s0))). The delta method applied to it gives
    (6.88) -- and the text states outright that (6.88) "is not the
    mean-squared prediction error of the inverse linked predictor (6.87).
    It is the prediction error of a different predictor of the original
    data."

    That other predictor is eq (6.90), got by assuming the new observation
    also satisfies the pseudo-data relation (6.78) and solving for it:

        Z_hat(s0) = mu_hat(s0) + (g'(mu_hat(s0)))^-1 (nu_hat(s0) - g(mu_hat(s0)))

    with mean-squared prediction error exactly (6.91),

        sigma^2_Z(s0) = (d mu / d eta)^2 sigma^2_nu(s0).

    ``prediction`` and ``mspe`` are the matched pair, (6.90) with (6.91).
    ``inverse_link_prediction`` is (6.87), returned separately and without
    an error measure attached, because the one that looks like it belongs
    to it does not. Gotway and Wolfinger (2003).

    Parameters
    ----------
    nu : array-like, shape (n,)
        Pseudo-data at the observed locations, eq (6.78).
    Sigma_nu : array-like, shape (n, n)
        Var[nu] = Sigma_S + Sigma_mu.
    sigma_0 : array-like, shape (n,)
        Covariance between the pseudo-data at s0 and at the observed sites.
    X : array-like, shape (n, k)
    x0 : array-like, shape (k,)
        Covariates at the prediction location.
    mu0 : float
        Current estimate of mu(s0), the expansion locus of eq (6.89).
    link_kind : {"log", "logit", "identity"}
    beta : array-like, optional
        GLS estimate. Computed from the data when omitted.

    Returns
    -------
    RichResult
        Keys: ``prediction``, ``mspe``, ``prediction_error``,
        ``inverse_link_prediction``, ``pseudo_scale_prediction``,
        ``pseudo_scale_mspe``, ``mspe_is_for``.

    References
    ----------
    Schabenberger Ch 6, Sec 6.3.6, eqs (6.87)-(6.91)
    """
    nu = np.asarray(nu, dtype=float).ravel()
    S = np.atleast_2d(np.asarray(Sigma_nu, dtype=float))
    s0 = np.asarray(sigma_0, dtype=float).ravel()
    X = np.atleast_2d(np.asarray(X, dtype=float))
    x0 = np.asarray(x0, dtype=float).ravel()
    n = nu.size
    if S.shape != (n, n) or s0.size != n or X.shape[0] != n:
        raise ValueError("`nu`, `Sigma_nu`, `sigma_0` and `X` must agree on n")
    if x0.size != X.shape[1]:
        raise ValueError("`x0` must have one entry per column of `X`")

    Sinv = np.linalg.inv(S)
    if beta is None:
        xsx = X.T @ Sinv @ X
        beta = np.linalg.solve(xsx, X.T @ Sinv @ nu)
    else:
        beta = np.asarray(beta, dtype=float).ravel()
        xsx = X.T @ Sinv @ X

    # universal kriging on the pseudo-scale, Sec. 5.3.3
    resid = nu - X @ beta
    nu0 = float(x0 @ beta + s0 @ Sinv @ resid)
    c00 = float(np.max(np.diag(S)))          # Var[nu(s0)], same stationary sill
    m = x0 - X.T @ Sinv @ s0                 # unbiasedness correction
    var0 = float(c00 - s0 @ Sinv @ s0 + m @ np.linalg.solve(xsx, m))
    var0 = max(var0, 0.0)

    out = predict_glm(nu0, var0, mu0, link_kind)
    payload = dict(out)
    payload["beta"] = beta
    payload["link"] = link_kind
    payload["pseudo_scale_note"] = (
        "kriging is done on the pseudo-data, where the model is linear; the "
        "GLM enters only on the return trip")
    lines = [("prediction, eq (6.90)", payload["prediction"]),
             ("prediction std error, eq (6.91)", payload["prediction_error"]),
             ("inverse-link predictor, eq (6.87)",
              payload["inverse_link_prediction"]),
             ("pseudo-scale prediction", payload["pseudo_scale_prediction"])]
    return RichResult(title="Spatial prediction in a GLM",
                      summary_lines=lines, payload=payload)


def cheatsheet():
    return ("spglmk: spatial prediction in GLMs (Sec. 6.3.6) -- eq (6.90) with "
            "its own MSPE (6.91), kept apart from the inverse-link (6.87)")

# Names the lazy map still points at from before a rename.
# Without these, morie.fn.<name> raises AttributeError.
schabenberger_spatial_glm_kriging = schabenberger_glm_kriging  # pre-rename spelling, kept live
