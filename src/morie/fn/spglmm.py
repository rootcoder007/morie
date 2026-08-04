# SPDX-License-Identifier: AGPL-3.0-or-later
"""Spatial generalized linear mixed model: the conditional specification."""

from . import _array_core as np

from ._richresult import RichResult
from ._schab_glmm import (canonical_link, conditional_mean,
                          conditional_variance, marginal_moments_lognormal,
                          naive_marginal_mean)

__all__ = ["schabenberger_glmm_conditional"]


def schabenberger_glmm_conditional(X, beta, S, sigma2=1.0, family="poisson",
                                   link_kind=None, correlation=None):
    """The conditional specification of a spatial GLMM, Sec. 6.3.4.

    Data are taken to be conditionally dependent on an underlying smooth
    spatial process {S(s)}. Given S(s), Z(s) is in the exponential family,
    and the link relates the CONDITIONAL mean to the covariates and to the
    latent field, eq (6.73),

        g[mu(s)] = x(s)' beta + S(s),

    with the conditional variance depending on the mean, eq (6.74),

        Var[Z(s)|S] = sigma^2 v(mu(s)).

    S(s) enters as an addition to the intercept -- a random intercept that
    varies with location. Spatial dependence is deferred entirely to
    {S(s)}: the data are conditionally independent given S.

    THE TRAP THIS FUNCTION EXISTS TO MAKE VISIBLE. In a linear model the
    marginal and conditional specifications give the same inference. In a
    GLM they do not, and the text is explicit about why:

        E[Z(s)] = E_S[g^-1(x(s)'beta + S(s))] != g^-1(x(s)'beta)

    "Taking expectations is a linear operation and does not carry through in
    case of a nonlinear link function." Evaluating the inverse link at the
    linear predictor does NOT give the marginal mean.

    So both are returned. ``conditional_mean`` is mu(s) given the realised
    S; ``naive_marginal_mean`` is the quantity people reach for by mistake;
    and for the log link, where Example 6.6 derives the correction in closed
    form, ``marginal_mean`` and ``marginal_variance`` give the right answer
    along with the ratio between them. Under a log link that ratio is
    exp{sigma_S^2 / 2}, so the error grows with the variance of the latent
    field rather than staying negligible.

    Example 6.6 also shows Var[Z(s)] > E[Z(s)] even when sigma^2 = 1: the
    latent field induces overdispersion AND autocorrelation, both of which
    depend on the mean, so the conditional model can be used with
    non-stationary spatial processes.

    Parameters
    ----------
    X : array-like, shape (n, k)
        Design matrix.
    beta : array-like, shape (k,)
    S : array-like, shape (n,)
        Realised values of the latent spatial field.
    sigma2 : float
        Dispersion parameter in eq (6.74).
    family : {"poisson", "binomial", "gaussian"}
    link_kind : {"log", "logit", "identity"}, optional
        Defaults to the canonical link of ``family``.
    correlation : array-like, optional
        Corr[S(s_i), S(s_j)]. When given with a log link, the marginal
        covariance of Example 6.6 is returned too.

    Returns
    -------
    RichResult
        Keys: ``conditional_mean``, ``conditional_variance``,
        ``naive_marginal_mean``, and for the log link ``marginal_mean``,
        ``marginal_variance``, ``marginal_ratio`` and optionally
        ``marginal_covariance``.

    References
    ----------
    Schabenberger Ch 6, Sec 6.3.4, eqs (6.73)-(6.74) and Example 6.6
    """
    if link_kind is None:
        link_kind = canonical_link(family)
    mu = conditional_mean(X, beta, S, link_kind=link_kind)
    payload = {
        "conditional_mean": mu,
        "conditional_variance": conditional_variance(mu, sigma2, family),
        "naive_marginal_mean": naive_marginal_mean(X, beta, link_kind),
        "family": family, "link": link_kind, "sigma2": float(sigma2),
    }
    lines = [("family", family), ("link", link_kind)]

    if link_kind == "log":
        sigma2_S = float(np.var(np.asarray(S, dtype=float).ravel(), ddof=0))
        mom = marginal_moments_lognormal(X, beta, sigma2_S, sigma2=sigma2,
                                         rho=correlation)
        payload["marginal_mean"] = mom["mean"]
        payload["marginal_variance"] = mom["variance"]
        payload["sigma2_S"] = sigma2_S
        payload["marginal_ratio"] = float(np.exp(sigma2_S / 2.0))
        if correlation is not None:
            payload["marginal_covariance"] = mom["covariance"]
        payload["marginal_note"] = (
            "E[Z(s)] is NOT g^-1(x(s)'beta): under the log link the marginal "
            f"mean exceeds the naive value by exp(sigma_S^2/2) = "
            f"{payload['marginal_ratio']:.4f}")
        lines += [("sigma_S^2", sigma2_S),
                  ("marginal / naive mean", payload["marginal_ratio"])]
    else:
        payload["marginal_note"] = (
            "the marginal mean is E_S[g^-1(x'beta + S)], which has no closed "
            "form for this link; g^-1(x'beta) is NOT it")

    return RichResult(title="Spatial GLMM, conditional specification",
                      summary_lines=lines, payload=payload)


def cheatsheet():
    return ("spglmm: conditional specification of a spatial GLMM (Sec. 6.3.4) "
            "-- eq (6.73) with the Example 6.6 marginal correction")

# Names the lazy map still points at from before a rename.
# Without these, morie.fn.<name> raises AttributeError.
schabenberger_spatial_glmm = schabenberger_glmm_conditional  # pre-rename spelling, kept live
