# SPDX-License-Identifier: AGPL-3.0-or-later
"""Penalized quasi-likelihood / pseudo-likelihood for spatial GLMMs."""

from . import _array_core as np

from ._richresult import RichResult
from ._schab_glmm import (canonical_link, fit_pseudo_likelihood, pql_score,
                          reml_objective)

__all__ = ["schabenberger_pql"]


def schabenberger_pql(z, X, Sigma_S, family="poisson", link_kind=None,
                      sigma2=1.0, R=None, max_iter=100, tol=1e-8,
                      check_score=True):
    """Fit a spatial GLMM by pseudo-likelihood, Sec. 6.3.5.

    The problem is linearised with a first-order Taylor expansion of the
    link, giving the pseudo-data of eq (6.78),

        nu_i = g(mu_i) + g'(mu_i)(Z(s_i) - mu_i),

    whose marginal moments are E[nu] = X beta and
    Var[nu] = Sigma_S + Sigma_mu. That is a linear mixed model, so the
    machinery of Sec. 5.5.2-5.5.3 applies and the estimates follow from
    (6.80)-(6.82), with S predicted by (6.81). Because Sigma_mu depends on
    beta the whole thing is relinearised and repeated -- the six-step
    algorithm the text sets out, which this follows exactly.

    Wolfinger and O'Connell (1993) call this pseudo-likelihood; Breslow and
    Clayton (1993) reach the same place by a Laplace approximation with
    Fisher scoring and call it penalized quasi-likelihood. The text settles
    the relationship: the two objective functions "differ ... only by a
    constant amount. The two approaches will thus yield the same
    estimates." With ``check_score=True`` the PQL score equations are
    evaluated at the fitted values and returned, so the equivalence is
    checked on every fit rather than taken on trust.

    MARGINAL VERSUS CONDITIONAL. The text is explicit that there is a
    choice about where the spatial dependence lives. A marginal model puts
    it in R, a spatial correlation matrix, and sets S = 0; a conditional
    model puts it in Sigma_S and takes R = I. Passing ``R`` moves this
    function from the second case to the first.

    ONE CORRECTION TO THE PRINTED SCORE EQUATIONS. Sec. 6.3.5.3 writes them
    with Sigma_mu, but Sigma_mu was defined at (6.79) as the covariance of
    the PSEUDO-data, which carries Psi^-1 on both sides. The scores need the
    data-scale covariance. Taken literally they are wrong by a factor of
    Psi^2 and do not vanish at the solution; see `_schab_glmm.pql_score`.

    Parameters
    ----------
    z : array-like, shape (n,)
        Observed responses.
    X : array-like, shape (n, k)
    Sigma_S : array-like, shape (n, n)
        Covariance of the latent spatial field.
    family : {"poisson", "binomial", "gaussian"}
    link_kind : str, optional
        Defaults to the canonical link.
    sigma2 : float
        Dispersion parameter.
    R : array-like, optional
        Working correlation of the conditional part. Default I, the
        conditional model; supply a spatial correlation for a marginal one.
    max_iter, tol : int, float
        Outer relinearisation controls.
    check_score : bool
        Evaluate the PQL score equations at the solution.

    Returns
    -------
    RichResult
        Keys: ``beta``, ``se_beta``, ``cov_beta``, ``S``, ``mu``,
        ``sigma2``, ``pseudo_data``, ``Sigma_nu``, ``reml``, ``n_iter``,
        ``converged``, and when checked ``score_beta_max`` and
        ``score_S_max``.

    References
    ----------
    Schabenberger Ch 6, Sec 6.3.5, eqs (6.78)-(6.85)
    """
    if link_kind is None:
        link_kind = canonical_link(family)
    fit = fit_pseudo_likelihood(z, X, Sigma_S, family=family,
                                link_kind=link_kind, sigma2=sigma2, R=R,
                                max_iter=max_iter, tol=tol)
    payload = dict(fit)
    payload["reml"] = reml_objective(X, fit["Sigma_nu"], fit["pseudo_data"])
    payload["specification"] = "marginal" if R is not None else "conditional"
    lines = [("family", family), ("link", link_kind),
             ("specification", payload["specification"]),
             ("iterations", fit["n_iter"]), ("converged", fit["converged"]),
             ("-2 restricted log likelihood", payload["reml"])]

    if not fit["converged"]:
        payload["warning"] = (
            f"the doubly iterative scheme did not converge in {max_iter} "
            f"outer steps; the estimates below are wherever it stopped")

    if check_score:
        sc = pql_score(z, X, fit["beta"], fit["S"], Sigma_S, family,
                       link_kind, sigma2=sigma2, R=R)
        payload["score_beta_max"] = float(np.max(np.abs(sc["score_beta"])))
        payload["score_S_max"] = float(np.max(np.abs(sc["score_S"])))
        payload["pql_pl_equivalent"] = bool(
            payload["score_beta_max"] < 1e-6 and payload["score_S_max"] < 1e-6)
        lines.append(("PQL score vanishes (PQL = PL)",
                      payload["pql_pl_equivalent"]))
        if not payload["pql_pl_equivalent"]:
            payload["score_warning"] = (
                "the PQL score equations do not vanish at the "
                "pseudo-likelihood solution, so the two are not agreeing "
                "here as Sec. 6.3.5.3 says they should -- treat the fit as "
                "unconverged")

    return RichResult(title="Spatial GLMM by pseudo-likelihood (PQL)",
                      summary_lines=lines, payload=payload)


def cheatsheet():
    return ("sppql: pseudo-likelihood / penalized quasi-likelihood for spatial "
            "GLMMs (Sec. 6.3.5) -- eqs (6.78)-(6.85), six-step algorithm")

# Names the lazy map still points at from before a rename.
# Without these, morie.fn.<name> raises AttributeError.
schabenberger_pql_glmm = schabenberger_pql  # pre-rename spelling, kept live
