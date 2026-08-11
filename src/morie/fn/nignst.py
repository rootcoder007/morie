# SPDX-License-Identifier: AGPL-3.0-or-later
"""Conjugate normal model: N-Inv-chi2 posterior update (BDA3 3.3)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["nignst", "normal_invgamma_update"]


def nignst(y, mu0, kappa0, nu0, sigma0_sq):
    """
    Exact conjugate update for a normal sample with unknown mean and
    variance under the normal-inverse-chi-square prior
    N-Inv-chi2(mu0, sigma0^2/kappa0; nu0, sigma0^2), i.e.
    sigma^2 ~ Inv-chi2(nu0, sigma0^2) and
    mu | sigma^2 ~ N(mu0, sigma^2/kappa0). This prior is the
    (mu, sigma^2) parameterization of the normal-inverse-gamma family
    with shape nu0/2 and scale nu0 sigma0^2 / 2.

    The posterior is N-Inv-chi2(mu_n, sigma_n^2/kappa_n; nu_n,
    sigma_n^2) with (Gelman et al., BDA3, Section 3.3, Eq. 3.8):

        mu_n         = (kappa0 mu0 + n ybar) / (kappa0 + n)
        kappa_n      = kappa0 + n
        nu_n         = nu0 + n
        nu_n sigma_n^2 = nu0 sigma0^2 + (n - 1) s^2
                         + kappa0 n (ybar - mu0)^2 / (kappa0 + n)

    The marginal posterior of mu is t_nu_n(mu_n, sigma_n^2/kappa_n)
    and the posterior predictive for one new observation is
    t_nu_n(mu_n, sigma_n^2 (kappa_n + 1)/kappa_n) (BDA3, Section 3.3).

    Parameters
    ----------
    y : array-like
        Observations.
    mu0 : float
        Prior location of mu.
    kappa0 : float
        Prior pseudo-count for the mean.
    nu0 : float
        Prior degrees of freedom for sigma^2.
    sigma0_sq : float
        Prior scale of sigma^2.

    Returns
    -------
    result : RichResult
        Keys: mu_n, kappa_n, nu_n, sigma_n_sq, mu_scale_sq (posterior
        t scale^2 of mu), pred_scale_sq (posterior predictive t
        scale^2), estimate (mu_n), n, ybar, s_sq.

    References
    ----------
    Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari,
    A. and Rubin, D. B. (2013), Bayesian Data Analysis, 3rd ed.,
    Chapman and Hall/CRC, Section 3.3, Eqs. 3.7-3.8 (posterior
    parameters), marginal-t and predictive-t statements. Local copy:
    fetched-wave3/gelman-etal-2013-bayesian-data-analysis-3ed.pdf
    """
    yv = np.asarray(y, dtype=float)
    n = len(yv)
    if n < 1:
        raise ValueError("need at least one observation")
    kappa0 = float(kappa0)
    nu0 = float(nu0)
    sigma0_sq = float(sigma0_sq)
    mu0 = float(mu0)
    if kappa0 <= 0.0 or nu0 <= 0.0 or sigma0_sq <= 0.0:
        raise ValueError("kappa0, nu0, sigma0_sq must be positive")
    ybar = float(np.mean(yv))
    s_sq = float(np.var(yv, ddof=1)) if n > 1 else 0.0
    kappa_n = kappa0 + n
    nu_n = nu0 + n
    mu_n = (kappa0 * mu0 + n * ybar) / kappa_n
    nusq = (nu0 * sigma0_sq + (n - 1) * s_sq
            + kappa0 * n * (ybar - mu0) ** 2 / kappa_n)
    sigma_n_sq = nusq / nu_n
    return RichResult(payload={
        "estimate": mu_n,
        "mu_n": mu_n, "kappa_n": kappa_n, "nu_n": nu_n,
        "sigma_n_sq": sigma_n_sq,
        "mu_scale_sq": sigma_n_sq / kappa_n,
        "pred_scale_sq": sigma_n_sq * (kappa_n + 1.0) / kappa_n,
        "n": n, "ybar": ybar, "s_sq": s_sq,
        "method": "BDA3 Section 3.3 N-Inv-chi2 conjugate update",
    })


normal_invgamma_update = nignst


def cheatsheet():
    return "nignst(y, mu0, kappa0, nu0, sigma0_sq) -> exact N-Inv-chi2 posterior parameters."
