# morie.fn -- function file (rootcoder007/morie)
"""Generalized gamma AFT model."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult
from ._surv import prepare

__all__ = ["generalized_gamma_aft"]


def generalized_gamma_aft(time, event, X, max_iter=500, tol=1e-6):
    r"""Fit the three-parameter generalized gamma, which nests its competitors.

    The generalized gamma adds a shape parameter :math:`q` on top of the AFT
    location and scale, and the families used elsewhere in this module are all
    special cases of it:

    ==================  ==========================
    :math:`q = 1`       Weibull
    :math:`q = 0`       log-normal
    :math:`q = \sigma`  gamma
    :math:`q = 1, \sigma = 1`  exponential
    ==================  ==========================

    That nesting is the point. Because the alternatives are special cases, a
    **likelihood-ratio test** against the fitted :math:`q` is a principled way
    to choose among them, instead of comparing non-nested AICs and hoping. The
    test statistic against Weibull is :math:`2(\ell_{gg} - \ell_{weib})` on one
    degree of freedom, and it is returned.

    The price is that :math:`q` is weakly identified: it is estimated from the
    tail behaviour, so it needs a lot of events, and with few of them the fit
    will happily wander. ``se_q`` and the convergence flag are the things to
    look at before trusting a family choice made this way.

    Parameters
    ----------
    time, event, X : array-like
        Survival data. Times must be strictly positive.
    max_iter, tol
        Optimiser controls.

    Returns
    -------
    RichResult
        ``beta``, ``sigma``, ``q``, ``loglik``, ``lr_vs_weibull``,
        ``p_vs_weibull``, ``lr_vs_lognormal``, ``preferred``.

    References
    ----------
    Prentice, R. L. (1974). A log gamma model and its maximum likelihood
        estimation. *Biometrika*, 61(3), 539-544.
    Kalbfleisch, J. D., & Prentice, R. L. (2002). *The Statistical Analysis
        of Failure Time Data* (2nd ed.). Wiley.

    Examples
    --------
    On Weibull data the fitted shape sits near 1 and the likelihood-ratio test
    against Weibull does not reject.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(1200, 2))
    >>> mu = 1.0 + 0.7 * X[:, 0] - 0.4 * X[:, 1]
    >>> T = np.exp(mu + 0.6 * np.log(rng.exponential(1.0, 1200)))
    >>> C = rng.exponential(float(np.exp(mu).mean()) * 6, 1200)
    >>> t, e = np.minimum(T, C), (T <= C).astype(float)
    >>> r = generalized_gamma_aft(t, e, X)
    >>> bool(r["p_vs_weibull"] > 0.05)
    True
    >>> str(r["preferred"])
    'weibull'

    Regression coefficients are still recovered.

    >>> [bool(abs(r["beta"][i] - v) < 0.2) for i, v in enumerate([1.0, 0.7, -0.4])]
    [True, True, True]

    On log-normal data it prefers log-normal instead -- the nesting doing its
    job.

    >>> T2 = np.exp(mu + 0.6 * rng.normal(size=1200))
    >>> t2, e2 = np.minimum(T2, C), (T2 <= C).astype(float)
    >>> bool(generalized_gamma_aft(t2, e2, X)["lr_vs_lognormal"]
    ...      < generalized_gamma_aft(t2, e2, X)["lr_vs_weibull"])
    True
    """
    from scipy.optimize import minimize
    from scipy.special import gammaln
    from scipy.stats import chi2, gamma as gamma_dist

    t, e, Xm = prepare(time, event, X)
    if np.any(t <= 0):
        raise ValueError("AFT models need strictly positive times")
    A = np.column_stack([np.ones(t.size), Xm])
    p = A.shape[1]
    logt = np.log(t)

    def nll(theta):
        b, ls, q = theta[:p], theta[p], theta[p + 1]
        sigma = np.exp(np.clip(ls, -20, 20))
        z = (logt - A @ b) / sigma
        if abs(q) < 1e-6:                       # log-normal limit
            from scipy.stats import norm

            ld, lsv = norm.logpdf(z), norm.logsf(z)
        else:
            qi = 1.0 / q**2
            w = q * z
            u = qi * np.exp(np.clip(w, -500, 500))
            ld = (np.log(abs(q)) - gammaln(qi) + qi * np.log(qi)
                  + qi * w - u)
            sf = gamma_dist.sf(u, qi) if q > 0 else gamma_dist.cdf(u, qi)
            lsv = np.log(np.maximum(sf, 1e-300))
        val = -float(np.sum(np.where(e > 0, ld - np.log(sigma), lsv)))
        return val if np.isfinite(val) else 1e12

    start = np.r_[np.linalg.lstsq(A, logt, rcond=None)[0], 0.0, 1.0]
    res = minimize(nll, start, method="Nelder-Mead",
                   options={"maxiter": max_iter * 20, "xatol": tol, "fatol": tol})
    beta, log_sigma, q = res.x[:p], float(res.x[p]), float(res.x[p + 1])
    ll = float(-res.fun)

    # A likelihood-ratio test needs the restricted model RE-MAXIMISED with q
    # fixed, not merely evaluated at the unrestricted estimates. Evaluating at
    # the unrestricted point understates the restricted likelihood, inflates
    # the statistic and makes the test reject a correct family.
    def _restricted(q_fixed):
        r = minimize(lambda th: nll(np.r_[th, q_fixed]),
                     np.r_[beta, log_sigma], method="Nelder-Mead",
                     options={"maxiter": max_iter * 20, "xatol": tol, "fatol": tol})
        return float(-r.fun)

    ll_w = _restricted(1.0)
    ll_ln = _restricted(0.0)
    lr_w = float(max(2.0 * (ll - ll_w), 0.0))
    lr_ln = float(max(2.0 * (ll - ll_ln), 0.0))
    p_w = float(chi2.sf(lr_w, 1))
    p_ln = float(chi2.sf(lr_ln, 1))
    preferred = ("generalized gamma" if (p_w < 0.05 and p_ln < 0.05)
                 else "weibull" if lr_w <= lr_ln else "lognormal")
    return RichResult(
        title="Generalized gamma AFT",
        summary_lines=[("n", int(t.size)), ("events", int(e.sum())),
                       ("q", q), ("loglik", ll), ("preferred", preferred)],
        warnings=(["q is weakly identified and is estimated from tail "
                   "behaviour; with few events a family choice made this way "
                   "is not reliable"]
                  + ([] if res.success else ["the optimiser did not converge"])),
        payload={
            "beta": beta, "sigma": float(np.exp(log_sigma)),
            "log_sigma": log_sigma, "q": q, "loglik": ll,
            "lr_vs_weibull": lr_w, "p_vs_weibull": p_w,
            "lr_vs_lognormal": lr_ln, "p_vs_lognormal": p_ln,
            "preferred": preferred, "aic": float(2 * (p + 2) - 2 * ll),
            "n": int(t.size), "converged": bool(res.success),
            "method": "generalized_gamma_aft",
        },
    )


def cheatsheet():
    return "ggmaft: nests Weibull (q=1) and lognormal (q=0), so family choice is an LR test -- but q needs many events"
