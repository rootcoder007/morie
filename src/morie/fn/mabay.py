# morie.fn -- k02 batch (rootcoder007/morie)
"""Conjugate normal-normal random-effects meta-analysis (empirical Bayes).

Source consulted: Higgins, J.P.T., Thompson, S.G. and Spiegelhalter, D.J.
(2009), A re-evaluation of random-effects meta-analysis, *JRSS Series A*
172(1), 137-159, section 2.  With y_i ~ N(theta_i, v_i) and
theta_i ~ N(mu, tau^2), a flat prior on mu gives the closed-form posterior

    mu | y  ~ N( sum w*_i y_i / sum w*_i ,  1 / sum w*_i ),  w*_i = 1/(v_i + tau^2)

and the study-level posteriors are the classical shrinkage estimates

    theta_i | y ~ N( (y_i/v_i + mu/tau^2) / (1/v_i + 1/tau^2), 1/(1/v_i + 1/tau^2) )

tau^2 is plugged in at its DerSimonian-Laird moment estimate, which is the
empirical-Bayes reading of the paper's hierarchical model; the fully Bayesian
version integrates over tau^2 and is not attempted here (it has no closed
form).  ``shrinkage`` reports 1 - Var(theta_i | y)/v_i.
"""

from __future__ import annotations

from . import _array_core as np
from .k02util import k02dl, k02z

from ._richresult import RichResult

__all__ = ["ma_bayes_random_effects"]


def ma_bayes_random_effects(yi, vi, tau2=None, level=0.95):
    """Empirical-Bayes posterior for the pooled effect and each study.

    Parameters
    ----------
    yi, vi : array-like
        Study effects and their within-study variances.
    tau2 : float, optional
        Between-study variance; DerSimonian-Laird if not supplied.
    level : float, default 0.95
        Credible-interval level.

    Returns
    -------
    RichResult
        estimate, se, ci_lower, ci_upper, tau2, theta_mean, theta_sd,
        shrinkage, n, method.
    """
    y = np.asarray(yi, dtype=float).ravel()
    v = np.asarray(vi, dtype=float).ravel()
    t2 = k02dl(y, v)[0] if tau2 is None else float(tau2)
    ws = 1.0 / (v + t2)
    sws = float(np.sum(ws))
    mu = float(np.sum(ws * y)) / sws
    se = float(np.sqrt(1.0 / sws))
    crit = k02z(0.5 + 0.5 * float(level))
    if t2 > 0.0:
        prec = 1.0 / v + 1.0 / t2
        th = (y / v + mu / t2) / prec
        sd = np.sqrt(1.0 / prec)
    else:
        th = np.full(len(y), mu)
        sd = np.zeros(len(y))
    return RichResult(
        payload={
            "estimate": mu,
            "se": se,
            "ci_lower": float(mu - crit * se),
            "ci_upper": float(mu + crit * se),
            "tau2": float(t2),
            "theta_mean": th.tolist(),
            "theta_sd": sd.tolist(),
            "shrinkage": (1.0 - sd**2 / v).tolist(),
            "n": int(len(y)),
            "method": "Conjugate normal-normal random-effects posterior (Higgins, Thompson & Spiegelhalter 2009, sec. 2)",
        }
    )


# CANONICAL TEST
# >>> y = [0.10, 0.30, -0.20, 0.45, 0.05, 0.22]
# >>> v = [0.02, 0.05, 0.03, 0.08, 0.01, 0.04]
# >>> r = ma_bayes_random_effects(y, v)
# >>> assert abs(r["estimate"] - 0.0920094772579361) < 1e-13   # = DL pooled
# >>> assert all(0.0 < s < 1.0 for s in r["shrinkage"])


def cheatsheet():
    return "mabay(yi, vi): empirical-Bayes normal-normal random-effects posterior."


mabayesrandomeffects = ma_bayes_random_effects
