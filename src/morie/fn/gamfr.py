# morie.fn -- function file (rootcoder007/morie)
"""Gamma-frailty Cox model, alternative front-end."""

from __future__ import annotations

from ._richresult import RichResult
from .coxfrl import cox_frailty

__all__ = ["gamma_frailty_cox"]


def gamma_frailty_cox(time, event, X, cluster, **kwargs):
    r"""Cox model with gamma-distributed shared frailty.

    Same estimator as :func:`~morie.fn.coxfrl.cox_frailty`, exposed under the
    name the frailty literature uses. The gamma is the standard choice because
    it is conjugate to the Poisson-like structure of the counting process, so
    the frailty update has a closed form and the model is tractable; the
    log-normal alternative needs numerical integration and buys little.

    The gamma frailty also has a closed-form marginal: integrating out
    :math:`w` gives a survivor function
    :math:`S(t) = \left(1 + \theta \Lambda(t)\right)^{-1/\theta}`, which is a
    Burr distribution -- so the *population* hazard ratio attenuates toward 1
    over time even though the *conditional* one is constant. That attenuation
    is a property of frailty models, not a modelling error, and it is why a
    marginal and a conditional hazard ratio need never agree.

    Parameters
    ----------
    time, event, X, cluster : array-like
        As for :func:`~morie.fn.coxfrl.cox_frailty`.
    **kwargs
        Passed through.

    Returns
    -------
    RichResult
        ``beta``, ``se``, ``theta``, ``kendall_tau``, ``frailty``,
        ``marginal_attenuation``.

    References
    ----------
    Hougaard, P. (2000). *Analysis of Multivariate Survival Data*. Springer.
    Therneau, T. M., & Grambsch, P. M. (2000). *Modeling Survival Data:
        Extending the Cox Model*. Springer.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> k = np.repeat(np.arange(50), 8)
    >>> w = rng.gamma(2.0, 0.5, 50)[k]
    >>> X = rng.normal(size=(400, 1))
    >>> T = rng.exponential(1 / (w * np.exp(X[:, 0] * 0.9)))
    >>> C = rng.exponential(2.0, 400)
    >>> t, e = np.minimum(T, C), (T <= C).astype(float)
    >>> r = gamma_frailty_cox(t, e, X, k)
    >>> bool(r["theta"] > 0.05)
    True

    Kendall's tau within a cluster follows theta/(theta+2), a bounded and
    interpretable scale.

    >>> bool(0.0 <= r["kendall_tau"] < 1.0)
    True
    """
    r = cox_frailty(time, event, X, cluster, **kwargs)
    th = r["theta"]
    return RichResult(
        title="Gamma-frailty Cox model",
        summary_lines=[("clusters", int(r["n_clusters"])), ("theta", float(th)),
                       ("Kendall tau", float(r["kendall_tau"]))],
        warnings=list(r.warnings),
        payload={
            "beta": r["beta"], "se": r["se"], "z": r["z"],
            "p_value": r["p_value"], "hazard_ratio": r["hazard_ratio"],
            "theta": th, "kendall_tau": r["kendall_tau"],
            "frailty": r["frailty"], "clusters": r["clusters"],
            "n_clusters": r["n_clusters"], "loglik": r["loglik"],
            # Marginal HRs attenuate toward 1 over time under gamma frailty.
            "marginal_attenuation": True,
            "converged": r["converged"], "method": "gamma_frailty_cox",
        },
    )


def cheatsheet():
    return "gamfr: gamma is conjugate so the frailty update is closed-form; marginal HR attenuates toward 1"
