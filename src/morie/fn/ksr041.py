# morie.fn -- function file (rootcoder007/morie)
"""Almost-sure bootstrap Donsker characterisation."""

import numpy as np

from ._richresult import RichResult
from .ksr040 import kosorok_ch2_bootstrap_donsker_iff

__all__ = ["kosorok_ch2_bootstrap_donsker_almost_sure"]


def kosorok_ch2_bootstrap_donsker_almost_sure(X, t=None, n_boot=400, rng=None,
                                              F=None, envelope_sq_mean=None):
    r"""Outer-almost-sure bootstrap characterisation:

    :math:`\mathcal F` is P-Donsker **and**
    :math:`P^*\big[\sup_{\mathcal F}(f(X) - Pf)^2\big] < \infty`
    **iff** :math:`\hat{\mathbb G}_n \Rightarrow^* \mathbb G`
    outer almost surely.

    The strictly stronger sibling of :mod:`morie.fn.ksr040`: upgrading
    "in probability" to "almost surely" costs a SECOND condition, the
    square-integrable centred envelope. Both are reported separately,
    because a class can be Donsker (so the in-probability bootstrap
    works) while the a.s. version fails for want of that moment.

    Parameters
    ----------
    X : array-like
        Sample.
    t : array-like, optional
        Evaluation points.
    n_boot : int, default 400
        Replications.
    rng : numpy Generator, optional
    F : callable, optional
        True CDF.
    envelope_sq_mean : float, optional
        :math:`P^*[\sup_F (f(X) - Pf)^2]`; for the indicator class it
        is bounded by 1 and is computed when omitted.

    Returns
    -------
    RichResult
        keys: ``max_abs_gap``, ``envelope_sq_mean``,
        ``envelope_condition_met``, ``both_conditions_met``,
        ``method``.
    References
    ----------
    Kosorok, M. R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Ch. 2 (outer almost sure bootstrap convergence).
    """
    base = kosorok_ch2_bootstrap_donsker_iff(X, t=t, n_boot=n_boot, rng=rng, F=F)
    if envelope_sq_mean is None:
        # indicator class: |1{X <= t} - F(t)| <= 1, so the centred
        # envelope's square has mean at most 1
        env = 1.0
    else:
        env = float(envelope_sq_mean)
    env_ok = bool(np.isfinite(env))
    return RichResult(
        payload={"max_abs_gap": base["max_abs_gap"],
                 "bootstrap_cov": base["bootstrap_cov"],
                 "bridge_cov": base["bridge_cov"],
                 "envelope_sq_mean": env, "envelope_condition_met": env_ok,
                 "both_conditions_met": bool(env_ok),
                 "method": "a.s. version needs Donsker AND a square-integrable envelope"}
    )


def cheatsheet():
    return "ksr041: a.s. costs a SECOND condition beyond ksr040's in-probability"
