# morie.fn -- function file (rootcoder007/morie)
"""One-step estimator from a root-n consistent starting value."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["onestep", "kosorok_one_step_estimator"]


def onestep(x, theta0, kind="huber", k=1.345):
    """A single Newton step on the estimating equation from theta0.

    The point of the one-step estimator is that ONE step is enough: if
    theta0 is root-n consistent then the update is asymptotically
    equivalent to the full Z-estimator, so iterating to convergence
    buys nothing asymptotically and can cost stability.  The step is
    therefore taken exactly once, not looped.

    ``derivative`` is the denominator; when it is near zero the step is
    unstable and the estimator should not be trusted, so it is returned
    rather than hidden.

    Formula: theta_1 = theta_0 + (P_n psi_{theta_0}) / ( -P_n psi'_{theta_0} ),
             psi'(u) = -1 (mean), -1{|u| <= k} (Huber)

    Parameters
    ----------
    x : array-like
        The sample.
    theta0 : float
        Starting value, assumed root-n consistent.
    kind : {"mean", "huber"}
        Which estimating function.
    k : float
        Huber tuning constant.

    Returns
    -------
    RichResult
        ``estimate``, ``theta0``, ``step``, ``psi_mean``,
        ``derivative``, ``n_used`` (points inside the Huber window),
        ``n``.

    References
    ----------
    Kosorok (2008), Introduction to Empirical Processes and
    Semiparametric Inference, Section 2.2.5, for the Z-estimator
    framework Psi_n(theta) = P_n psi_theta this step is taken on;
    the full text of the book was fetched and searched, and the phrase
    "one-step estimator" does NOT appear in it, so the one-step
    construction itself is cited to its own sources: Le Cam (1956), On
    the asymptotic theory of estimation and testing hypotheses,
    Proceedings of the Third Berkeley Symposium 1, 129-156, and Bickel,
    Klaassen, Ritov & Wellner (1993), Efficient and Adaptive Estimation
    for Semiparametric Models, Section 2.5.
    """
    x = C.vec(x)
    n = len(x)
    if n < 1:
        raise ValueError("the sample must be non-empty")
    theta0 = float(theta0)
    kind = str(kind).lower()
    k = float(k)
    if kind == "mean":
        psi = [v - theta0 for v in x]
        dpsi = [-1.0] * n
    elif kind == "huber":
        if k <= 0:
            raise ValueError("the Huber constant k must be positive")
        psi = [max(-k, min(k, v - theta0)) for v in x]
        dpsi = [-1.0 if abs(v - theta0) <= k else 0.0 for v in x]
    else:
        raise ValueError("kind must be 'mean' or 'huber'")
    pm = sum(psi) / n
    dm = sum(dpsi) / n
    if dm == 0.0:
        raise ValueError(
            "the mean derivative is zero; every point is outside the "
            "Huber window and the one-step update is undefined")
    step = pm / (-dm)
    return RichResult(payload={
        "estimate": theta0 + step, "theta0": theta0, "step": step,
        "psi_mean": pm, "derivative": dm,
        "n_used": float(sum(1 for v in dpsi if v != 0.0)), "n": float(n),
        "method": "One-step estimator on the Z-estimating equation"})


kosorok_one_step_estimator = onestep


def cheatsheet():
    return "ksr15: theta_1 = theta_0 + P_n psi / (-P_n psi'), taken ONCE"
