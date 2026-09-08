# morie.fn -- function file (rootcoder007/morie)
"""Renewal-equation reproduction number with reporting delay."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['rtrenew', 'epinow2']


def rtrenew(incidence, gen_int, delays=None):
    """Renewal-equation reproduction number with reporting delay.

    EpiNow2's generative model is a renewal equation for infections plus a convolution D_t = xi sum_tau xi(tau) I_{t-tau} mapping infections to reports. This function inverts the first exactly and the second approximately: the delay is undone by a deterministic integer back-shift of the rounded mean delay rather than by the package's Bayesian deconvolution, which is stated here rather than hidden because it changes the answer near the series end.


    Formula: I_t = R_t sum_{tau=1}^{gmax} g(tau) I_{t-tau}, so R_t = I_t / sum_tau g(tau) I_{t-tau}

    Parameters
    ----------
    incidence : array-like
        Reported cases per time step.
    gen_int : array-like
        Discretised generation-time pmf over lags 1..gmax.
    delays : array-like, optional
        Reporting-delay pmf over lags 0..xi_max.

    Returns
    -------
    RichResult
        ``rt``, ``time``, ``infections``, ``shift``, ``mean_rt``, ``n``.

    References
    ----------
    Abbott, Hellewell, Sherratt et al (2020), EpiNow2.  Model definition
    verified against the package's own estimate_infections() vignette,
    which states I_t = R_t sum_tau g(tau) I_{t-tau} and the delay
    convolution D_t = xi sum_tau xi(tau) I_{t-tau}.
    """
    y = C.vec(incidence)
    w = C.vec(gen_int)
    if sum(w) <= 0:
        raise ValueError("gen_int must have positive mass")
    w = [v / sum(w) for v in w]
    shift = 0
    if delays is not None:
        d = C.vec(delays)
        if sum(d) <= 0:
            raise ValueError("delays must have positive mass")
        d = [v / sum(d) for v in d]
        shift = int(round(sum(i * p for i, p in enumerate(d))))
    infections = y[shift:] if shift else list(y)
    n = len(infections); s = len(w)
    times, rt = [], []
    for t in range(s, n):
        force = sum(w[k] * infections[t - k - 1] for k in range(s))
        times.append(t)
        rt.append(infections[t] / force if force > 0 else float("nan"))
    good = [v for v in rt if v == v]
    return RichResult(payload={
        "rt": rt, "time": times, "infections": infections, "shift": shift,
        "mean_rt": (sum(good) / len(good)) if good else float("nan"), "n": n,
        "method": "Renewal-equation Rt with reporting delay"})


epinow2 = rtrenew


def cheatsheet():
    return "epimet: Renewal-equation reproduction number with reporting delay."
