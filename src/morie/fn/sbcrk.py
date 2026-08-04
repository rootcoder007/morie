# morie.fn -- function file (rootcoder007/morie)
"""Simulation-based calibration ranks."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["sbcrank", "simulation_based_calibration_rank"]


def sbcrank(prior_draw, post_draws, bins=None):
    """Rank statistics for simulation-based calibration, and a uniformity test.

    The rank of the prior draw among the posterior draws is uniform on
    {0, ..., L} if and only if the sampler is correct, and the SHAPE of
    the departure names the fault: a U shape means the posterior is too
    narrow, a hump means too wide, a slope means biased.  So the
    histogram is returned, not just the p-value.

    The chi-square test needs the bin count to divide L + 1 exactly, or
    the bins have unequal expected counts and the test is wrong; that
    is checked rather than assumed.

    Formula: rank_j = #{ l : theta^{(l)}_j < theta_j^prior };
             under calibration rank ~ Uniform{0, ..., L};
             chi^2 = sum_b (O_b - E_b)^2 / E_b on (bins - 1) df

    Parameters
    ----------
    prior_draw : array-like
        One prior draw per replication, length J.
    post_draws : array-like, shape (J, L)
        Posterior draws for each replication.
    bins : int, optional
        Number of histogram bins; must divide L + 1 (default: L + 1).

    Returns
    -------
    RichResult
        ``rank``, ``histogram``, ``expected``, ``statistic``,
        ``p_value``, ``df``, ``bins``, ``J``, ``L``.

    References
    ----------
    Talts, Betancourt, Simpson, Vehtari & Gelman (2018), Validating
    Bayesian inference algorithms with simulation-based calibration,
    arXiv:1804.06788 -- the primary source for the rank statistic and
    for reading its histogram shape.  Gelman, Carlin, Stern, Dunson,
    Vehtari & Rubin (2013), Bayesian Data Analysis, 3rd edition, was
    fetched in full and searched; it predates simulation-based
    calibration and does NOT contain it, so it is not cited here.
    """
    pd = C.vec(prior_draw)
    P = C.mat(post_draws)
    J = len(pd)
    if len(P) != J:
        raise ValueError("one row of posterior draws per prior draw")
    L = len(P[0])
    if any(len(r) != L for r in P):
        raise ValueError("every replication needs the same number of draws")
    if J < 1 or L < 1:
        raise ValueError("at least one replication and one draw are needed")
    rk = [sum(1 for v in P[j] if v < pd[j]) for j in range(J)]
    K = L + 1 if bins is None else int(bins)
    if K < 1 or (L + 1) % K != 0:
        raise ValueError("bins must divide L + 1 exactly")
    width = (L + 1) // K
    hist = [0.0] * K
    for r in rk:
        hist[min(K - 1, r // width)] += 1.0
    exp = J / K
    chi = sum((hist[b] - exp) ** 2 / exp for b in range(K))
    df = K - 1
    return RichResult(payload={
        "rank": [float(v) for v in rk], "histogram": hist,
        "expected": exp, "statistic": chi,
        "p_value": 1.0 - C.pchisq(chi, df) if df >= 1 else float("nan"),
        "df": float(df), "bins": float(K), "J": float(J), "L": float(L),
        "method": "Simulation-based calibration ranks (Talts et al. 2018)"})


simulation_based_calibration_rank = sbcrank


def cheatsheet():
    return "sbcrk: rank of prior draw among L posterior draws; uniform if calibrated"
