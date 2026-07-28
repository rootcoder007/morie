# morie.fn -- function file (rootcoder007/morie)
"""Adaptive Polya tree prior: alpha_m = m^2 gives near-optimal density estimation rate."""

import numpy as np

from ._richresult import RichResult

__all__ = ["ghosal_pt_adaptive"]


def ghosal_pt_adaptive(x, s=None, n=None, levels=6, a_scale=1.0):
    r"""Rate attained by the adaptive Polya tree prior
    ``a_m = m^2`` (Ghosal Sec. 7.2.3):

    .. math:: \varepsilon_n \asymp n^{-s/(2s+1)}\log n
              \quad\text{for an } s\text{-smooth } p_0 .

    The word doing the work is ADAPTIVE: the same prior attains this
    for every ``s`` in a range WITHOUT being told ``s``. A prior
    tuned to one smoothness does better at that smoothness and worse
    everywhere else; ``a_m = m^2`` gives up a logarithmic factor and
    buys freedom from having to know the truth's smoothness in
    advance.

    That logarithm is the price and it is returned separately as
    ``log_factor`` rather than folded into the rate, because the
    difference between ``n^{-s/(2s+1)}`` and
    ``n^{-s/(2s+1)} log n`` is exactly what "near-optimal" means.

    Parameters
    ----------
    x : array-like
        Observations; used for the sample size when ``n`` is absent.
    s : float, optional
        Smoothness to report the rate at; a range is scanned when
        omitted, which is the point of adaptation.
    n : int, optional
        Sample size to evaluate at.
    levels, a_scale
        Recorded for the corresponding Polya tree fit.

    Returns
    -------
    RichResult
        keys: ``n``, ``smoothness``, ``rate``, ``minimax_rate``,
        ``log_factor``, ``ratio_to_minimax``, ``adaptive`` (True),
        ``requires_knowing_s`` (False), ``scan`` (s, rate pairs),
        ``method``.
    References
    ----------
    Ghosal and van der Vaart, Sec. 7.2.3; adaptation is Chapter 10.
    """
    from ._ghosal import minimax_rate

    xv = np.asarray(x, dtype=float).ravel()
    nn = int(xv.size) if n is None else int(n)
    if nn < 2:
        raise ValueError(f"n must be at least 2, got {nn}.")
    lg = float(np.log(nn))
    scan = [(float(sv), minimax_rate(nn, sv) * lg)
            for sv in (0.5, 1.0, 1.5, 2.0, 3.0)]
    sv = 1.0 if s is None else float(s)
    if sv <= 0:
        raise ValueError(f"smoothness must be positive, got {sv}.")
    mm = minimax_rate(nn, sv)
    return RichResult(payload={
        "n": nn, "smoothness": sv, "rate": mm * lg, "minimax_rate": mm,
        "log_factor": lg, "ratio_to_minimax": lg,
        "adaptive": True, "requires_knowing_s": False, "scan": scan,
        "a_rule": f"a_m = {float(a_scale)} * m^2", "levels": int(levels),
        "method": "Adaptive Polya tree: n^{-s/(2s+1)} log n for every s, without knowing s"})


def cheatsheet():
    return "gh_pt_adapt: the log n factor IS the price of not having to know the smoothness"
