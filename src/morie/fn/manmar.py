# morie.fn -- function file (rootcoder007/morie)
"""Bucher adjusted indirect comparison."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['bucherind', 'ma_network_indirect']


def bucherind(d_AB, v_AB, d_CB, v_CB, alpha=0.05):
    """Bucher adjusted indirect comparison.

    Comparing A with C through a common comparator B preserves randomisation within each trial, which naive comparison of single arms across trials does not. The variance simply adds because the two direct comparisons come from disjoint sets of trials; if they share trials, this is the wrong function.


    Formula: d_AC = d_AB - d_CB; var(d_AC) = v_AB + v_CB

    Parameters
    ----------
    d_AB : float
        Direct estimate of A versus B.
    v_AB : float
        Its variance.
    d_CB : float
        Direct estimate of C versus B.
    v_CB : float
        Its variance.
    alpha : float
        Two-sided significance level.

    Returns
    -------
    RichResult
        ``estimate``, ``variance``, ``se``, ``z``, ``p_value``, ``ci_lower``, ``ci_upper``.

    References
    ----------
    Bucher, Guyatt, Griffith and Walter (1997), The results of direct
    and indirect treatment comparisons in meta-analysis of randomized
    controlled trials, Journal of Clinical Epidemiology 50:683-691.
    Paywalled; d_AC = d_AB - d_CB with variances added is the standard
    published form, restated identically in every network meta-analysis
    source consulted.
    """
    d = float(d_AB) - float(d_CB)
    var = float(v_AB) + float(v_CB)
    if var <= 0:
        raise ValueError("variances must be positive")
    se = math.sqrt(var)
    z = d / se
    zc = C.qnorm(1.0 - float(alpha) / 2.0)
    return RichResult(payload={
        "estimate": d, "variance": var, "se": se, "z": z,
        "p_value": 2.0 * (1.0 - C.pnorm(abs(z))),
        "ci_lower": d - zc * se, "ci_upper": d + zc * se,
        "method": "Bucher adjusted indirect comparison"})


ma_network_indirect = bucherind


def cheatsheet():
    return "manmar: Bucher adjusted indirect comparison."
