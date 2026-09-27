"""Urban sprawl by relative Shannon entropy of built-up land (Yeh and Li 2001).

Yeh, A. G.-O. and Li, X. (2001). Measurement and monitoring of urban sprawl in a rapidly
growing region using entropy. Photogrammetric Engineering and Remote Sensing 67, 83-90.
"""

import math

from ._richresult import RichResult

__all__ = ["sprawl_entropy"]


def sprawl_entropy(built, area=None):
    r"""Relative entropy H_n = -sum_i p_i log(p_i) / log(n) of built-up land over n zones
    (rings or sectors), p_i = density_i / sum density, density_i = built_i / area_i.

    H_n near 1 means built-up land is spread evenly across the zones (dispersed, sprawling
    growth); near 0 it is concentrated in few zones (compact).

    Parameters
    ----------
    built : sequence
        Built-up area per zone.
    area : sequence, optional
        Zone areas (densities are used when given).

    Returns
    -------
    RichResult
        Keys: entropy (H), relative (H_n), max (log n), shares.

    References
    ----------
    Yeh, A. G.-O. and Li, X. (2001). Photogrammetric Engineering and Remote Sensing 67, 83-90.

    Examples
    --------
    >>> sprawl_entropy([5, 5, 5, 5])["relative"]
    1.0
    """
    b = [float(v) for v in built]
    n = len(b)
    if n < 2 or min(b) < 0:
        raise ValueError("need at least two zones with non-negative built-up area")
    dens = b if area is None else [v / float(a) for v, a in zip(b, area)]
    tot = 0.0
    for v in dens:
        tot += v
    p = [v / tot for v in dens]
    H = 0.0
    for v in p:
        if v > 0:
            H -= v * math.log(v)
    return RichResult(
        title="Sprawl entropy",
        summary_lines=[("relative entropy", H / math.log(n))],
        payload={"entropy": H, "relative": H / math.log(n), "max": math.log(n), "shares": p},
    )


def cheatsheet():
    return "sprawl: relative Shannon entropy of built-up land (Yeh and Li 2001)"
