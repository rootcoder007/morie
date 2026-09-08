# morie.fn -- function file (rootcoder007/morie)
"""Kesten-Stigum detectability threshold for the symmetric SBM."""

import math

from ._richresult import RichResult

__all__ = ["sgt_sbm_detect_threshold"]


def sgt_sbm_detect_threshold(a, b, k=2):
    """Is a planted partition detectable at these SBM parameters?

    ``a`` and ``b`` are the rescaled within- and between-group degrees
    ``c_in`` and ``c_out``, so the mean degree is
    ``c = (a + (k - 1) b) / k``.

    Formula: detectable iff ``|a - b| > k sqrt(c)``, equivalently
    ``(a - b)^2 > k (a + (k - 1) b)``.

    NOTE ON THE STUB'S FORMULA.  The placeholder this replaces carried
    ``(a - b)^2 > k (a + (k-1) b) / (k-1)``.  That extra ``1/(k-1)``
    is wrong for every ``k`` except 2, where the two coincide.  Equation
    (44) on page 14 of arXiv:1109.3041 reads ``|c_in - c_out| > q sqrt(c)``
    with ``c_in + (q-1) c_out = q c``; the page was rendered as an image
    and read visually, not from a text layer.

    Parameters
    ----------
    a, b : float
        Rescaled within- and between-group degrees, both non-negative.
    k : int, default 2
        Number of equally sized groups, at least 2.

    Returns
    -------
    RichResult
        ``detectable`` (1.0 / 0.0), ``estimate`` (the same), ``margin``
        (``|a - b| - k sqrt(c)``, positive when detectable), ``c``,
        ``threshold`` (``k sqrt(c)``).

    References
    ----------
    Decelle, A., Krzakala, F., Moore, C. & Zdeborova, L. (2011).
    Asymptotic analysis of the stochastic block model for modular
    networks and its algorithmic applications.  Physical Review E 84,
    066106.  doi:10.1103/PhysRevE.84.066106; equation (44).
    """
    a = float(a)
    b = float(b)
    k = int(k)
    if k < 2:
        raise ValueError("sgt_sbm_detect_threshold: k must be at least 2")
    if a < 0.0 or b < 0.0:
        raise ValueError("sgt_sbm_detect_threshold: degrees must be non-negative")
    c = (a + (k - 1) * b) / k
    thr = k * math.sqrt(c)
    margin = abs(a - b) - thr
    det = 1.0 if margin > 0.0 else 0.0
    return RichResult(payload={
        "detectable": det, "estimate": det, "margin": margin, "c": c,
        "threshold": thr, "k": k,
        "method": "Kesten-Stigum SBM detectability, |a-b| > k sqrt(c)"})


def cheatsheet():
    return "sgtsbnd: Kesten-Stigum SBM detectability threshold"
