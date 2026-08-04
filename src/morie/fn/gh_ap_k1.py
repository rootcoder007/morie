# morie.fn -- function file (rootcoder007/morie)
"""Fano's inequality: a lower bound on the error probability."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["fano", "ghosal_fano_ineq"]


def fano(M, mutual_info, base_e=True):
    """Fano lower bound on the error probability of an M-ary test.

    This is the direction the testing lemmas do not give: an
    IMPOSSIBILITY.  No procedure whatsoever can do better, so it is
    what turns a packing-set construction into a minimax lower bound.
    The bound is only informative once log M exceeds I + log 2, which
    is why lower-bound proofs work so hard to make the packing set
    large relative to the mutual information.

    Formula: P_err >= 1 - (I(theta; X) + log 2) / log M

    Parameters
    ----------
    M : int
        Number of hypotheses, M >= 2.
    mutual_info : float
        I(theta; X), non-negative, in the same units as the logarithm.
    base_e : bool
        True for nats (natural log), False for bits (log base 2).

    Returns
    -------
    RichResult
        ``bound`` (clipped to [0, 1]), ``raw_bound``, ``log_M``,
        ``informative`` (1 when the raw bound exceeds 0), ``M``.

    References
    ----------
    Fano (1961), Transmission of Information: A Statistical Theory of
    Communications, MIT Press, and Cover & Thomas (2006), Elements of
    Information Theory, 2nd edition, Theorem 2.10.1, for the form used
    here.  The worklist filed this under "Ghosal Appendix K"; the copy
    of Ghosal & van der Vaart (2017) held in the corpus was searched in
    full and the word "Fano" does NOT occur in it, so the attribution
    could not be confirmed and the primary sources are cited instead.
    """
    M = int(M)
    I = float(mutual_info)
    if M < 2:
        raise ValueError("M must be at least 2")
    if I < 0:
        raise ValueError("the mutual information must be non-negative")
    lg = math.log(M) if base_e else math.log(M, 2.0)
    l2 = math.log(2.0) if base_e else 1.0
    raw = 1.0 - (I + l2) / lg
    return RichResult(payload={
        "bound": min(1.0, max(0.0, raw)), "raw_bound": raw, "log_M": lg,
        "informative": 1.0 if raw > 0.0 else 0.0, "M": float(M),
        "method": "Fano inequality lower bound"})


ghosal_fano_ineq = fano


def cheatsheet():
    return "gh_ap_k1: P_err >= 1 - (I + log 2)/log M"
