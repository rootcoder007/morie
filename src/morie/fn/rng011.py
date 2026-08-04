# morie.fn -- function file (rootcoder007/morie)
"""Shannon entropy of a discrete process (Rangayyan eq. 3.11)."""


from math import fsum, log

from ._rgcore import aslist
from ._richresult import RichResult

__all__ = ["shannon", "rangayyan_ch3_shannon_entropy_discrete"]


def shannon(p, levels=None):
    """Shannon entropy of an L-level quantized process, in bits.

    Rangayyan (2024) eq. (3.11):
        H = - sum_{l=0}^{L-1} p(eta_l) log2[p(eta_l)].

    Parameters
    ----------
    p : array-like
        Probabilities of the L quantized values, or -- when ``levels`` is
        given -- raw observations to be binned into that many equal-width
        levels and converted to relative frequencies.
    levels : int, optional
        Number of quantization levels.

    Notes
    -----
    Zero-probability levels contribute nothing (p log p -> 0).  The book
    states entropy is maximal for a uniform PDF, which is log2(L) bits;
    that ceiling is returned alongside so the value can be read as a
    fraction of the maximum.
    """
    vals = aslist(p)
    if not vals:
        raise ValueError("need at least one value")
    if levels is not None:
        lv = int(levels)
        if lv < 1:
            raise ValueError("levels must be positive")
        lo, hi = min(vals), max(vals)
        span = hi - lo
        counts = [0] * lv
        for v in vals:
            k = 0 if span == 0 else min(lv - 1, int((v - lo) / span * lv))
            counts[k] += 1
        probs = [c / len(vals) for c in counts]
    else:
        if any(v < 0 for v in vals):
            raise ValueError("probabilities must be nonnegative")
        total = fsum(vals)
        if total <= 0:
            raise ValueError("probabilities must sum to a positive value")
        probs = [v / total for v in vals]
    ln2 = log(2.0)
    h = -fsum(q * log(q) / ln2 for q in probs if q > 0)
    lv = len(probs)
    return RichResult(payload={
        "entropy": float(h), "units": "bits", "levels": lv,
        "max_entropy": log(lv) / ln2 if lv > 1 else 0.0,
        "probabilities": probs,
        "method": "Rangayyan (2024) eq. (3.11)"})


rangayyan_ch3_shannon_entropy_discrete = shannon  # pre-policy spelling


def cheatsheet():
    return "rng011: Shannon entropy, Rangayyan eq. (3.11)"
