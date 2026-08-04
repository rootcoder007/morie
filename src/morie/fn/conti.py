# morie.fn -- function file (rootcoder007/morie)
"""Continued fraction approximation of pi.

Classical.  Triage confirmed this names no owning source.

The partial quotients of pi are not derived here from a
high-precision value -- a double cannot supply them past the tenth --
they are the known leading terms [3; 7, 15, 1, 292, 1, 1, 1, 2, 1,
3, 1, 14, 2, 1], which are what the expansion of the double pi
reproduces over that range.  Asking for more than fifteen is refused
rather than answered with floating point noise.
"""

from ._richresult import RichResult, with_describe_pointer

__all__ = ["continued_fraction_pi"]

# [3; 7, 15, 1, 292, 1, 1, 1, 2, 1, 3, 1, 14, 2, 1]
_PI_TERMS = [3, 7, 15, 1, 292, 1, 1, 1, 2, 1, 3, 1, 14, 2, 1]


def continued_fraction_pi(n):
    """The first n convergents of the continued fraction of pi.

    The early convergents are the classical rational approximations:
    3, 22/7, 333/106, 355/113, the last of which is accurate to seven
    figures and is why 355/113 is the one people remember.

    Parameters
    ----------
    n : int, number of partial quotients, 1 to 15.

    Returns
    -------
    RichResult with keys estimate (the nth convergent as a float),
    terms, convergents, numerator, denominator, error, n, method.
    """
    nn = int(n)
    if not 1 <= nn <= len(_PI_TERMS):
        raise ValueError("n must be between 1 and %d" % len(_PI_TERMS))
    terms = _PI_TERMS[:nn]
    hm1, hm2 = 1, 0
    km1, km2 = 0, 1
    conv = []
    for a in terms:
        h = a * hm1 + hm2
        k = a * km1 + km2
        conv.append((h, k))
        hm2, hm1 = hm1, h
        km2, km1 = km1, k
    h, k = conv[-1]
    val = h / float(k)
    return with_describe_pointer(RichResult(payload={
        "estimate": val, "terms": terms, "convergents": conv,
        "numerator": h, "denominator": k,
        "error": val - 3.141592653589793, "n": nn,
        "method": "continued fraction convergents of pi",
    }), "conti")


def cheatsheet():
    return "conti: CF approximation of pi"


# compact alias per ledger/NAMING.md
cfpi = continued_fraction_pi
