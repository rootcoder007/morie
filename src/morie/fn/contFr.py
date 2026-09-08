# morie.fn -- function file (rootcoder007/morie)
"""Continued fraction expansion of a real number.

Classical number theory.  Triage confirmed this names no owning
source; the standard algorithm is implemented and no citation is
manufactured.
"""

import math

from ._richresult import RichResult, with_describe_pointer

__all__ = ["continued_fraction"]


def _convergents(terms):
    """h_k = a_k h_{k-1} + h_{k-2}, k_k = a_k k_{k-1} + k_{k-2}, the
    standard recurrence for the numerators and denominators of the
    convergents."""
    hm1, hm2 = 1, 0
    km1, km2 = 0, 1
    out = []
    for a in terms:
        h = a * hm1 + hm2
        k = a * km1 + km2
        out.append((h, k))
        hm2, hm1 = hm1, h
        km2, km1 = km1, k
    return out


def continued_fraction(x, n):
    """Simple continued fraction a_0 + 1/(a_1 + 1/(a_2 + ...)).

    Each step takes the integer part, subtracts it and inverts the
    remainder.  The expansion terminates early if the remainder hits
    exactly zero, which happens for rationals.

    Accuracy warning.  The remainder loses roughly one decimal digit
    per term, so on a double only the first ten or so partial
    quotients of an irrational are trustworthy; terms past that are
    an artefact of the floating point residue, not of the number.
    ``n`` is therefore capped at 20 and ``reliable_terms`` reports how
    many terms the residue still supported.

    Parameters
    ----------
    x : float, the number to expand.
    n : int, maximum number of partial quotients (at most 20).

    Returns
    -------
    RichResult with keys estimate (the last convergent as a float),
    terms, convergents, reliable_terms, residual, n, method.
    """
    nn = int(n)
    if nn < 1:
        raise ValueError("need at least one term")
    if nn > 20:
        raise ValueError("a double supports at most 20 partial quotients")
    v = float(x)
    terms = []
    r = v
    reliable = 0
    for i in range(nn):
        a = math.floor(r)
        terms.append(int(a))
        r = r - a
        # the residue is meaningful while it is above the rounding
        # floor of the value it was carved out of
        # Stop as soon as the remainder drops below the rounding floor
        # of the value it came from.  Continuing past that point emits
        # a partial quotient built entirely out of floating point
        # residue -- 415/93 is exactly [4; 2, 6, 7], and one more step
        # invents a spurious 262386368408.
        if abs(r) <= 1e-12 * max(1.0, abs(v)):
            break
        reliable = i + 1
        r = 1.0 / r
    conv = _convergents(terms)
    h, k = conv[-1]
    return with_describe_pointer(RichResult(payload={
        "estimate": h / float(k), "terms": terms,
        "convergents": conv, "reliable_terms": reliable,
        "residual": float(v - h / float(k)), "n": len(terms),
        "method": "simple continued fraction expansion",
    }), "contFr")


def cheatsheet():
    return "contFr: Continued fraction expansion"


# compact alias per ledger/NAMING.md
contfrac = continued_fraction
