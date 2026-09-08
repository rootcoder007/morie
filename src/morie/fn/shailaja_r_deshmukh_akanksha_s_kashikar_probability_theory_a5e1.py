# morie.fn -- function file (rootcoder007/morie)
"""Independence of k events."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["indevk", "shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a_chapter_5_equation_1"]


def indevk(p, joint):
    """Independence of k events.

    P(A_i1 ... A_ir) = prod_l P(A_il) for every 2 <= r <= k   (Deshmukh eq. 5.1).

    Independence of k events requires every one of the
    2^k - k - 1 subset conditions to hold, not just pairwise ones.
    ``joint[m]`` is P of the intersection of the events whose bits are
    set in the mask m, for m = 0 .. 2^k - 1; masks with fewer than two
    bits set are ignored.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Independence of k events", payload=_c.indevk(p=p, joint=joint))


shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a_chapter_5_equation_1 = indevk


def cheatsheet():
    return "shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a5e1: Independence of k events"
