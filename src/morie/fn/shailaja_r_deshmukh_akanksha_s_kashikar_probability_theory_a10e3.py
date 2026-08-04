# morie.fn -- function file (rootcoder007/morie)
"""Degenerate limiting distribution of the sample mean."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["degencdf", "shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a_chapter_10_equation_3"]


def degencdf(x, mu):
    """Degenerate limiting distribution of the sample mean.

    P[Xbar_n <= x] -> 0 if x < mu, 1 if x > mu   (Deshmukh eq. 10.3).

    Limiting distribution function of the sample mean of iid variables
    with finite mean mu: by Khintchine's WLLN the sample mean converges
    in probability, hence in law, to the degenerate law at mu.  The book
    is explicit that the limit at x = mu is not determined by the given
    information, so it is returned as NaN rather than guessed at 1/2.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Degenerate limiting distribution of the sample mean", payload=_c.degencdf(x=x, mu=mu))


shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a_chapter_10_equation_3 = degencdf


def cheatsheet():
    return "shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a10e3: Degenerate limiting distribution of the sample mean"
