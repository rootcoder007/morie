# morie.fn -- function file (rootcoder007/morie)
"""Independence of two random variables."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["indrv2", "shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a_chapter_5_equation_3"]


def indrv2(joint):
    """Independence of two random variables.

    P[X1 in S1, X2 in S2] = P[X1 in S1] P[X2 in S2]   (Deshmukh eq. 5.3).

    Independence of two random variables as factorisation of the joint
    distribution into its marginals.  ``joint`` is a probability table
    over a finite partition of the two ranges; the marginals are its row
    and column sums, and the deviation reported is the largest absolute
    departure from the product.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Independence of two random variables", payload=_c.indrv2(joint=joint))


shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a_chapter_5_equation_3 = indrv2


def cheatsheet():
    return "shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a5e3: Independence of two random variables"
