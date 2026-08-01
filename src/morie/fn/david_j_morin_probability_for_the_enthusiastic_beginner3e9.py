"""Independence of two discrete variables: the joint pmf factorizes.

Implements eq (3.9) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_9"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_9(joint, tol=1e-9):
    """Independence of two discrete variables: the joint pmf factorizes.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.9).
    """
    independent, px, py = _morin.joint_independent(joint, tol)
    payload = {"independent": independent,
               "marginal_x": [float(v) for v in px],
               "marginal_y": [float(v) for v in py]}
    lines = [("independent", independent)]
    return RichResult(
        title="Independence of two discrete variables: the joint pmf factorizes.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e9: Independence of two discrete variables: the joint pmf factorizes. Morin (2016) eq (3.9)."
