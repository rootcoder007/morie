"""Binomial pmf normalization: sum over k equals 1.

Implements eq (4.10) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_10"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_10(n, p):
    """Binomial pmf normalization: sum over k equals 1.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (4.10).
    """
    pmf = _morin.binomial_pmf_vector(n, p)
    payload = {"pmf": [float(x) for x in pmf], "total": float(pmf.sum())}
    lines = [("sum of pmf", float(pmf.sum()))]
    return RichResult(
        title="Binomial pmf normalization: sum over k equals 1.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner4e10: Binomial pmf normalization: sum over k equals 1. Morin (2016) eq (4.10)."
