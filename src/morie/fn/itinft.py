# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Item information function for the two-parameter logistic.

Lord (1980), *Applications of Item Response Theory to Practical
Testing Problems*, Lawrence Erlbaum, chapter 5:

    I_j(theta) = a_j^2 P_j(theta) (1 - P_j(theta)),

which for the 2PL peaks at theta = b_j with the value a_j^2 / 4.  Test
information is the sum over items, and the asymptotic standard error of
theta-hat is its reciprocal square root.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["item_information_function"]


def item_information_function(theta, a, b):
    """Information of one or more 2PL items at one or more abilities.

    Parameters
    ----------
    theta : array-like
        Ability values.
    a, b : array-like or float
        Item slopes and difficulties; a scalar is recycled.
    """
    th = core.vec(theta)
    av = core.vec(a)
    bv = core.vec(b)
    if len(th) == 0:
        raise ValueError("item_information_function: theta is empty")
    if len(av) != len(bv):
        raise ValueError("item_information_function: a and b have different lengths")
    if len(av) == 0:
        raise ValueError("item_information_function: no item parameters")
    for v in av:
        if v <= 0:
            raise ValueError("item_information_function: slopes must be positive")
    info = []
    test_info = []
    for t in th:
        row = []
        s = 0.0
        for j in range(len(av)):
            p = core.sigmoid(av[j] * (t - bv[j]))
            v = av[j] * av[j] * p * (1.0 - p)
            row.append(v)
            s += v
        info.append(row)
        test_info.append(s)
    se = [float("inf") if v <= 0 else 1.0 / math.sqrt(v) for v in test_info]
    return RichResult(
        title="Item information function",
        summary_lines=[("abilities", len(th)), ("items", len(av))],
        payload={
            "estimate": max(test_info),
            "information": info,
            "test_information": test_info,
            "se": se,
            "n": len(th),
            "method": "I_j(theta) = a_j^2 P (1 - P), Lord (1980) ch. 5",
        },
    )


def cheatsheet():
    return "itinft: item information function"
