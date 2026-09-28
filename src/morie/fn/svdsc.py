# morie.fn -- function file (rootcoder007/morie)
"""Discounted (Grofman) spatial utility"""

from __future__ import annotations

import math

from ._containers import DescriptiveResult


def _vec(v):
    return [float(a) for a in (v.tolist() if hasattr(v, "tolist") else v)]


def _d2(a, b):
    return math.fsum((p - q) ** 2 for p, q in zip(a, b))


def discount_utility(x, *, ideal_point=None, status_quo=None, discount: float = 0.5, beta: float = 1.0):
    r"""Grofman discounting-model utility of a candidate's platform ``x``.

    Voters expect only a fraction ``discount`` of the promised move away from
    the status quo to be enacted, so they evaluate the perceived position
    ``s + discount (x - s)``: ``U = -beta ||v - s - discount (x - s)||^2``.
    ``discount = 1`` is pure proximity voting.

    References
    ----------
    Grofman, B. (1985). The neglected role of the status quo in models of
    issue voting. *Journal of Politics*, 47(1), 230-237.

    Examples
    --------
    >>> discount_utility([3.0], ideal_point=[1.0], status_quo=[0.0], discount=0.5).value
    -0.25
    """
    x = _vec(x)
    v = [0.0] * len(x) if ideal_point is None else _vec(ideal_point)
    s = [0.0] * len(x) if status_quo is None else _vec(status_quo)
    perceived = [b + discount * (a - b) for a, b in zip(x, s)]
    return DescriptiveResult(name="svdsc", value=-beta * _d2(v, perceived), extra={"perceived_position": perceived})


disc = discount_utility


def cheatsheet() -> str:
    return "discount_utility(x, ideal_point, status_quo, discount) -> Grofman discounted utility"
