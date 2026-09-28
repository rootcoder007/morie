# morie.fn -- function file (rootcoder007/morie)
"""Mixed proximity-directional (unified) spatial utility"""

from __future__ import annotations

import math

from ._containers import DescriptiveResult


def _vec(v):
    return [float(a) for a in (v.tolist() if hasattr(v, "tolist") else v)]


def _d2(a, b):
    return math.fsum((p - q) ** 2 for p, q in zip(a, b))


def mixed_utility(x, *, ideal_point=None, neutral=None, beta: float = 0.5):
    r"""Merrill-Grofman unified utility mixing directional and proximity components.

    ``U = 2 (1 - beta) (v - n) . (x - n) - beta ||v - x||^2`` with ``n`` the
    neutral point (default origin): ``beta = 1`` is pure proximity,
    ``beta = 0`` twice the Rabinowitz-Macdonald directional scalar product.

    References
    ----------
    Merrill, S. and Grofman, B. (1999). *A Unified Theory of Voting:
    Directional and Proximity Spatial Models*. Cambridge University Press.

    Examples
    --------
    >>> r = mixed_utility([2.0], ideal_point=[1.0], beta=0.5)
    >>> r.value, r.extra["directional"], r.extra["proximity"]
    (1.5, 2.0, -1.0)
    """
    x = _vec(x)
    v = [0.0] * len(x) if ideal_point is None else _vec(ideal_point)
    n = [0.0] * len(x) if neutral is None else _vec(neutral)
    dirn = math.fsum((a - c) * (b - c) for a, b, c in zip(v, x, n))
    prox = -_d2(v, x)
    return DescriptiveResult(
        name="svmxu", value=2 * (1 - beta) * dirn + beta * prox, extra={"directional": dirn, "proximity": prox}
    )


mixe = mixed_utility


def cheatsheet() -> str:
    return "mixed_utility(x, ideal_point, neutral, beta) -> Merrill-Grofman unified utility"


# compact alias per ledger/NAMING.md
mixedutility = mixed_utility
