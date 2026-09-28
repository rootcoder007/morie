# morie.fn -- function file (rootcoder007/morie)
"""Proximity voting model"""

from __future__ import annotations

import math

from ._containers import DescriptiveResult


def _vec(v):
    return [float(a) for a in (v.tolist() if hasattr(v, "tolist") else v)]


def _d2(a, b):
    return math.fsum((p - q) ** 2 for p, q in zip(a, b))


def proximity_vote(x, *, ideal_point=None, status_quo=None, p: float = 2.0):
    r"""Deterministic proximity vote: 1 if ``x`` is closer than ``status_quo`` to ``ideal_point``, 0.5 on a tie, else 0.

    Distance is the Minkowski ``p``-norm (``p = 2`` Euclidean, ``p = 1``
    city block; Enelow and Hinich 1984, chapter 2).

    References
    ----------
    Enelow, J. M. and Hinich, M. J. (1984). *The Spatial Theory of Voting*.
    Cambridge University Press.

    Examples
    --------
    >>> proximity_vote([1.0, 1.0], ideal_point=[0.0, 0.0], status_quo=[2.0, 0.0], p=1).value
    0.5
    """
    x = _vec(x)
    v = [0.0] * len(x) if ideal_point is None else _vec(ideal_point)
    s = [0.0] * len(x) if status_quo is None else _vec(status_quo)
    dx = sum(abs(a - b) ** p for a, b in zip(v, x)) ** (1 / p)
    ds = sum(abs(a - b) ** p for a, b in zip(v, s)) ** (1 / p)
    val = 1.0 if dx < ds else 0.5 if dx == ds else 0.0
    return DescriptiveResult(name="svpxv", value=val, extra={"distance_x": dx, "distance_status_quo": ds})


prox = proximity_vote


def cheatsheet() -> str:
    return "proximity_vote(x, ideal_point, status_quo, p) -> deterministic proximity vote"


# compact alias per ledger/NAMING.md
proximityvote = proximity_vote
