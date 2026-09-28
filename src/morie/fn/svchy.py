# morie.fn -- function file (rootcoder007/morie)
"""Cauchy spatial voting probability"""

from __future__ import annotations

import math

from ._containers import DescriptiveResult
from .spatialvote import _cdf


def _vec(v):
    return [float(a) for a in (v.tolist() if hasattr(v, "tolist") else v)]


def _d2(a, b):
    return math.fsum((p - q) ** 2 for p, q in zip(a, b))


def cauchy_vote(x, *, ideal_point=None, status_quo=None, beta: float = 1.0):
    r"""Cauchy spatial voting probability: probability a voter at ``ideal_point`` votes for ``x`` over ``status_quo``.

    Quadratic utility ``U(z) = -beta ||v - z||^2`` plus random error; the
    voter picks ``x`` when ``U(x) + e_x > U(s) + e_s``, so ``P = F(U(x) -
    U(s))`` with ``F`` the cauchy cdf of the utility-difference noise.
    ``ideal_point`` and ``status_quo`` default to the origin.

    References
    ----------
    Enelow, J. M. and Hinich, M. J. (1984). *The Spatial Theory of Voting*.
    Cambridge University Press.

    Examples
    --------
    >>> r = cauchy_vote([1.0], ideal_point=[0.0], status_quo=[2.0])
    >>> round(r.value, 6)
    0.897584
    """
    x = _vec(x)
    v = [0.0] * len(x) if ideal_point is None else _vec(ideal_point)
    s = [0.0] * len(x) if status_quo is None else _vec(status_quo)
    du = beta * (_d2(v, s) - _d2(v, x))
    return DescriptiveResult(name="svchy", value=_cdf(du, "cauchy"), extra={"utility_difference": du})


cauc = cauchy_vote


def cheatsheet() -> str:
    return "cauchy_vote(x, ideal_point, status_quo) -> Cauchy spatial voting probability"


# compact alias per ledger/NAMING.md
cauchyvote = cauchy_vote
