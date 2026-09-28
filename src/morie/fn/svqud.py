# morie.fn -- function file (rootcoder007/morie)
"""Quadratic spatial utility function"""

from __future__ import annotations

import math

from ._containers import DescriptiveResult


def _vec(v):
    return [float(a) for a in (v.tolist() if hasattr(v, "tolist") else v)]


def _d2(a, b):
    return math.fsum((p - q) ** 2 for p, q in zip(a, b))


def quad_utility(x, *, ideal_point=None, salience=None):
    r"""Weighted quadratic spatial utility ``U(x) = -(x - v)' A (x - v)``.

    ``salience`` is a vector (diagonal ``A``) or a full symmetric
    positive-definite matrix; default identity (Euclidean loss). Off-diagonal
    terms make preferences non-separable across issues (Enelow and Hinich
    1984, chapter 2).

    References
    ----------
    Enelow, J. M. and Hinich, M. J. (1984). *The Spatial Theory of Voting*.
    Cambridge University Press.

    Examples
    --------
    >>> quad_utility([1.0, 2.0], ideal_point=[0.0, 0.0], salience=[[1.0, 0.5], [0.5, 1.0]]).value
    -7.0
    """
    x = _vec(x)
    v = [0.0] * len(x) if ideal_point is None else _vec(ideal_point)
    d = [a - b for a, b in zip(x, v)]
    k = len(d)
    if salience is None:
        A = [[float(i == j) for j in range(k)] for i in range(k)]
    else:
        S = salience.tolist() if hasattr(salience, "tolist") else list(salience)
        if S and isinstance(S[0], (list, tuple)):
            A = [[float(a) for a in r] for r in S]
        else:
            A = [[float(S[i]) if i == j else 0.0 for j in range(k)] for i in range(k)]
    u = -math.fsum(d[i] * A[i][j] * d[j] for i in range(k) for j in range(k))
    return DescriptiveResult(name="svqud", value=u, extra={"loss": -u})


quad = quad_utility


def cheatsheet() -> str:
    return "quad_utility(x, ideal_point, salience) -> weighted quadratic spatial utility"


# compact alias per ledger/NAMING.md
quadutility = quad_utility
