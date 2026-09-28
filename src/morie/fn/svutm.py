# morie.fn -- function file (rootcoder007/morie)
"""Spatial utility maximizer"""

from __future__ import annotations

import math

from ._containers import DescriptiveResult


def _vec(v):
    return [float(a) for a in (v.tolist() if hasattr(v, "tolist") else v)]


def _d2(a, b):
    return math.fsum((p - q) ** 2 for p, q in zip(a, b))


def utility_max(x, *, ideal_point=None, utility: str = "quadratic", scale: float = 1.0):
    r"""The alternative (row of ``x``) that maximises a voter's spatial utility, with all utilities.

    ``utility``: ``quadratic`` ``-d^2``, ``linear`` ``-d`` or ``gaussian``
    ``exp(-d^2 / (2 scale^2))``, ``d`` the Euclidean distance to
    ``ideal_point`` (default origin). All three are decreasing in ``d``, so
    the choice is the nearest alternative; ties go to the first. ``value`` is
    its 0-based index.

    References
    ----------
    Enelow, J. M. and Hinich, M. J. (1984). *The Spatial Theory of Voting*.
    Cambridge University Press.

    Examples
    --------
    >>> r = utility_max([[3.0, 0.0], [1.0, 1.0], [0.0, 2.0]], ideal_point=[0.0, 0.0])
    >>> r.value, r.extra["utilities"]
    (1, [-9.0, -2.0, -4.0])
    """
    X = x.tolist() if hasattr(x, "tolist") else list(x)
    if X and not isinstance(X[0], (list, tuple)):
        X = [X]
    X = [[float(a) for a in r] for r in X]
    v = [0.0] * len(X[0]) if ideal_point is None else _vec(ideal_point)
    D2 = [_d2(v, a) for a in X]
    if utility == "quadratic":
        U = [-d for d in D2]
    elif utility == "linear":
        U = [-math.sqrt(d) for d in D2]
    elif utility == "gaussian":
        U = [math.exp(-d / (2 * scale**2)) for d in D2]
    else:
        raise ValueError("utility must be quadratic, linear or gaussian")
    best = max(range(len(U)), key=lambda j: (U[j], -j))
    return DescriptiveResult(name="svutm", value=best, extra={"utilities": U, "choice": X[best]})


util = utility_max


def cheatsheet() -> str:
    return "utility_max(alternatives, ideal_point, utility) -> utility-maximising alternative"


# compact alias per ledger/NAMING.md
utilitymax = utility_max
