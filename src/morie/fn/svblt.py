# morie.fn -- function file (rootcoder007/morie)
"""Boltzmann (softmax) spatial voting"""

from __future__ import annotations

import math

from ._containers import DescriptiveResult


def _vec(v):
    return [float(a) for a in (v.tolist() if hasattr(v, "tolist") else v)]


def _d2(a, b):
    return math.fsum((p - q) ** 2 for p, q in zip(a, b))


def _alts(x, status_quo):
    X = x.tolist() if hasattr(x, "tolist") else list(x)
    if X and not isinstance(X[0], (list, tuple)):
        X = [X]
    X = [[float(a) for a in r] for r in X]
    if status_quo is not None or len(X) == 1:
        X.append([0.0] * len(X[0]) if status_quo is None else _vec(status_quo))
    return X


def boltzmann_vote(x, *, ideal_point=None, status_quo=None, temperature: float = 1.0):
    r"""Boltzmann (multinomial logit) choice probabilities over spatial alternatives.

    ``P_j = exp(-d_j^2 / T) / sum_k exp(-d_k^2 / T)`` with ``d_j`` the distance
    from ``ideal_point`` to alternative ``j``. ``x`` is one alternative (then
    compared with ``status_quo``, default the origin) or a matrix of
    alternatives (``status_quo`` appended when given). ``value`` is the
    probability of the first alternative.

    References
    ----------
    McFadden, D. (1974). Conditional logit analysis of qualitative choice
    behavior. In P. Zarembka (ed.), *Frontiers in Econometrics*, 105-142.

    Examples
    --------
    >>> r = boltzmann_vote([[0.0], [1.0], [2.0]], ideal_point=[0.0])
    >>> [round(p, 6) for p in r.extra["probabilities"]]
    [0.721399, 0.265388, 0.013213]
    """
    X = _alts(x, status_quo)
    v = [0.0] * len(X[0]) if ideal_point is None else _vec(ideal_point)
    U = [-_d2(v, a) / temperature for a in X]
    m = max(U)
    e = [math.exp(u - m) for u in U]
    s = math.fsum(e)
    P = [a / s for a in e]
    return DescriptiveResult(name="svblt", value=P[0], extra={"probabilities": P, "utilities": U})


bolt = boltzmann_vote


def cheatsheet() -> str:
    return "boltzmann_vote(x, ideal_point, temperature) -> softmax spatial choice probabilities"


# compact alias per ledger/NAMING.md
boltzmannvote = boltzmann_vote
