"""Shared primitives for the item response theory modules.

Kept in one place so ``icrf``, ``irt2pl``, ``irt3pl``, ``iinfo``, ``thetml``,
``irtras``, ``rsmand``, ``irtnrm`` and ``nrm`` cannot drift apart. The R arm
of this file is ``R/aaa_helpers_irt.R``.

No numpy: pure ``math`` throughout.
"""

from __future__ import annotations

import math

INF = float("inf")

__all__ = ["INF", "seq_", "broadcast", "expit", "softmax", "as_matrix"]


def seq_(x):
    """Coerce a scalar or array-like to a plain list."""
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, (int, float)):
        return [x]
    return list(x)


def broadcast(v, n, name):
    """Recycle a length-1 value to length ``n``; otherwise demand length ``n``."""
    vals = [float(u) for u in seq_(v)]
    if len(vals) == 1:
        return vals * n
    if len(vals) != n:
        raise ValueError(
            "%s has length %d; expected 1 or %d" % (name, len(vals), n)
        )
    return vals


def expit(z):
    """1/(1+exp(-z)) written so neither tail overflows."""
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def softmax(eta):
    """exp(eta_k) / sum_h exp(eta_h), shifted by the maximum for stability."""
    m = max(eta)
    ex = [math.exp(e - m) for e in eta]
    s = sum(ex)
    return [e / s for e in ex]


def as_matrix(x, name):
    """Coerce to a list of equal-length lists of floats."""
    if hasattr(x, "tolist"):
        x = x.tolist()
    rows = [seq_(r) for r in x]
    if not rows:
        raise ValueError("%s is empty." % name)
    w = len(rows[0])
    for r in rows:
        if len(r) != w:
            raise ValueError("%s has ragged rows." % name)
    return [[float(v) for v in r] for r in rows]
