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


def rsm_probs(theta, b, tau):
    r"""Andrich rating scale category probabilities for one item.

    Categories h = 0, ..., m with m = len(tau). With cumulative thresholds
    :math:`T_h = \sum_{j\le h} \tau_j` (and :math:`T_0 = 0`),

    .. math::
        \eta_h = h(\theta - b) - T_h, \qquad
        P(X = h) = \frac{e^{\eta_h}}{\sum_l e^{\eta_l}} .

    Returns ``(probs, eta)``.
    """
    m = len(tau)
    cum = 0.0
    eta = [0.0]
    for j in range(m):
        cum += tau[j]
        eta.append((j + 1) * (theta - b) - cum)
    return softmax(eta), eta


def nrm_probs(theta, a, c):
    r"""Bock nominal response category probabilities for one item.

    .. math::
        \eta_r = a_r \theta + c_r, \qquad
        P(X = r) = e^{\eta_r} / \sum_s e^{\eta_s}.

    Returns ``(probs, eta)``.
    """
    eta = [a[r] * theta + c[r] for r in range(len(a))]
    return softmax(eta), eta


def cat_moments(probs, scores):
    """Mean and variance of the category scores under ``probs``.

    For any of these exponential-family category models the Fisher
    information in one item is exactly the variance of the category score,
    because the linear predictor is score * theta plus a constant.
    """
    mu = 0.0
    for h in range(len(probs)):
        mu += probs[h] * scores[h]
    v = 0.0
    for h in range(len(probs)):
        v += probs[h] * (scores[h] - mu) ** 2
    return mu, v
