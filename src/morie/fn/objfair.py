# morie.fn -- function file (rootcoder007/morie)
"""Individual fairness: Lipschitz constraint on a classifier."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["individual_fairness_lipschitz"]


def individual_fairness_lipschitz(h_values, x_pairs, L=1.0, metric=None):
    """Audit a scoring function against the Lipschitz fairness condition.

    Dwork et al.'s condition is that similar individuals receive similar
    treatment, made precise as a Lipschitz constraint on the map from
    individuals to outcome distributions:

        |h(x) - h(x')| <= L d(x, x')   for every pair (x, x').

    A pair violates it when the score gap exceeds ``L`` times the
    distance.  The audit reports the worst offending pair and the whole
    violation count, plus the smallest ``L`` that would make the given
    scores admissible,

        L_required = max_{pairs} |h(x) - h(x')| / d(x, x'),

    which is the Lipschitz seminorm of ``h`` on the observed pairs.

    Pairs at distance zero are treated separately: two individuals the
    metric cannot tell apart must receive the SAME score, so any gap at
    all is a violation and no finite ``L`` repairs it.  Silently
    dividing by zero there would report ``inf`` for a genuine violation
    and ``nan`` for a compliant identical pair, which is the wrong way
    round.

    Parameters
    ----------
    h_values : array-like, shape (n,)
        Scores assigned by the classifier, one per individual.
    x_pairs : array-like, shape (m, 2)
        Index pairs ``(i, j)`` to audit, 0-based.
    L : float, default 1.0
        Lipschitz constant, non-negative.
    metric : array-like, shape (m,) or None
        Distance ``d(x_i, x_j)`` for each pair.  If ``None``, every
        distance is taken to be 1, which turns the audit into a plain
        score-gap bound.

    Returns
    -------
    RichResult
        ``estimate`` (``L_required``), ``L_required``, ``n_violations``,
        ``violation_rate``, ``max_gap``, ``max_pair_i``, ``max_pair_j``,
        ``fair`` (1 if no pair violates), ``n``, ``n_pairs``.

    References
    ----------
    Dwork, C., Hardt, M., Pitassi, T., Reingold, O. & Zemel, R. (2012).
    Fairness through awareness.  Proceedings of the 3rd Innovations in
    Theoretical Computer Science Conference (ITCS '12), 214--226.
    doi:10.1145/2090236.2090255
    """
    h = C.vec(h_values)
    n = len(h)
    if n == 0:
        raise ValueError("individual_fairness_lipschitz: h_values is empty")
    lam = float(L)
    if lam < 0.0:
        raise ValueError("individual_fairness_lipschitz: L must be non-negative")
    pairs = [(int(p[0]), int(p[1])) for p in x_pairs]
    m = len(pairs)
    if m == 0:
        raise ValueError("individual_fairness_lipschitz: x_pairs is empty")
    for i, j in pairs:
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError("individual_fairness_lipschitz: pair index out of range")
    if metric is None:
        d = [1.0] * m
    else:
        d = C.vec(metric)
        if len(d) != m:
            raise ValueError("individual_fairness_lipschitz: metric and x_pairs differ in length")
        if any(v < 0.0 for v in d):
            raise ValueError("individual_fairness_lipschitz: distances must be non-negative")
    viol = 0
    lreq = 0.0
    maxgap = 0.0
    bi, bj = -1, -1
    for k in range(m):
        i, j = pairs[k]
        gap = abs(h[i] - h[j])
        if gap > maxgap:
            maxgap = gap
            bi, bj = i, j
        if d[k] == 0.0:
            if gap > 0.0:
                viol += 1
                lreq = float("inf")
        else:
            if gap > lam * d[k]:
                viol += 1
            ratio = gap / d[k]
            if ratio > lreq:
                lreq = ratio
    return RichResult(payload={
        "estimate": lreq, "L_required": lreq, "n_violations": float(viol),
        "violation_rate": viol / float(m), "max_gap": maxgap,
        "max_pair_i": float(bi), "max_pair_j": float(bj),
        "fair": 1.0 if viol == 0 else 0.0, "L": lam, "n": n,
        "n_pairs": float(m),
        "method": "Individual fairness audit (Dwork et al. 2012)"})


def cheatsheet():
    return "objfair: Lipschitz individual-fairness audit"


individualfairnesslipschitz = individual_fairness_lipschitz
