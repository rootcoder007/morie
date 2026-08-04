# morie.fn -- function file (rootcoder007/morie)
"""MuZero pUCT action selection with min-max normalised values."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["mzpuct", "muzero_efficient_exploration"]


def mzpuct(Q, N, P, c1=1.25, c2=19652.0, qmin=None, qmax=None):
    """Score every action of a search node by the MuZero pUCT rule.

    Each simulation descends the tree by maximising an upper confidence
    bound that trades the prior P(s,a) against the mean value Q(s,a):

        a = argmax_a [ Q(s,a) + P(s,a) sqrt(sum_b N(s,b)) / (1 + N(s,a))
                       * ( c1 + log( (sum_b N(s,b) + c2 + 1) / c2 ) ) ]

    Because values are unbounded outside two-player zero-sum games, the Q
    term is first rescaled onto [0, 1] by the smallest and largest values
    seen so far in the tree,

        Qbar(s,a) = (Q(s,a) - qmin) / (qmax - qmin),

    which is what makes the fixed constants c1 = 1.25 and c2 = 19652
    transferable across environments.

    Parameters
    ----------
    Q : array-like
        Mean action values at this node.
    N : array-like
        Visit counts, same length as Q.
    P : array-like
        Prior policy at this node, same length as Q.
    c1, c2 : float
        Exploration constants; the paper's values are the defaults.
    qmin, qmax : float or None
        Tree-wide value bounds for the normalisation.  ``None`` takes the
        minimum and maximum of ``Q`` itself; if they coincide the Q term
        is left unscaled at zero.

    Returns
    -------
    RichResult
        ``score``, ``qbar``, ``exploration``, ``best``, ``sumn``, ``k``.

    References
    ----------
    Schrittwieser, J., Antonoglou, I., Hubert, T., Simonyan, K.,
    Sifre, L., Schmitt, S., Guez, A., Lockhart, E., Hassabis, D.,
    Graepel, T., Lillicrap, T. and Silver, D. (2020), "Mastering Atari,
    Go, chess and shogi by planning with a learned model", Nature 588,
    604-609; arXiv:1911.08265.  Read from the ar5iv rendering of the
    arXiv source.  Equation (2) is the pUCT rule above and states c1 = 1.25,
    c2 = 19652; the paragraph following Equation (4) introduces the
    min-max normalisation of Q by the extreme values observed in the
    search tree.
    """
    Q = C.vec(Q)
    N = C.vec(N)
    P = C.vec(P)
    k = len(Q)
    if len(N) != k or len(P) != k:
        raise ValueError("Q, N and P must have the same length")
    if any(v < 0.0 for v in N):
        raise ValueError("visit counts must be non-negative")
    lo = min(Q) if qmin is None else float(qmin)
    hi = max(Q) if qmax is None else float(qmax)
    rng = hi - lo
    qb = [0.0] * k if rng <= 0.0 else [(v - lo) / rng for v in Q]
    sn = sum(N)
    c1, c2 = float(c1), float(c2)
    u = c1 + math.log((sn + c2 + 1.0) / c2)
    ex = [P[a] * math.sqrt(sn) / (1.0 + N[a]) * u for a in range(k)]
    sc = [qb[a] + ex[a] for a in range(k)]
    best = max(range(k), key=lambda a: sc[a])
    return RichResult(payload={
        "score": sc, "qbar": qb, "exploration": ex, "best": best,
        "sumn": sn, "k": k,
        "method": "MuZero pUCT selection (Schrittwieser et al. 2020 eq. 2)"})


muzero_efficient_exploration = mzpuct


def cheatsheet():
    return "agmuef: MuZero pUCT action selection with min-max normalised values."
