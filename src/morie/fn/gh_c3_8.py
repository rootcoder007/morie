# morie.fn -- function file (rootcoder007/morie)
"""Feasibility of a moment sequence as a prior on [0, 1]."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["momprior", "ghosal_moment_prior"]


def momprior(moments):
    """Check that a moment sequence is realisable by a measure on [0, 1].

    Ghosal and van der Vaart note that inducing a prior from a prior on
    the moment sequence is possible in principle but that "maintaining
    the necessary constraints in the prior specification linking
    various moments is difficult".  This is that constraint, made
    checkable: Hausdorff's condition that the alternating finite
    differences of the moment sequence are all non-negative, which is
    necessary AND sufficient for a measure on [0, 1] with those
    moments to exist.

    ``min_difference`` is returned rather than just a yes/no, because a
    sequence can be feasible by an arbitrarily small margin and a prior
    that wanders across the boundary is the failure mode being guarded
    against.

    Formula: m realisable on [0, 1] iff
             (-1)^k (Delta^k m)_j >= 0 for all j, k >= 0, where
             (Delta m)_j = m_{j+1} - m_j

    Parameters
    ----------
    moments : array-like
        m_0, m_1, ..., m_n with m_0 = 1.

    Returns
    -------
    RichResult
        ``feasible`` (1/0), ``min_difference``, ``n_violations``,
        ``order``, ``differences`` (the full triangle, row k holding
        (-1)^k Delta^k m).

    References
    ----------
    Ghosal & van der Vaart (2017), Fundamentals of Nonparametric
    Bayesian Inference, Section 3.4.4 (Construction through Moments),
    which states that on a bounded interval "the sequence of moments
    uniquely determines the probability measure", so a prior on the
    measure can be induced from one on the moments, and warns about the
    linking constraints.  That section gives NO formula; the
    realisability condition itself is Hausdorff (1921), Summationsmethoden
    und Momentfolgen I, Mathematische Zeitschrift 9, 74-109, and is
    cited to its own source.  Ghosal read from the copy of the book
    held in the corpus.
    """
    m = C.vec(moments)
    N = len(m)
    if N < 1:
        raise ValueError("at least the zeroth moment is required")
    if abs(m[0] - 1.0) > 1e-12:
        raise ValueError("m_0 must equal 1 for a probability measure")
    tri = [list(m)]
    cur = list(m)
    for k in range(1, N):
        cur = [cur[j] - cur[j + 1] for j in range(len(cur) - 1)]
        tri.append(list(cur))
    worst = math.inf
    bad = 0
    for row in tri:
        for v in row:
            if v < worst:
                worst = v
            if v < -1e-12:
                bad += 1
    return RichResult(payload={
        "feasible": 1.0 if bad == 0 else 0.0, "min_difference": worst,
        "n_violations": float(bad), "order": float(N - 1),
        "differences": tri,
        "method": "Hausdorff moment feasibility, Ghosal Section 3.4.4"})


ghosal_moment_prior = momprior


def cheatsheet():
    return "gh_c3_8: realisable on [0,1] iff all (-1)^k Delta^k m >= 0"
