# morie.fn -- function file (rootcoder007/morie)
"""Criterion-function set estimate for moment inequalities."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["qcritset", "bound_intersection"]


def qcritset(mbar, se=None, n=1, cutoff=None):
    """Chernozhukov-Hong-Tamer criterion and its level set.

    A parameter satisfying moment inequalities E m_j(theta) <= 0 for all
    j is identified only up to a set.  Chernozhukov, Hong and Tamer turn
    that set into the argmin of a criterion which penalises violations
    and ignores slack,

        Q_n(theta) = sum_j [ max( mbar_j(theta) / sigma_j , 0 ) ]^2,

    so Q_n is zero exactly where every inequality holds and positive
    elsewhere.  The set estimate is the level set

        C_n = { theta : n Q_n(theta) <= cutoff },

    the cutoff coming from the asymptotic distribution or a subsampled
    quantile; it is supplied here rather than simulated, which keeps the
    routine free of a random stream.

    Parameters
    ----------
    mbar : array-like, shape (g, J)
        Sample moment means, one row per candidate theta on a grid.
    se : array-like or None
        Moment standard deviations sigma_j, same shape or length J.
        ``None`` leaves the moments unstandardised.
    n : int
        Sample size behind the moment means, used for the n Q_n scaling.
    cutoff : float or None
        Level-set cutoff; ``None`` uses min(n Q_n) so the reported set is
        the argmin.

    Returns
    -------
    RichResult
        ``Q``, ``nQ``, ``argmin``, ``minQ``, ``inset``, ``nin``,
        ``cutoff``, ``g``, ``J``.

    References
    ----------
    Chernozhukov, V., Hong, H. and Tamer, E. (2007), "Estimation and
    confidence regions for parameter sets in econometric models",
    Econometrica 75(5), 1243-1284, whose Sect. 2 defines the criterion
    Q_n(theta) as the sum of squared positive parts of the standardised
    moment violations and the set estimate as its level set at a
    data-dependent cutoff.  Standard published form; the Econometrica
    article was not in the local corpus and was not read for this
    implementation.
    """
    M = C.mat(mbar)
    g, J = len(M), len(M[0])
    if se is None:
        S = [[1.0] * J for _ in range(g)]
    else:
        sv = C.mat(se)
        if len(sv) == 1 and len(sv[0]) == J:
            S = [sv[0][:] for _ in range(g)]
        elif len(sv) == g and len(sv[0]) == J:
            S = sv
        else:
            raise ValueError("se must be length J or the shape of mbar")
    if any(v <= 0.0 for r in S for v in r):
        raise ValueError("standard deviations must be strictly positive")
    n = float(n)
    if n <= 0.0:
        raise ValueError("n must be positive")
    Q = []
    for i in range(g):
        Q.append(sum(max(M[i][j] / S[i][j], 0.0) ** 2 for j in range(J)))
    nQ = [n * v for v in Q]
    mn = min(nQ)
    am = nQ.index(mn)
    cut = mn if cutoff is None else float(cutoff)
    ins = [1 if nQ[i] <= cut else 0 for i in range(g)]
    return RichResult(payload={
        "Q": Q, "nQ": nQ, "argmin": am, "minQ": mn, "inset": ins,
        "nin": sum(ins), "cutoff": cut, "g": g, "J": J,
        "method": "Criterion-function set estimate (Chernozhukov-Hong-Tamer 2007)"})


bound_intersection = qcritset


def cheatsheet():
    return "bdintp: Criterion-function set estimate for moment inequalities."
