# morie.fn -- tail3 batch (rootcoder007/morie)
"""MWEM: multiplicative weights with the exponential mechanism.

Source consulted: Hardt, M., Ligett, K. & McSherry, F. (2012). A simple and
practical algorithm for differentially private data release.  NIPS 2012,
arXiv:1012.4763, Figure 1:

    A_0 = n times the uniform distribution over D
    for i = 1..T
      1. exponential mechanism: sample q_i from Q with parameter eps/2T and
         score s_i(B, q) = |q(A_{i-1}) - q(B)|
      2. laplace mechanism: m_i = q_i(B) + Lap(2T/eps)
      3. multiplicative weights:
         A_i(x) proportional to A_{i-1}(x) exp(q_i(x)(m_i - q_i(A_{i-1}))/2n)
    output A = avg_i A_i

Both random draws are supplied by the caller (``gumbel`` for the exponential
mechanism, realised by the Gumbel-max trick, and ``lap`` for the Laplace
noise) so that a run is reproducible and the R mirror follows it exactly.
Passing zeros gives the noiseless multiplicative-weights skeleton.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["mwem"]


def mwem(B, queries, eps=1.0, T=10, gumbel=None, lap=None):
    """Run MWEM and report the resulting query error.

    Parameters
    ----------
    B : array-like
        Histogram of the private data set over the universe D.
    queries : array-like
        Query matrix, one linear query per row, entries in [-1, 1].
    eps : float
        Total privacy budget.
    T : int
        Number of iterations.
    gumbel : array-like, optional
        ``T`` by ``|Q|`` standard Gumbel draws for the exponential mechanism.
        Zeros (the default) select the highest-scoring query outright.
    lap : array-like, optional
        ``T`` Laplace(1) draws; scaled internally by ``2T/eps``.

    Returns
    -------
    RichResult
        estimate (max query error), maxerr, meanerr, A, selected, T, n,
        method.

    References
    ----------
    Hardt, Ligett & McSherry (2012), arXiv:1012.4763, Figure 1.
    """
    b = np.atleast_1d(np.asarray(B, dtype=float)).ravel()
    Q = np.atleast_2d(np.asarray(queries, dtype=float))
    d = int(b.size)
    nq = int(Q.shape[0])
    n = float(np.sum(b))
    Ti = int(T)
    qb = [sum(float(Q[k, x]) * float(b[x]) for x in range(d)) for k in range(nq)]
    A = [n / d] * d
    acc = [0.0] * d
    selected = []
    scale = 2.0 * Ti / float(eps) if float(eps) > 0.0 else 0.0
    epsi = float(eps) / (2.0 * Ti) if Ti > 0 else 0.0
    for i in range(Ti):
        qa = [sum(float(Q[k, x]) * A[x] for x in range(d)) for k in range(nq)]
        score = [abs(qa[k] - qb[k]) for k in range(nq)]
        util = [epsi * score[k] / 2.0 for k in range(nq)]
        if gumbel is not None:
            G = np.atleast_2d(np.asarray(gumbel, dtype=float))
            util = [util[k] + float(G[i, k]) for k in range(nq)]
        pick = 0
        for k in range(1, nq):
            if util[k] > util[pick]:
                pick = k
        selected.append(pick)
        m = qb[pick]
        if lap is not None:
            lv = np.atleast_1d(np.asarray(lap, dtype=float)).ravel()
            m = m + scale * float(lv[i])
        diff = m - qa[pick]
        newA = [A[x] * float(np.exp(float(Q[pick, x]) * diff / (2.0 * n))) for x in range(d)]
        tot = sum(newA)
        A = [v * n / tot for v in newA] if tot > 0.0 else list(A)
        for x in range(d):
            acc[x] += A[x]
    out = [v / Ti for v in acc] if Ti > 0 else list(A)
    err = [abs(sum(float(Q[k, x]) * out[x] for x in range(d)) - qb[k]) for k in range(nq)]
    return RichResult(
        payload={
            "estimate": float(max(err)),
            "maxerr": float(max(err)),
            "meanerr": float(sum(err) / nq),
            "A": np.asarray(out, dtype=float),
            "selected": selected,
            "T": Ti,
            "nqueries": nq,
            "n": float(n),
            "method": "MWEM (Hardt, Ligett & McSherry 2012)",
        }
    )


# CANONICAL TEST
# >>> # a uniform histogram is already the MWEM starting point: zero error
# >>> B = [5.0, 5.0, 5.0, 5.0]
# >>> Q = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]]
# >>> r = mwem(B, Q, eps=1.0, T=2, gumbel=[[0.0, 0.0], [0.0, 0.0]], lap=[0.0, 0.0])
# >>> assert r["maxerr"] < 1e-12


def cheatsheet():
    return "mwem(B, queries, eps, T, gumbel, lap): MWEM private release."
