# morie.fn -- function file (rootcoder007/morie)
"""Channel capacity by the Blahut-Arimoto algorithm."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['chancap', 'channel_capacity']


def chancap(P, iters=200):
    """Channel capacity by the Blahut-Arimoto algorithm.

    Capacity is a maximisation over input distributions with no closed form in general, but the objective is concave and the alternating update is a coordinate ascent on it, so the iterate increases monotonically. The iteration count is fixed rather than tolerance-driven so both language arms take identical paths; the returned mutual information per iteration lets a caller see the monotone increase and judge whether the budget was enough.


    Formula: q(x|y) = r(x) P(y|x) / sum_x' r(x') P(y|x'); r(x) <- exp(sum_y P(y|x) log q(x|y)) normalised; C = max_r I(X;Y)

    Parameters
    ----------
    P : array-like, shape (m, n)
        Channel matrix; row x is the output distribution given input x.
    iters : int
        Fixed number of alternations.

    Returns
    -------
    RichResult
        ``capacity_bits``, ``capacity_nats``, ``input_dist``, ``trace`` (nats per iteration), ``iterations``.

    References
    ----------
    Blahut (1972), Computation of channel capacity and rate-distortion
    functions, IEEE Transactions on Information Theory 18:460-473;
    Arimoto (1972), same volume, 14-20.  Neither is held locally; the
    alternating update is the standard published form of the algorithm.
    """
    P = C.mat(P)
    m = len(P); n = len(P[0])
    for row in P:
        if any(v < 0 for v in row):
            raise ValueError("channel probabilities must be non-negative")
        if abs(sum(row) - 1.0) > 1e-9:
            raise ValueError("each row of P must sum to 1")
    r = [1.0 / m] * m
    trace = []
    for _ in range(int(iters)):
        qy = [sum(r[i] * P[i][j] for i in range(m)) for j in range(n)]
        logr = []
        for i in range(m):
            s = 0.0
            for j in range(n):
                if P[i][j] > 0 and qy[j] > 0:
                    s += P[i][j] * math.log(P[i][j] / qy[j])
            logr.append(s)
        mx = max(logr)
        w = [r[i] * math.exp(logr[i] - mx) for i in range(m)]
        tw = sum(w)
        r = [v / tw for v in w]
        trace.append(sum(r[i] * logr[i] for i in range(m)))
    qy = [sum(r[i] * P[i][j] for i in range(m)) for j in range(n)]
    cap = 0.0
    for i in range(m):
        for j in range(n):
            if P[i][j] > 0 and qy[j] > 0:
                cap += r[i] * P[i][j] * math.log(P[i][j] / qy[j])
    return RichResult(payload={
        "capacity_bits": cap / math.log(2.0), "capacity_nats": cap,
        "input_dist": r, "trace": trace, "iterations": int(iters),
        "method": "Channel capacity (Blahut-Arimoto)"})


channel_capacity = chancap


def cheatsheet():
    return "chnntp: Channel capacity by the Blahut-Arimoto algorithm."
