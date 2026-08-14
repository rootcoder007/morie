# morie.fn -- function file (rootcoder007/morie)
"""Distribution of a simple random walk after a fixed number of steps."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["random_walk"]


def random_walk(G, start=0, steps=1):
    """Exact law of the walk, propagated rather than simulated.

    Simulating the walk would make the two language arms agree only in
    distribution.  The transition law is a linear map, so the whole
    distribution can be pushed forward exactly -- same numbers in both
    arms, and no Monte Carlo error to argue about.

    Formula: ``P(j | i) = A_ij / k_i``; the returned vector is
    ``e_start P^steps``.

    Parameters
    ----------
    G : array-like, shape (n, n)
        Non-negative weight matrix; every row sum must be positive.
    start : int, default 0
        Starting node, 0-based (the R arm takes it 1-based and reports
        0-based indices back, as everywhere else in this package).
    steps : int, default 1
        Number of steps, non-negative.

    Returns
    -------
    RichResult
        ``p`` (the distribution), ``estimate`` (its largest entry),
        ``argmax`` (0-based), ``p_start`` (the return probability),
        ``n``.

    References
    ----------
    Lovasz, L. (1996).  Random walks on graphs: a survey.  In
    Combinatorics, Paul Erdos is Eighty, Vol. 2, pages 1-46, Janos
    Bolyai Mathematical Society, Budapest.
    """
    M = C.mat(G)
    n = len(M)
    if n == 0:
        raise ValueError("random_walk: graph is empty")
    for r in M:
        if len(r) != n:
            raise ValueError("random_walk: graph must be square")
    start = int(start)
    steps = int(steps)
    if start < 0 or start >= n:
        raise ValueError("random_walk: start is outside the graph")
    if steps < 0:
        raise ValueError("random_walk: steps must be non-negative")
    d = []
    for i in range(n):
        s = 0.0
        for j in range(n):
            if M[i][j] < 0.0:
                raise ValueError("random_walk: weights must be non-negative")
            s += M[i][j]
        if s <= 0.0:
            raise ValueError("random_walk: every node must have positive degree")
        d.append(s)
    p = [0.0] * n
    p[start] = 1.0
    for _ in range(steps):
        q = [0.0] * n
        for i in range(n):
            if p[i] != 0.0:
                for j in range(n):
                    q[j] += p[i] * M[i][j] / d[i]
        p = q
    am = 0
    for i in range(n):
        if p[i] > p[am]:
            am = i
    return RichResult(payload={
        "p": p, "estimate": p[am], "argmax": am, "p_start": p[start], "n": n,
        "method": "Exact random-walk law e_start P^steps"})


def cheatsheet():
    return "randwk: Random walk on a network"

# public names resolved by fn/_lazy_map.json
randomwalk = random_walk
