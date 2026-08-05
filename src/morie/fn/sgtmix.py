# morie.fn -- function file (rootcoder007/morie)
"""Mixing time of a random walk from the absolute spectral gap."""

import math

from . import _s03core as core
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["sgt_mixing_time"]


def sgt_mixing_time(A, epsilon=0.01):
    """Relaxation-time bound on the mixing time of the walk on ``A``.

    The walk matrix ``P = D^-1 A`` is not symmetric, but it is similar to
    ``D^-1/2 A D^-1/2``, so its spectrum is real and can be taken from a
    symmetric eigenproblem.  The bound uses the ABSOLUTE spectral gap
    ``gamma* = 1 - max_{i>=2} |lambda_i|``, not ``1 - lambda_2``: a
    bipartite graph has ``lambda_n = -1`` and never mixes, which the
    signed gap would miss entirely.

    Formula: ``tau_mix(eps) <= log(1 / eps) / gamma*``.

    Parameters
    ----------
    A : array-like, shape (n, n)
        Symmetric non-negative adjacency; every row sum must be positive.
    epsilon : float, default 0.01
        Total-variation target.

    Returns
    -------
    RichResult
        ``tau_mix``, ``estimate``, ``gap`` (``gamma*``), ``slem``
        (the second-largest eigenvalue modulus), ``n``.

    References
    ----------
    Levin, D. A., Peres, Y. & Wilmer, E. L. (2017).  Markov Chains and
    Mixing Times, 2nd edition, American Mathematical Society;
    Theorem 12.4 gives ``t_mix(eps) <= log(1 / (eps pi_min)) / gamma*``,
    of which the form used here is the relaxation-time factor.
    """
    M = C.mat(A)
    n = len(M)
    if n == 0:
        raise ValueError("sgt_mixing_time: adjacency matrix is empty")
    for r in M:
        if len(r) != n:
            raise ValueError("sgt_mixing_time: adjacency matrix must be square")
    eps = float(epsilon)
    if not (0.0 < eps < 1.0):
        raise ValueError("sgt_mixing_time: epsilon must lie in (0, 1)")
    d = []
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += M[i][j]
        if s <= 0.0:
            raise ValueError("sgt_mixing_time: every node must have positive degree")
        d.append(s)
    S = [[M[i][j] / math.sqrt(d[i] * d[j]) for j in range(n)] for i in range(n)]
    vals, _ = core.jacobi(S)
    slem = 0.0
    for i in range(n - 1):
        a = abs(vals[i])
        if a > slem:
            slem = a
    gap = 1.0 - slem
    tau = math.log(1.0 / eps) / gap if gap > 0.0 else float("inf")
    return RichResult(payload={
        "tau_mix": tau, "estimate": tau, "gap": gap, "slem": slem, "n": n,
        "method": "Relaxation-time mixing bound, log(1/eps)/gamma*"})


def cheatsheet():
    return "sgtmix: Mixing time from the absolute spectral gap"
