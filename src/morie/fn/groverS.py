# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Grover's quantum search, simulated on the two-dimensional invariant plane.

Grover (1996), "A fast quantum mechanical algorithm for database
search", STOC '96, pp. 212-219, doi:10.1145/237814.237866.  Starting
from the uniform superposition, each Grover iteration (oracle sign
flip followed by inversion about the mean) rotates the state by 2theta
in the plane spanned by the marked and unmarked uniform states, where
sin theta = sqrt(M / N).  After k iterations the probability of
measuring a marked item is

    P(k) = sin^2( (2k + 1) theta ),

maximised at k* = round( pi / (4 theta) - 1/2 ), which is O(sqrt(N/M))
queries.  The N = 4, M = 1 case gives theta = pi/6 and P(1) = 1
exactly: one query suffices, with certainty.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["grover_search"]


def grover_search(oracle, N):
    """Simulate Grover iterations for a 0/1 marking vector of length N."""
    mark = [int(v) for v in core.vec(oracle)]
    n = int(N)
    if n < 2:
        raise ValueError("grover_search: N must be at least 2")
    if len(mark) != n:
        raise ValueError("grover_search: oracle must have N entries")
    for v in mark:
        if v not in (0, 1):
            raise ValueError("grover_search: oracle entries must be 0 or 1")
    M = sum(mark)
    if M == 0 or M == n:
        raise ValueError("grover_search: need at least one marked and one unmarked item")
    theta = math.asin(math.sqrt(M / float(n)))
    kopt = int(math.floor(math.pi / (4.0 * theta) - 0.5 + 0.5))
    amp = [1.0 / math.sqrt(n)] * n
    probs = [sum(amp[i] * amp[i] for i in range(n) if mark[i] == 1)]
    for _ in range(max(kopt, 1)):
        amp = [-amp[i] if mark[i] == 1 else amp[i] for i in range(n)]
        mean = sum(amp) / n
        amp = [2.0 * mean - amp[i] for i in range(n)]
        probs.append(sum(amp[i] * amp[i] for i in range(n) if mark[i] == 1))
    closed = math.sin((2.0 * kopt + 1.0) * theta) ** 2
    return RichResult(
        title="Grover search",
        summary_lines=[("N", n), ("marked", M), ("iterations", kopt)],
        payload={
            "estimate": probs[kopt],
            "p_success": probs[kopt],
            "p_closed_form": closed,
            "p_path": probs,
            "k_opt": kopt,
            "theta": theta,
            "n": n,
            "method": "P(k) = sin^2((2k+1) theta), sin theta = sqrt(M/N), Grover (1996)",
        },
    )


def cheatsheet():
    return "groverS: Grover quantum search"


# compact alias per ledger/NAMING.md
groversearch = grover_search
