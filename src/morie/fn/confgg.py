# morie.fn -- function file (rootcoder007/morie)
"""Configuration model with a prescribed degree sequence."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['configmodel', 'configuration_model']


def configmodel(degrees, seed=1):
    """Configuration model with a prescribed degree sequence.

    The realised graph is a multigraph: self-loops and repeated edges occur and are reported rather than removed, because deleting them is what silently changes the degree sequence you asked for. Stub pairing uses the shared minstd stream so both language arms build the same graph; the degree sum must be even, which is checked rather than quietly corrected.


    Formula: attach half-edges (stubs) uniformly at random in pairs; every vertex ends with its requested degree

    Parameters
    ----------
    degrees : array-like
        Requested degree for each vertex.
    seed : int
        Seed of the shared minstd stream.

    Returns
    -------
    RichResult
        ``edges``, ``degree`` (realised), ``self_loops``, ``multi_edges``, ``n``.

    References
    ----------
    Bender and Canfield (1978), The asymptotic number of labeled graphs
    with given degree sequences, JCTA 24:296-307; Molloy and Reed (1995),
    A critical point for random graphs with a given degree sequence,
    Random Structures and Algorithms 6:161-180.  Neither is held
    locally; uniform stub pairing is the standard published construction.
    """
    d = [int(v) for v in C.vec(degrees)]
    n = len(d)
    if any(v < 0 for v in d):
        raise ValueError("degrees must be non-negative")
    if sum(d) % 2 != 0:
        raise ValueError("sum of degrees must be even")
    stubs = []
    for i in range(n):
        stubs.extend([i] * d[i])
    g = C.Lcg(seed)
    edges, loops = [], 0
    while stubs:
        i = int(g.unif() * len(stubs))
        if i >= len(stubs):
            i = len(stubs) - 1
        a = stubs.pop(i)
        j = int(g.unif() * len(stubs))
        if j >= len(stubs):
            j = len(stubs) - 1
        b = stubs.pop(j)
        if a == b:
            loops += 1
        edges.append((a, b) if a <= b else (b, a))
    seen, multi = {}, 0
    for e in edges:
        seen[e] = seen.get(e, 0) + 1
    for e, k in seen.items():
        if e[0] != e[1] and k > 1:
            multi += k - 1
    real = [0] * n
    for a, b in edges:
        real[a] += 1
        real[b] += 1
    return RichResult(payload={
        "edges": edges, "degree": real, "self_loops": loops,
        "multi_edges": multi, "n": n,
        "method": "Configuration model (uniform stub pairing)"})


configuration_model = configmodel


def cheatsheet():
    return "confgg: Configuration model with a prescribed degree sequence."
