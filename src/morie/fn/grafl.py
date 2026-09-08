# morie.fn -- slice s03 (rootcoder007/morie)
"""Graphlet kernel.

Source consulted: Shervashidze, N., Vishwanathan, S. V. N., Petri, T.,
Mehlhorn, K. and Borgwardt, K. M. (2009).  Efficient graphlet kernels
for large graph comparison.  *AISTATS* 5, 488-495, which defines the
graphlet kernel as the inner product of the *normalised* counts of all
size-k induced subgraphs,

    k(G, G') = < f_G , f_G' >,   f_G = (# of graphlet type i in G) / C(n, k)

Their companion paper Shervashidze et al. (2011), *JMLR* 12, 2539-2561
(FETCHED) restates the same construction as its "third class" of graph
kernels.  The 2009 AISTATS volume was not retrievable here; the kernel
is quoted in its standard published form.

Graphlet types are identified by a canonical signature -- the edge count
together with the sorted degree sequence -- which separates every
isomorphism class for k = 3 (4 types) and k = 4 (11 types), so no
isomorphism test is needed and the enumeration is exact.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["graphlet_kernel"]


def _sig(A, idx):
    m = 0
    deg = [0] * len(idx)
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            if A[idx[a]][idx[b]] != 0.0:
                m += 1
                deg[a] += 1
                deg[b] += 1
    return "%d:%s" % (m, ",".join([str(x) for x in sorted(deg)]))


def _counts(G, kk, types):
    A = k.mat(G)
    n = len(A)
    c = {}
    idx = list(range(kk))
    while True:
        s = _sig(A, idx)
        if s not in types:
            types.append(s)
        c[s] = c.get(s, 0.0) + 1.0
        i = kk - 1
        while i >= 0 and idx[i] == n - kk + i:
            i -= 1
        if i < 0:
            break
        idx[i] += 1
        for j in range(i + 1, kk):
            idx[j] = idx[j - 1] + 1
    tot = math.comb(n, kk) if n >= kk else 0
    return c, float(tot)


def graphlet_kernel(G1, G2, k_size=3, normalize=True):
    """Inner product of the size-k graphlet frequency vectors.

    Returns
    -------
    RichResult with payload:
        estimate : the kernel value
        types    : the canonical signatures encountered, in order
        f1, f2   : the two frequency vectors
    """
    types = []
    c1, t1 = _counts(G1, int(k_size), types)
    c2, t2 = _counts(G2, int(k_size), types)
    f1 = [(c1.get(s, 0.0) / t1 if (normalize and t1 > 0.0) else c1.get(s, 0.0))
          for s in types]
    f2 = [(c2.get(s, 0.0) / t2 if (normalize and t2 > 0.0) else c2.get(s, 0.0))
          for s in types]
    dot = 0.0
    for i in range(len(types)):
        dot += f1[i] * f2[i]
    return RichResult(
        title="Graphlet kernel",
        summary_lines=[("k", dot), ("size", int(k_size))],
        payload={
            "estimate": dot,
            "types": types,
            "f1": f1,
            "f2": f2,
            "n_types": len(types),
            "method": "Graphlet kernel on size-k induced subgraphs (Shervashidze et al. 2009)",
        },
    )


def cheatsheet():
    return "grafl: Graphlet kernel"


graphletkernel = graphlet_kernel
