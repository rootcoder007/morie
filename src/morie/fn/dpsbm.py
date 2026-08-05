# morie.fn -- function file (rootcoder007/morie)
"""Dirichlet-process stochastic block model."""

import math

from . import _array_core as np
from . import _s03core as core
from ._richresult import RichResult

__all__ = ["dp_stochastic_block"]


def _log_beta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def dp_stochastic_block(adjacency, alpha=1.0, n_iter=30, seed=42):
    """
    Dirichlet-process stochastic block model

    Formula: DP prior on community labels; Bernoulli edges

    The block-to-block edge probabilities carry independent Beta(1,1)
    priors and are integrated out, so a block pair (r, s) with e edges
    out of t possible contributes B(1 + e, 1 + t - e) to the collapsed
    likelihood.  A node is then reassigned by the CRP prior times the
    change in that likelihood, with the number of blocks learned rather
    than fixed.

    Parameters
    ----------
    adjacency : array-like
        n x n symmetric 0/1 adjacency matrix, diagonal ignored.
    alpha : float
        Concentration, strictly positive.
    n_iter : int
        Number of sweeps.
    seed : int
        Seed of the deterministic stream.

    Returns
    -------
    result : dict
        Keys: estimate (number of blocks), z, counts, n_blocks,
        log_likelihood, n.

    References
    ----------
    Kemp, Tenenbaum, Griffiths, Yamada & Ueda (2006), AAAI-06:381-388.
    """
    A = core.mat(adjacency)
    n = len(A)
    if n == 0:
        raise ValueError("empty input: adjacency has no rows")
    if any(len(r) != n for r in A):
        raise ValueError("adjacency must be square")
    if not (alpha > 0.0):
        raise ValueError("alpha must be strictly positive")

    def block_ll(z, K):
        e = [[0.0] * K for _ in range(K)]
        t = [[0.0] * K for _ in range(K)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                a, b = z[i], z[j]
                t[a][b] += 1.0
                e[a][b] += 1.0 if A[i][j] > 0.5 else 0.0
        s = 0.0
        for r in range(K):
            for c in range(K):
                if t[r][c] > 0.0:
                    s += _log_beta(1.0 + e[r][c], 1.0 + t[r][c] - e[r][c])
        return s

    rng = np.random.default_rng(seed)
    z = [0] * n
    K = 1
    for _ in range(int(n_iter)):
        for i in range(n):
            best = None
            logw = []
            cand = []
            counts = [sum(1 for v in range(n) if v != i and z[v] == c)
                      for c in range(K)]
            for c in range(K):
                if counts[c] == 0:
                    continue
                zz = list(z)
                zz[c] = zz[c]
                zz[i] = c
                logw.append(math.log(counts[c]) + block_ll(zz, K))
                cand.append(c)
            zz = list(z)
            zz[i] = K
            logw.append(math.log(alpha) + block_ll(zz, K + 1))
            cand.append(K)
            mx = max(logw)
            w = [math.exp(v - mx) for v in logw]
            tot = sum(w)
            u = float(rng.uniform(0.0, 1.0)) * tot
            acc = 0.0
            pick = cand[-1]
            for q in range(len(w)):
                acc += w[q]
                if u <= acc:
                    pick = cand[q]
                    break
            z[i] = pick
            if pick == K:
                K += 1
            used = []
            for v in z:
                if v not in used:
                    used.append(v)
            used.sort()
            remap = dict((c, j) for j, c in enumerate(used))
            z = [remap[v] for v in z]
            K = len(used)
    counts = [sum(1 for v in z if v == c) for c in range(K)]
    return RichResult(payload={
        "estimate": K,
        "z": z,
        "counts": counts,
        "n_blocks": K,
        "log_likelihood": block_ll(z, K),
        "n": n,
        "method": "Dirichlet-process stochastic block model",
    })


def cheatsheet():
    return "dpsbm: Dirichlet-process stochastic block model"


# compact alias per ledger/NAMING.md
dpstochasticblock = dp_stochastic_block
