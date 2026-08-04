# morie.fn -- function file (rootcoder007/morie)
"""Blocked Gibbs via truncated stick breaking.

Implements eq. (5.7) + truncation sec. 5.3 setup of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_blk_gibbs"]


def _norm_pdf(x, mu, sd):
    z = (x - mu) / sd
    return math.exp(-0.5 * z * z) / (sd * math.sqrt(2.0 * math.pi))


def ghosal_blk_gibbs(data, K=10, alpha=1.0, tau=1.0, sigma=0.5,
                     n_sweeps=25, seed=42):
    """Truncated model (5.7): s_i | W ~ W, V_j ~ Be(1, M), theta_j ~
    G0; blocked Gibbs updates all K components jointly -- s | rest,
    V | s (Be(1 + n_j, M + n_{>j})), theta_j | rest (conjugate
    normal). Keys: estimate."""
    xs = _bnp._flat(data)
    n = len(xs)
    M = float(alpha)
    rng = np.random.default_rng(seed)
    th = [float(v) for v in rng.uniform(-1, 1, K)._flat()]
    W = [1.0 / K] * K
    s = [0] * n
    for _ in range(int(n_sweeps)):
        for i in range(n):
            wts = [W[j] * _norm_pdf(xs[i], th[j], sigma)
                   for j in range(K)]
            tot = sum(wts) or 1.0
            u = float(rng.uniform(0, 1)) * tot
            acc = 0.0
            s[i] = K - 1
            for j, wv in enumerate(wts):
                acc += wv
                if u <= acc:
                    s[i] = j
                    break
        counts = [sum(1 for v in s if v == j) for j in range(K)]
        V = []
        for j in range(K - 1):
            n_gt = sum(counts[j + 1:])
            V.append(float(rng.beta(1.0 + counts[j], M + n_gt)))
        V.append(1.0)
        W = _bnp.stick_breaking(V)
        for j in range(K):
            members = [xs[i] for i in range(n) if s[i] == j]
            prec = len(members) / sigma ** 2 + 1.0 / tau ** 2
            mean = (sum(members) / sigma ** 2) / prec
            th[j] = mean + math.sqrt(1.0 / prec) \
                * float(rng.normal(0, 1))
    k_used = len(set(s))
    res = RichResult(payload={"estimate": float(k_used),
                              "n_active": k_used,
                              "weights": W, "atoms": th,
                              "method": "blocked Gibbs, truncated stick breaking (GvdV 2017 eq. 5.7)"})
    return with_describe_pointer(res, "gh_c5_5")


def cheatsheet():
    return "gh_c5_5: Blocked Gibbs via truncated stick breaking"


# compact alias per ledger/NAMING.md
ghosalblkgibbs = ghosal_blk_gibbs
