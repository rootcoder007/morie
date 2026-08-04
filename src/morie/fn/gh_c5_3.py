# morie.fn -- function file (rootcoder007/morie)
"""Collapsed Gibbs sampler for DPM.

Implements Theorem 5.3, eq. (5.4)-(5.6); Algorithm 3 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_cgibbs"]


def _norm_pdf(x, mu, sd):
    z = (x - mu) / sd
    return math.exp(-0.5 * z * z) / (sd * math.sqrt(2.0 * math.pi))


def ghosal_cgibbs(data, alpha=1.0, tau=1.0, sigma=0.5, n_sweeps=30,
                  seed=42):
    """P(s_i = s | rest) proportional to N_{-i,s} psi(X_i; mu_s) for
    an existing cluster and to M int psi(X_i; theta) dG0(theta) for a
    new one (Thm 5.3 / Algorithm 3, conjugate normal-normal so the
    integral is N(0, tau^2 + sigma^2)). Runs Gibbs sweeps and reports
    the cluster count. Keys: estimate."""
    xs = _bnp._flat(data)
    n = len(xs)
    M = float(alpha)
    s_marg = math.sqrt(tau * tau + sigma * sigma)
    rng = np.random.default_rng(seed)
    z = list(range(n))                     # start: singletons
    for _ in range(int(n_sweeps)):
        for i in range(n):
            labels = sorted(set(z[j] for j in range(n) if j != i))
            wts = []
            for lab in labels:
                members = [xs[j] for j in range(n)
                           if j != i and z[j] == lab]
                # cluster scored at its current mean (Alg 2 flavor
                # with mu_s at the posterior mean of members)
                mprec = len(members) / sigma ** 2 + 1.0 / tau ** 2
                mmean = (sum(members) / sigma ** 2) / mprec
                wts.append(len(members)
                           * _norm_pdf(xs[i], mmean, sigma))
            wts.append(M * _norm_pdf(xs[i], 0.0, s_marg))
            tot = sum(wts)
            u = float(rng.uniform(0, 1)) * tot
            acc = 0.0
            pick = len(wts) - 1
            for k, wv in enumerate(wts):
                acc += wv
                if u <= acc:
                    pick = k
                    break
            z[i] = labels[pick] if pick < len(labels) \
                else (max(z) + 1 + i)
    k = len(set(z))
    res = RichResult(payload={"estimate": float(k),
                              "n_clusters": k, "labels": z,
                              "method": "collapsed Gibbs for DPM (GvdV 2017 Thm 5.3/Alg 3)"})
    return with_describe_pointer(res, "gh_c5_3")


def cheatsheet():
    return "gh_c5_3: Collapsed Gibbs sampler for DPM"


# compact alias per ledger/NAMING.md
ghosalcgibbs = ghosal_cgibbs
