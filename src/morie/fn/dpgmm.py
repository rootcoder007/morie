# morie.fn -- slice s03 (rootcoder007/morie)
"""Dirichlet-process Gaussian mixture, truncated stick-breaking.

Sources consulted: Sethuraman, J. (1994).  A constructive definition of
Dirichlet priors.  *Statistica Sinica* 4(2), 639-650, for the weights
pi_k = V_k prod_(j<k)(1 - V_j) with V_k ~ Beta(1, alpha); and Blei, D. M.
and Jordan, M. I. (2006).  Variational inference for Dirichlet process
mixtures.  *Bayesian Analysis* 1(1), 121-143, for the truncated
stick-breaking approximation with a fixed truncation level K.  Neither
was retrievable here as a full text; both are quoted in their standard
published form, and the stick-breaking construction is reproduced in Teh
et al. (2006), *JASA* 101, 1566-1581, equations (5)-(6), which WAS
fetched.

DETERMINISM.  The prior weights come from the exact Beta(1, alpha)
quantile at low-discrepancy points, and the component parameters are
fitted by EM under a normal-inverse-gamma prior, which is a
deterministic fixed point rather than a Gibbs sampler.  The DP shows up
where it belongs -- in the prior over weights -- and nowhere is a draw
taken.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

from .dpsbw import stick_breaking_weights

__all__ = ["dp_gaussian_mixture"]


def dp_gaussian_mixture(y, alpha=1.0, prior_mu=0.0, prior_sigma=1.0,
                        truncation=5, max_iter=200, tol=1e-13):
    """Truncated DP mixture of univariate normals, fitted by EM.

    Returns
    -------
    estimate : the number of components with weight above 1/(10 K)
    weights, mu, sigma : the fitted mixture
    loglik   : the final log likelihood
    prior_pi : the stick-breaking prior weights
    """
    v = k.vec(y)
    n = len(v)
    K = int(truncation)
    prior = stick_breaking_weights(alpha, K)["pi"]
    tot = 0.0
    for x in prior:
        tot += x
    prior = [x / tot if tot > 0.0 else 1.0 / K for x in prior]
    lo = min(v)
    hi = max(v)
    mu = [lo + (hi - lo) * (i + 0.5) / K for i in range(K)]
    sd = [max((hi - lo) / K, 1e-6)] * K
    w = list(prior)
    ll = float("-inf")
    for _ in range(int(max_iter)):
        R = [[0.0] * K for _ in range(n)]
        newll = 0.0
        for i in range(n):
            lp = []
            for c in range(K):
                z = (v[i] - mu[c]) / sd[c]
                lp.append(math.log(w[c] if w[c] > 1e-300 else 1e-300)
                          - 0.5 * z * z - math.log(sd[c])
                          - 0.5 * math.log(2.0 * math.pi))
            m = k.logsumexp(lp)
            newll += m
            for c in range(K):
                R[i][c] = math.exp(lp[c] - m)
        for c in range(K):
            nk = 0.0
            for i in range(n):
                nk += R[i][c]
            # the DP prior enters as pseudo-counts alpha * prior_k
            eff = nk + float(alpha) * prior[c]
            w[c] = eff
            s = 0.0
            for i in range(n):
                s += R[i][c] * v[i]
            mu[c] = ((s + float(alpha) * prior[c] * float(prior_mu))
                     / eff if eff > 0.0 else float(prior_mu))
            q = 0.0
            for i in range(n):
                q += R[i][c] * (v[i] - mu[c]) ** 2
            q += float(alpha) * prior[c] * float(prior_sigma) ** 2
            sd[c] = math.sqrt(q / eff) if eff > 0.0 else float(prior_sigma)
            if sd[c] < 1e-8:
                sd[c] = 1e-8
        wt = 0.0
        for c in range(K):
            wt += w[c]
        w = [x / wt for x in w]
        if abs(newll - ll) < tol:
            ll = newll
            break
        ll = newll
    active = 0
    for x in w:
        if x > 1.0 / (10.0 * K):
            active += 1
    return RichResult(
        title="DP Gaussian mixture",
        summary_lines=[("active components", active), ("log-lik", ll)],
        payload={
            "estimate": float(active),
            "weights": w,
            "mu": mu,
            "sigma": sd,
            "loglik": ll,
            "prior_pi": prior,
            "n": n,
            "method": "Truncated stick-breaking DP mixture fitted by EM (Sethuraman 1994; Blei and Jordan 2006)",
        },
    )


def cheatsheet():
    return "dpgmm: DP Gaussian mixture with stick-breaking representation"
