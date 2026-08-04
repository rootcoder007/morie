# morie.fn -- slice s03 (rootcoder007/morie)
"""Dirichlet exploration noise at the MCTS root.

Source consulted (FETCHED): Silver, D. et al. (2018), A general
reinforcement learning algorithm that masters chess, shogi and Go
through self-play, arXiv:1712.01815, which states that "Dirichlet noise
Dir(alpha) was added to the prior probabilities in the root node".  The
mixture itself is written out in Silver et al. (2017), *Nature* 550,
354-359, and reproduced in Schrittwieser et al. (2020),
arXiv:1911.08265 (FETCHED), appendix C:

    P(s,a) = (1 - eps) p_a + eps eta_a,   eta ~ Dir(alpha)

with eps = 0.25 and alpha = {0.3, 0.15, 0.03} for chess, shogi and Go.

DETERMINISM.  A random Dirichlet draw would make the two arms disagree,
so ``eta`` is either supplied by the caller or built deterministically:
a Dirichlet(alpha, ..., alpha) vector is generated as normalised
Gamma(alpha) variates, and the Gamma variates are obtained by inverting
the Gamma CDF at van der Corput points rather than by rejection
sampling.  Nothing here consults a clock or a seed.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["alphazero_dirichlet_noise"]


def _gamma_lower_reg(a, x, iters=400):
    """Regularised lower incomplete gamma P(a, x), by its series."""
    if x <= 0.0:
        return 0.0
    if x < a + 1.0:
        term = 1.0 / a
        s = term
        for n in range(1, iters):
            term *= x / (a + n)
            s += term
            if abs(term) < abs(s) * 1e-16:
                break
        return s * math.exp(-x + a * math.log(x) - math.lgamma(a))
    b = x + 1.0 - a
    c = 1e300
    d = 1.0 / b
    h = d
    for i in range(1, iters):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-16:
            break
    return 1.0 - math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def _gamma_quantile(a, p):
    """Invert P(a, x) = p by bisection -- monotone, so this is exact to 1e-13."""
    lo, hi = 0.0, 1.0
    while _gamma_lower_reg(a, hi) < p and hi < 1e8:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _gamma_lower_reg(a, mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def alphazero_dirichlet_noise(p, alpha=0.3, eps=0.25, eta=None):
    """Mix Dirichlet noise into the root priors.

    Parameters
    ----------
    p : array-like
        Root priors from the policy head.
    alpha : float
        Dirichlet concentration; 0.3 chess, 0.15 shogi, 0.03 Go.
    eps : float
        Mixing weight; AlphaZero uses 0.25.
    eta : array-like, optional
        A Dirichlet vector supplied by the caller.  When omitted a
        deterministic one is constructed (see the module docstring).

    Returns
    -------
    RichResult with payload:
        estimate : the mixed prior of action 0
        p_noisy  : the full mixed prior vector
        eta      : the Dirichlet vector actually used
        entropy  : Shannon entropy of the mixed prior, in nats
    """
    pr = k.vec(p)
    m = len(pr)
    a = float(alpha)
    e = float(eps)
    if eta is None:
        raw = [_gamma_quantile(a, k.vdc(i, 2)) for i in range(m)]
        tot = 0.0
        for x in raw:
            tot += x
        et = [x / tot if tot > 0.0 else 1.0 / m for x in raw]
    else:
        et = k.vec(eta)
        tot = 0.0
        for x in et:
            tot += x
        if tot > 0.0:
            et = [x / tot for x in et]
    mixed = [(1.0 - e) * pr[i] + e * et[i] for i in range(m)]
    h = 0.0
    for x in mixed:
        if x > 0.0:
            h -= x * math.log(x)
    return RichResult(
        title="Dirichlet root noise",
        summary_lines=[("alpha", a), ("eps", e)],
        payload={
            "estimate": mixed[0] if mixed else float("nan"),
            "p_noisy": mixed,
            "eta": et,
            "entropy": h,
            "alpha": a,
            "eps": e,
            "method": "Dirichlet exploration noise at the MCTS root",
        },
    )


def cheatsheet():
    return "agdirn: Dirichlet exploration noise at MCTS root"
