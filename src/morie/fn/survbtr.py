# morie.fn -- slice s03 (rootcoder007/morie)
"""BART for survival outcomes.

Source consulted: Sparapani, R. A., Logan, B. R., McCulloch, R. E. and
Laud, P. W. (2016).  Nonparametric survival analysis using Bayesian
additive regression trees (BART).  *Statistics in Medicine* 35(16),
2741-2753, and Chipman, H. A., George, E. I. and McCulloch, R. E.
(2010).  BART: Bayesian additive regression trees.  *The Annals of
Applied Statistics* 4(1), 266-298.  Sparapani et al.'s device is to
recast survival as a sequence of binary events on a person-period grid,

    p_ij = P(T = t_j | T >= t_j, x_i) = Phi( mu + f(t_j, x_i) )

with f a sum of regression trees, so that BART is applied to the
discrete hazard rather than to the time itself.  Both papers are
paywalled; the person-period recasting and the probit link are quoted in
their standard published form.

DETERMINISM.  Chipman et al.'s backfitting MCMC is replaced by
*boosted* backfitting: trees are grown greedily on the working residual
and shrunk by the same factor the prior would impose.  That is a
deterministic sum-of-trees fit, not a posterior sample, and the method
string says so -- no credible intervals are claimed.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["bart_survival"]


def _stump(X, r, w):
    """Best weighted single-split regression stump over all columns."""
    n = len(X)
    p = len(X[0]) if n else 0
    best = None
    for a in range(p):
        vals = sorted(set([X[i][a] for i in range(n)]))
        for t in range(len(vals) - 1):
            thr = 0.5 * (vals[t] + vals[t + 1])
            sl = wl = sr = wr = 0.0
            for i in range(n):
                if X[i][a] <= thr:
                    sl += w[i] * r[i]
                    wl += w[i]
                else:
                    sr += w[i] * r[i]
                    wr += w[i]
            if wl <= 0.0 or wr <= 0.0:
                continue
            ml = sl / wl
            mr = sr / wr
            gain = wl * ml * ml + wr * mr * mr
            if best is None or gain > best[0]:
                best = (gain, a, thr, ml, mr)
    return best


def bart_survival(time, event, X=None, n_trees=5, shrink=0.3, grid=None):
    """Sum-of-trees fit to the discrete hazard on a person-period grid.

    Returns
    -------
    estimate : the fitted hazard of the first person-period cell
    hazard   : fitted hazard per person-period row
    surv     : fitted survival per subject at the last grid time
    trees    : the (column, threshold, left, right) of each stump
    """
    t = k.vec(time)
    e = k.vec(event)
    n = len(t)
    Xr = k.mat(X) if X is not None else [[0.0] for _ in range(n)]
    g = sorted(set(k.vec(grid))) if grid is not None else sorted(
        set([t[i] for i in range(n) if e[i] > 0.5]))
    rows = []
    ys = []
    who = []
    for i in range(n):
        for j in range(len(g)):
            if g[j] > t[i]:
                break
            rows.append([g[j]] + list(Xr[i]))
            ys.append(1.0 if (abs(g[j] - t[i]) < 1e-12 and e[i] > 0.5) else 0.0)
            who.append(i)
    m = len(rows)
    pbar = k.mean(ys) if m else 0.5
    if pbar <= 0.0:
        pbar = 0.5 / m
    if pbar >= 1.0:
        pbar = 1.0 - 0.5 / m
    f = [k.qnorm(pbar)] * m
    trees = []
    for _ in range(int(n_trees)):
        r = []
        w = []
        for i in range(m):
            p = k.pnorm(f[i])
            p = min(max(p, 1e-8), 1.0 - 1e-8)
            dens = math.exp(-0.5 * f[i] * f[i]) / math.sqrt(2.0 * math.pi)
            grad = (ys[i] - p) * dens / (p * (1.0 - p))
            hess = dens * dens / (p * (1.0 - p))
            r.append(grad / hess if hess > 0.0 else 0.0)
            w.append(hess)
        st = _stump(rows, r, w)
        if st is None:
            break
        _, a, thr, ml, mr = st
        trees.append([float(a), thr, ml, mr])
        for i in range(m):
            f[i] += float(shrink) * (ml if rows[i][a] <= thr else mr)
    haz = [k.pnorm(v) for v in f]
    surv = [1.0] * n
    for i in range(m):
        surv[who[i]] *= (1.0 - haz[i])
    return RichResult(
        title="BART survival",
        summary_lines=[("person-periods", m), ("trees", len(trees))],
        payload={
            "estimate": haz[0] if haz else float("nan"),
            "hazard": haz,
            "surv": surv,
            "trees": trees,
            "grid": g,
            "n": n,
            "method": ("Person-period probit hazard with a boosted sum of stumps "
                       "(Sparapani et al. 2016 recasting; deterministic fit, not a posterior sample)"),
        },
    )


def cheatsheet():
    return "survbtr: BART for survival outcomes"


bartsurvival = bart_survival
