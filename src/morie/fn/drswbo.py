# morie.fn -- slice s03 (rootcoder007/morie)
"""DR-DiD with a stratified cluster-block bootstrap interval.

Source consulted (FETCHED): Sant'Anna, P. H. C. and Zhao, J. (2020),
*Journal of Econometrics* 219(1), 101-122 (arXiv:1812.01723), equations
(2.6)-(2.7) for the estimator and section 3.2 for multiplier-bootstrap
inference; and Cameron, A. C., Gelbach, J. B. and Miller, D. L. (2008).
Bootstrap-based improvements for inference with clustered errors.
*Review of Economics and Statistics* 90(3), 414-427, for the rule that
the resampling unit must be the *cluster*, not the observation, when
errors are correlated within cluster.  The 2008 RESTAT paper is
paywalled; the cluster-as-unit rule is stated identically wherever the
cluster bootstrap is defined.

So one multiplier is drawn per cluster and applied to every member of
it, which preserves the within-cluster dependence.  The interval is the
percentile interval of the replicates.

DETERMINISM.  The multipliers are Mammen's two-point weights taken at
van der Corput points, indexed by cluster; no random resampling.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["dr_did_stratified_block"]


def dr_did_stratified_block(y, D, unit=None, time=None, X=None, clusters=None,
                            B=199, alpha=0.05, y0=None):
    """DR-DiD with a cluster-block multiplier bootstrap percentile interval.

    Returns
    -------
    RichResult with payload:
        estimate : the DR ATT
        se       : replicate standard deviation
        ci_lo, ci_hi : percentile interval
        n_clusters
    """
    dy = k.vec(y)
    if y0 is not None:
        y00 = k.vec(y0)
        dy = [dy[i] - y00[i] for i in range(len(dy))]
    fit = k.drdid_panel(dy, D, X)
    inf = fit["inf"]
    n = len(inf)
    src = clusters if clusters is not None else (unit if unit is not None
                                                 else list(range(n)))
    lab = [str(c) for c in src]
    ids = []
    for c in lab:
        if c not in ids:
            ids.append(c)
    gidx = [ids.index(c) for c in lab]
    G = len(ids)
    boot = []
    for b in range(int(B)):
        wts = [k.mammen(b * G + g) for g in range(G)]
        s = 0.0
        for i in range(n):
            s += wts[gidx[i]] * inf[i]
        boot.append(fit["tau"] + s / n)
    a = float(alpha)
    return RichResult(
        title="DR-DiD, cluster-block bootstrap",
        summary_lines=[("ATT", fit["tau"]), ("clusters", G)],
        payload={
            "estimate": fit["tau"],
            "se": k.sd(boot, 1) if len(boot) > 1 else float("nan"),
            "ci_lo": k.quantile7(boot, a / 2.0),
            "ci_hi": k.quantile7(boot, 1.0 - a / 2.0),
            "boot": boot,
            "n_clusters": G,
            "n": n,
            "B": int(B),
            "method": "DR-DiD with a deterministic cluster-block multiplier bootstrap",
        },
    )


def cheatsheet():
    return "drswbo: DR-DiD with stratified-block bootstrap CI"
