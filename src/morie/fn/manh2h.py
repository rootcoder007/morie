# morie.fn -- function file (rootcoder007/morie)
"""Node-splitting inconsistency check for one network edge."""

import math

from . import _macore as ma
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["ma_network_node_split"]


def ma_network_node_split(yi, vi, design, edge):
    """Ask whether the direct evidence on one comparison agrees with the rest.

    A network estimate is a weighted blend of what the head-to-head trials
    say and what the rest of the network implies.  If those two disagree
    the blend is meaningless, and no global fit statistic will show it,
    because the blend absorbs the disagreement.  Splitting the node
    computes them separately and tests the difference, which is the only
    way to localise inconsistency to an edge.

    Formula: ``direct`` is the inverse-variance pool of the studies making
    that comparison; ``indirect`` is the consistency-model estimate of the
    same contrast from all the other studies; the test is
    ``z = (direct - indirect)/sqrt(v_dir + v_ind)`` -- Dias et al. (2010)
    Section 3.

    Parameters
    ----------
    yi : array-like, shape (n,)
        Contrast estimates.
    vi : array-like, shape (n,)
        Their sampling variances, strictly positive.
    design : array-like, shape (n, 2)
        Baseline and comparator treatment labels per study.
    edge : array-like, shape (2,)
        The comparison to split, as (baseline, comparator).

    Returns
    -------
    RichResult
        ``direct``, ``v_direct``, ``indirect``, ``v_indirect``, ``diff``,
        ``z``, ``p``, ``k_direct``, ``k_indirect``.

    References
    ----------
    Dias, S., Welton, N. J., Caldwell, D. M. and Ades, A. E. (2010).
    Checking consistency in mixed treatment comparison meta-analysis.
    Statistics in Medicine 29(7-8):932-944.  doi:10.1002/sim.3767.
    """
    y = [float(t) for t in core.vec(yi)]
    v = [float(t) for t in core.vec(vi)]
    n = len(y)
    if n == 0:
        raise ValueError("no studies")
    if len(v) != n:
        raise ValueError("yi and vi must have equal length")
    if any(t <= 0.0 for t in v):
        raise ValueError("sampling variances must be strictly positive")
    D = core.mat(design)
    if len(D) != n or len(D[0]) != 2:
        raise ValueError("design must be n by 2")
    e = [int(t) for t in core.vec(edge)]
    if len(e) != 2 or e[0] == e[1]:
        raise ValueError("edge must be two distinct treatment labels")
    dir_idx = []
    rest = []
    for i in range(n):
        t1, t2 = int(D[i][0]), int(D[i][1])
        if (t1, t2) == (e[0], e[1]):
            dir_idx.append((i, 1.0))
        elif (t1, t2) == (e[1], e[0]):
            dir_idx.append((i, -1.0))
        else:
            rest.append(i)
    if not dir_idx:
        raise ValueError("no study makes that comparison directly")
    if not rest:
        raise ValueError("no indirect evidence remains once the edge is split")
    sw = sum(1.0 / v[i] for i, _ in dir_idx)
    direct = sum(s * y[i] / v[i] for i, s in dir_idx) / sw
    v_dir = 1.0 / sw
    Xr, treats, T = ma.net_design([[D[i][0], D[i][1]] for i in rest])
    if e[0] not in treats or e[1] not in treats:
        raise ValueError("the split edge is disconnected from the rest")
    p = T - 1
    w = [1.0 / v[i] for i in rest]
    yr = [y[i] for i in rest]
    beta, cov, _ = ma.wls(Xr, yr, w)
    pos = {t: j for j, t in enumerate(treats)}
    cvec = [0.0] * p
    if pos[e[0]] > 0:
        cvec[pos[e[0]] - 1] -= 1.0
    if pos[e[1]] > 0:
        cvec[pos[e[1]] - 1] += 1.0
    indirect = sum(cvec[j] * beta[j] for j in range(p))
    v_ind = sum(cvec[r] * cov[r][s] * cvec[s]
                for r in range(p) for s in range(p))
    diff = direct - indirect
    sd = math.sqrt(v_dir + v_ind)
    z = diff / sd if sd > 0.0 else float("nan")
    pv = 2.0 * (1.0 - core.pnorm(abs(z))) if sd > 0.0 else float("nan")
    return RichResult(payload={
        "direct": direct, "v_direct": v_dir, "indirect": indirect,
        "v_indirect": v_ind, "diff": diff, "z": z, "p": pv,
        "k_direct": len(dir_idx), "k_indirect": len(rest),
        "method": "Node-splitting inconsistency check"})


def cheatsheet():
    return "manh2h: node-splitting check of direct against indirect evidence"
