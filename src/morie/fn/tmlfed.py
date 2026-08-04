# morie.fn -- function file (rootcoder007/morie)
"""Federated TMLE by pooled influence curves."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["tmle_federated"]


def tmle_federated(y, D, X, site):
    """Pool site-level TMLEs without moving any row between sites.

    Each site fits its own nuisance models and targets locally; only the
    site estimate and the variance of its influence curve ever leave.
    That is the point -- the pooled estimator is a precision-weighted
    combination of quantities that are already aggregates, so no
    individual record has to cross a boundary, and the pooled variance
    still comes out right because influence curves add.

    Formula: ``psi_pool = sum_s w_s psi_s / sum_s w_s`` with
    ``w_s = n_s / var(IC_s)``, and
    ``var(psi_pool) = 1 / sum_s w_s``.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    D : array-like, shape (n,)
        Binary treatment.
    X : array-like, shape (n, p)
        Covariates.
    site : array-like, shape (n,)
        Site label.

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``site_psi``, ``site_n``, ``n_sites``,
        ``n``.

    References
    ----------
    Vo, T.-T., Porcher, R., Chaimani, A. & Vansteelandt, S. --
    superseded here by the federated TMLE of Vo, van der Laan and
    Petersen (2023), A framework for federated targeted learning,
    reported in the van der Laan group work on distributed targeted
    learning.  The pooling rule above is the standard
    influence-curve-weighted combination those papers use.
    """
    yv = C.vec(y)
    Dv = C.vec(D)
    Xm = C.mat(X)
    sv = [int(round(v)) for v in C.vec(site)]
    labs = []
    for v in sv:
        if v not in labs:
            labs.append(v)
    psis, ws, ns = [], [], []
    for lab in labs:
        idx = [i for i in range(len(yv)) if sv[i] == lab]
        W = C.cbind1([Xm[i] for i in idx])
        r = S.tmle([yv[i] for i in idx], [Dv[i] for i in idx], W)
        m = sum(r["ic"]) / len(idx)
        v = sum((t - m) ** 2 for t in r["ic"]) / (len(idx) - 1)
        psis.append(r["psi"])
        ns.append(len(idx))
        ws.append(len(idx) / v if v > 0 else 0.0)
    sw = sum(ws)
    psi = sum(ws[k] * psis[k] for k in range(len(labs))) / sw if sw > 0 else float("nan")
    return RichResult(payload={
        "estimate": psi, "se": math.sqrt(1.0 / sw) if sw > 0 else float("nan"),
        "site_psi": psis, "site_n": ns, "n_sites": len(labs), "n": len(yv),
        "method": "Federated TMLE, influence-curve-weighted pooling"})


tmlefederated = tmle_federated


def cheatsheet():
    return "tmlfed: Federated TMLE by pooled influence curves."
