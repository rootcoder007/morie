# morie.fn -- slice s03 (rootcoder007/morie)
"""AutoInt: multi-head self-attention over feature embeddings.

Source consulted (FETCHED): Song, W. et al. (2019).  AutoInt: automatic
feature interaction learning via self-attentive neural networks.  *CIKM*
28, 1161-1170 (arXiv:1810.11921).  Its equation (2) embeds a numeric
field as e_m = v_m x_m; equation (3) is the key-value attention

    alpha^(h)_(m,k) = exp( psi^(h)(e_m, e_k) )
                      / sum_(l=1)^M exp( psi^(h)(e_m, e_l) )
    psi^(h)(e_m, e_k) = < W^(h)_Query e_m , W^(h)_Key e_k >

equation (4) is the head output ehat^(h)_m = sum_k alpha^(h)_(m,k)
W^(h)_Value e_k, and equation (6) adds the residual,

    e^Res_m = ReLU( ehat_m + W_Res e_m )

so that the original feature survives the interaction layer.  All four
are implemented; the projections are supplied by the caller, and the
identity is used when they are not, which is the degenerate case rather
than an invented parameterisation.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["autoint"]


def autoint(X, y=None, K=1, Wq=None, Wk=None, Wv=None, Wres=None, v=None):
    """One AutoInt interacting layer over M field embeddings.

    Parameters
    ----------
    X : 2-D array-like
        Field embeddings e_1..e_M, one row per field; or the raw field
        values x_m when ``v`` is given, in which case e_m = v_m x_m.
    y : array-like, optional
        Labels; when given, the logistic loss of the pooled output is
        reported.
    K : int
        Number of attention heads.  Heads share the caller's projections
        unless ``Wq`` etc. are given as a list of K matrices.
    Wq, Wk, Wv : 2-D array-like or list, optional
        Query, key and value projections.
    Wres : 2-D array-like, optional
        The residual projection of equation (6).
    v : 2-D array-like, optional
        Field embedding vectors, used with a 1-D ``X``.

    Returns
    -------
    RichResult with payload:
        estimate  : the pooled output (sum of all interacted embeddings)
        e_res     : the interacted embeddings, one row per field
        attention : alpha for head 0
        loss      : logistic loss when y is given
    """
    if v is not None:
        xs = k.vec(X)
        vs = k.mat(v)
        E = [[vs[m][j] * xs[m] for j in range(len(vs[m]))] for m in range(len(xs))]
    else:
        E = k.mat(X)
    M = len(E)
    d = len(E[0]) if M else 0
    heads = int(K)

    def proj(W, hh):
        if W is None:
            return None
        if isinstance(W, list) and W and isinstance(W[0], list) and \
                isinstance(W[0][0], (list, tuple)):
            return k.mat(W[hh % len(W)])
        return k.mat(W)

    acc = [[0.0] * d for _ in range(M)]
    att0 = []
    for h in range(heads):
        Q = proj(Wq, h)
        Kp = proj(Wk, h)
        Vp = proj(Wv, h)
        q = [k.matvec(Q, E[m]) if Q is not None else list(E[m]) for m in range(M)]
        kk = [k.matvec(Kp, E[m]) if Kp is not None else list(E[m]) for m in range(M)]
        vv = [k.matvec(Vp, E[m]) if Vp is not None else list(E[m]) for m in range(M)]
        for m in range(M):
            logits = []
            for l in range(M):
                s = 0.0
                for j in range(len(q[m])):
                    s += q[m][j] * kk[l][j]
                logits.append(s)
            a = k.softmax(logits)
            if h == 0:
                att0.append(a)
            for j in range(len(vv[0])):
                t = 0.0
                for l in range(M):
                    t += a[l] * vv[l][j]
                acc[m][j] += t
    res = []
    for m in range(M):
        r = k.matvec(k.mat(Wres), E[m]) if Wres is not None else list(E[m])
        res.append([k.relu(acc[m][j] + r[j]) for j in range(len(r))])
    pooled = 0.0
    for m in range(M):
        for j in range(len(res[m])):
            pooled += res[m][j]
    loss = float("nan")
    if y is not None:
        yy = k.vec(y)
        p = k.sigmoid(pooled)
        loss = 0.0
        for t in yy:
            loss -= t * math.log(max(p, 1e-300)) + (1.0 - t) * math.log(max(1.0 - p, 1e-300))
        loss /= len(yy) if yy else 1.0
    return RichResult(
        title="AutoInt interacting layer",
        summary_lines=[("fields", M), ("heads", heads)],
        payload={
            "estimate": pooled,
            "e_res": res,
            "attention": att0,
            "loss": loss,
            "method": "AutoInt self-attentive interaction layer (Song et al. 2019, eqs. 2-6)",
        },
    )


def cheatsheet():
    return "autoI: AutoInt -- multi-head self-attention for CTR"
