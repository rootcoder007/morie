# morie.fn -- function file (rootcoder007/morie)
"""DeepFM: factorization machine plus deep network on a shared embedding."""

import math

from . import _array_core as np
from . import _s03core as core
from ._richresult import RichResult

__all__ = ["deepfm"]


def fm_second_order(V, x):
    """0.5 sum_f ((sum_i v_if x_i)^2 - sum_i (v_if x_i)^2).

    The O(kn) identity for the pairwise term.  It is exactly equal to
    the naive double sum over i < j, which is what makes it checkable
    rather than merely fast.
    """
    n = len(x)
    K = len(V[0])
    tot = 0.0
    for f in range(K):
        s1 = 0.0
        s2 = 0.0
        for i in range(n):
            t = V[i][f] * x[i]
            s1 += t
            s2 += t * t
        tot += s1 * s1 - s2
    return 0.5 * tot


def deepfm(X, y=None, K=4, mlp_h=4, w0=0.0, seed=42, deep_scale=1.0):
    """
    DeepFM

    Formula: FM (low-order) + DNN (high-order) on shared embedding

    The wide part is a factorization machine over the shared embedding
    V, the deep part an MLP over the same V, and the logit is their sum
    through a sigmoid.  Sharing the embedding is the whole point: no
    hand-crafted crosses and no separate pretraining.  With the deep
    branch scaled to zero the prediction is exactly the FM prediction,
    which is the reduction used to check the wiring.

    Parameters
    ----------
    X : array-like
        n x p feature matrix.
    y : array-like or None
        Binary labels, for the reported log loss.
    K : int
        Embedding dimension.
    mlp_h : int
        Hidden width of the deep branch.
    w0 : float
        Global bias.
    seed : int
        Seed of the deterministic stream.
    deep_scale : float
        Multiplier on the deep branch; 0 leaves the FM alone.

    Returns
    -------
    result : dict
        Keys: estimate (mean predicted probability), p_hat, fm_part,
        deep_part, logloss, n, p, K.

    References
    ----------
    Guo, Tang, Ye, Li & He (2017), DeepFM: A Factorization-Machine
    based Neural Network for CTR Prediction, IJCAI 2017:1725-1731.
    Rendle (2010), Factorization Machines, ICDM 2010:995-1000.
    """
    Xm = core.mat(X)
    n = len(Xm)
    if n == 0:
        raise ValueError("empty input: X has no rows")
    p = len(Xm[0])
    K = int(K)
    if K < 1:
        raise ValueError("K must be at least 1")
    h = int(mlp_h)
    if h < 1:
        raise ValueError("mlp_h must be at least 1")
    rng = np.random.default_rng(seed)
    w = [float(rng.normal(0.0, 0.1)) for _ in range(p)]
    V = [[float(rng.normal(0.0, 0.1)) for _ in range(K)] for _ in range(p)]
    W1 = [[float(rng.normal(0.0, 0.1)) for _ in range(h)]
          for _ in range(p * K)]
    b1 = [float(rng.normal(0.0, 0.1)) for _ in range(h)]
    W2 = [float(rng.normal(0.0, 0.1)) for _ in range(h)]
    fm, dp, ph = [], [], []
    for i in range(n):
        x = Xm[i]
        lin = w0 + sum(w[j] * x[j] for j in range(p))
        wide = lin + fm_second_order(V, x)
        emb = [V[j][f] * x[j] for j in range(p) for f in range(K)]
        hid = []
        for t in range(h):
            s = b1[t]
            for q in range(p * K):
                s += emb[q] * W1[q][t]
            hid.append(core.relu(s))
        deep = sum(hid[t] * W2[t] for t in range(h))
        fm.append(wide)
        dp.append(deep_scale * deep)
        ph.append(core.sigmoid(wide + deep_scale * deep))
    ll = float("nan")
    if y is not None:
        yv = core.vec(y)
        if len(yv) != n:
            raise ValueError("y must have one label per row")
        if any(v not in (0.0, 1.0) for v in yv):
            raise ValueError("y must be binary 0/1")
        ll = -sum(yv[i] * math.log(ph[i] + 1e-300)
                  + (1 - yv[i]) * math.log(1 - ph[i] + 1e-300)
                  for i in range(n)) / n
    return RichResult(payload={
        "estimate": sum(ph) / n,
        "p_hat": ph,
        "fm_part": fm,
        "deep_part": dp,
        "logloss": ll,
        "n": n,
        "p": p,
        "K": K,
        "method": "DeepFM: factorization machine plus deep network",
    })


def cheatsheet():
    return "deepF: DeepFM (factorization machine plus deep network)"
