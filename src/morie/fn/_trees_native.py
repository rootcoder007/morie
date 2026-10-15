# morie.fn -- shared helper (rootcoder007/morie)
"""Native second-order gradient-boosted trees.

Mirrors ``r-package/morie/R/trees_native.R`` so the two languages compute the
same thing. Used where scikit-learn cannot express the objective -- notably
the L1/L2 penalties on leaf weights, which ``GradientBoosting*`` has no
parameter for.

The regularised objective is XGBoost's, per the project's own documentation
(https://xgboost.readthedocs.io/en/stable/tutorials/model.html):

    w_j*  = -G_j / (H_j + lambda)
    Gain  = 0.5 [ G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda)
                  - (G_L+G_R)^2/(H_L+H_R+lambda) ] - gamma

and the boosting loop is Algorithm 10.3 with the shrinkage of eq. (10.41) in
Hastie, Tibshirani & Friedman (2009), *The Elements of Statistical Learning*,
2nd edn, pp. 360-361.

References
----------
Chen, T. & Guestrin, C. (2016). XGBoost: a scalable tree boosting system.
*KDD '16*, 785-794.

Friedman, J. H. (2001). Greedy function approximation: a gradient boosting
machine. *Annals of Statistics*, 29(5), 1189-1232.
"""

from __future__ import annotations

import math
from itertools import accumulate

from . import _array_core as np

__all__ = ["gb_fit", "gb_predict"]


def _soft_threshold(g: float, alpha: float) -> float:
    if alpha <= 0:
        return g
    if g > alpha:
        return g - alpha
    if g < -alpha:
        return g + alpha
    return 0.0


def _leaf_weight(G, H, lam, alpha):
    return -_soft_threshold(G, alpha) / (H + lam)


def _best_split(Xc, g, h, idx, min_node, lam, gamma_pen):
    """Best (feature, threshold) for one node, or None if no split helps.

    ``Xc`` is the design as a list of columns, ``g``/``h`` and ``idx`` are
    lists. The node search runs once per node of every tree, so as array
    expressions (fancy indexing, argsort, cumsum per column) it was most of
    a boosting fit; the arithmetic below is the same, element by element:
    exactly rounded node totals, running prefix sums in sorted order, the
    stable (mergesort) order, and the first maximiser of the gain.
    """
    G = math.fsum(g[i] for i in idx)
    H = math.fsum(h[i] for i in idx)
    parent = G * G / (H + lam)
    n = len(idx)
    best = None
    if n < 2:
        return best
    for j, col in enumerate(Xc):
        o = sorted(idx, key=col.__getitem__)
        vs = [col[i] for i in o]
        gs = list(accumulate(g[i] for i in o))
        hs = list(accumulate(h[i] for i in o))
        # Only positions where the value actually changes are valid splits.
        cut = [c for c in range(n - 1) if vs[c] < vs[c + 1] and c + 1 >= min_node and n - c - 1 >= min_node]
        if not cut:
            continue
        gain = []
        for c in cut:
            GL, HL = gs[c], hs[c]
            GR, HR = G - GL, H - HL
            gain.append(0.5 * (GL * GL / (HL + lam) + GR * GR / (HR + lam) - parent) - gamma_pen)
        top = max(gain)
        k = gain.index(top)
        if top > 0 and (best is None or top > best[0]):
            best = (float(top), j, float((vs[cut[k]] + vs[cut[k] + 1]) / 2))
    return best


def _grow(Xc, g, h, idx, depth, max_depth, min_node, lam, alpha, gamma_pen, imp):
    G, H = math.fsum(g[i] for i in idx), math.fsum(h[i] for i in idx)
    leaf = {"leaf": True, "w": _leaf_weight(G, H, lam, alpha)}
    if depth >= max_depth or len(idx) < 2 * min_node:
        return leaf
    s = _best_split(Xc, g, h, idx, min_node, lam, gamma_pen)
    if s is None:
        return leaf
    gain, j, thr = s
    imp[j] += gain
    col = Xc[j]
    left = [i for i in idx if col[i] <= thr]
    right = [i for i in idx if col[i] > thr]
    if not left or not right:
        return leaf
    return {
        "leaf": False,
        "j": j,
        "thr": thr,
        "left": _grow(Xc, g, h, left, depth + 1, max_depth, min_node, lam, alpha, gamma_pen, imp),
        "right": _grow(Xc, g, h, right, depth + 1, max_depth, min_node, lam, alpha, gamma_pen, imp),
    }


def _predict_tree(node, X):
    out = np.zeros(X.shape[0])
    stack = [(node, np.arange(X.shape[0]))]
    while stack:
        nd, rows = stack.pop()
        if rows.size == 0:
            continue
        if nd["leaf"]:
            out[rows] = nd["w"]
            continue
        m = X[rows, nd["j"]] <= nd["thr"]
        stack.append((nd["left"], rows[m]))
        stack.append((nd["right"], rows[~m]))
    return out


def gb_fit(
    X,
    y,
    task="regression",
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_node=1,
    reg_lambda=0.0,
    reg_alpha=0.0,
    gamma_pen=0.0,
):
    """Fit a native gradient-boosted tree ensemble.

    ``task="regression"`` uses squared-error loss, for which the negative
    gradient is the ordinary residual (ESL Table 10.2). ``"classification"``
    uses the binomial deviance on the logit scale, so the Newton step
    ``-G/(H+lambda)`` is the exact line search of Algorithm 10.3 step 2(c).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    imp = np.zeros(p)
    if task == "regression":
        yv = np.asarray(y, dtype=float).ravel()
        f0 = float(yv.mean())
    else:
        yv = np.asarray(y).ravel()
        _, yv = np.unique(yv, return_inverse=True)
        yv = yv.astype(float)
        pbar = min(max(float(yv.mean()), 1e-6), 1 - 1e-6)
        f0 = float(np.log(pbar / (1 - pbar)))
    f = np.full(n, f0)
    Xc = [list(c) for c in zip(*X.tolist())]
    trees = []
    for _ in range(int(n_estimators)):
        if task == "regression":
            g = f - yv
            h = np.ones(n)
        else:
            pr = 1.0 / (1.0 + np.exp(-f))
            g = pr - yv
            h = np.maximum(pr * (1 - pr), 1e-6)
        tree = _grow(
            Xc,
            g.tolist(),
            h.tolist(),
            list(range(n)),
            0,
            int(max_depth),
            int(min_node),
            float(reg_lambda),
            float(reg_alpha),
            float(gamma_pen),
            imp,
        )
        trees.append(tree)
        f = f + learning_rate * _predict_tree(tree, X)
    total = imp.sum()
    return {
        "f0": f0,
        "trees": trees,
        "learning_rate": float(learning_rate),
        "task": task,
        "importance": imp / total if total > 0 else imp,
        "fitted": f if task == "regression" else 1.0 / (1.0 + np.exp(-f)),
    }


def gb_predict(fit, X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    f = np.full(X.shape[0], fit["f0"])
    for tree in fit["trees"]:
        f = f + fit["learning_rate"] * _predict_tree(tree, X)
    return f if fit["task"] == "regression" else 1.0 / (1.0 + np.exp(-f))
