# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""XGBoost: regularized gradient boosting with second-order Taylor approximation."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_xgboost"]


def _leaf_weight(G, H, lam):
    return -G / (H + lam)


def _gain(GL, HL, GR, HR, lam, gamma):
    G, H = GL + GR, HL + HR
    return 0.5 * (GL * GL / (HL + lam) + GR * GR / (HR + lam) - G * G / (H + lam)) - gamma


def _build(X, g, h, depth, max_depth, lam, gamma, min_child_weight):
    G, H = float(np.sum(g)), float(np.sum(h))
    node = {"weight": float(_leaf_weight(G, H, lam)), "G": G, "H": H, "leaf": True}
    if depth >= max_depth or X.shape[0] < 2:
        return node
    best = None
    for j in range(X.shape[1]):
        order = np.argsort(X[:, j], kind="mergesort")
        xs, gs, hs = X[order, j], g[order], h[order]
        GL = HL = 0.0
        for i in range(xs.size - 1):
            GL += gs[i]
            HL += hs[i]
            if xs[i + 1] == xs[i]:
                continue
            GR, HR = G - GL, H - HL
            if HL < min_child_weight or HR < min_child_weight:
                continue
            gain = _gain(GL, HL, GR, HR, lam, gamma)
            if best is None or gain > best[0]:
                best = (gain, j, 0.5 * (xs[i] + xs[i + 1]))
    if best is None or best[0] <= 0:
        return node
    gain, j, thr = best
    left = X[:, j] <= thr
    node.update(
        {
            "leaf": False,
            "feature": int(j),
            "threshold": float(thr),
            "gain": float(gain),
            "left": _build(X[left], g[left], h[left], depth + 1, max_depth, lam, gamma, min_child_weight),
            "right": _build(X[~left], g[~left], h[~left], depth + 1, max_depth, lam, gamma, min_child_weight),
        }
    )
    return node


def _predict(node, X):
    out = np.empty(X.shape[0])
    for i in range(X.shape[0]):
        nd = node
        while not nd["leaf"]:
            nd = nd["left"] if X[i, nd["feature"]] <= nd["threshold"] else nd["right"]
        out[i] = nd["weight"]
    return out


def geron_xgboost(
    X,
    y,
    n_estimators=10,
    learning_rate=0.3,
    max_depth=3,
    reg_lambda=1.0,
    gamma=0.0,
    min_child_weight=1.0,
    objective="squared",
):
    """
    XGBoost: regularized gradient boosting with second-order Taylor approximation.

    Formula: obj = sum_i L(y_i, y_hat_i) + sum_t Omega(f_t)

    Implements the actual XGBoost derivation rather than plain gradient
    boosting. Expanding the loss to second order gives, for a fixed tree
    structure, the closed-form optimal leaf weight and split gain

    ``w_j = -G_j / (H_j + lambda)`` and
    ``gain = 0.5*(G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - G^2/(H+lambda)) - gamma``

    where ``G`` and ``H`` sum the first and second derivatives in the
    node. Splits are found by exact greedy enumeration over sorted
    feature values, a split is kept only when its gain is positive
    (that is `gamma` acting as pre-pruning), and `min_child_weight`
    rejects children with too little Hessian mass.

    Parameters
    ----------
    X : array-like
        Design matrix (n, d).
    y : array-like
        Targets; ``{0, 1}`` for ``objective="logistic"``.
    n_estimators : int, default 10
        Boosting rounds (>= 1).
    learning_rate : float, default 0.3
        Shrinkage eta in (0, 1].
    max_depth : int, default 3
        Maximum tree depth (>= 1).
    reg_lambda : float, default 1.0
        L2 penalty on leaf weights (>= 0).
    gamma : float, default 0.0
        Minimum gain required to split (>= 0).
    min_child_weight : float, default 1.0
        Minimum Hessian sum in a child (>= 0).
    objective : {"squared", "logistic"}, default "squared"
        Loss whose g and h drive the boosting.

    Returns
    -------
    result : RichResult
        Keys: predicted, trees, base_score, loss_curve, feature_importance,
        estimate, n, method.

    Examples
    --------
    A single unshrunk stump on a step target recovers the two group means
    exactly: base = 5.5, g = base - y, so the left leaf weight is
    -(9)/2 = -4.5 and the fitted values are 1 and 10.

    >>> r = geron_xgboost([[0.0], [1.0], [2.0], [3.0]], [1.0, 1.0, 10.0, 10.0],
    ...                   n_estimators=1, learning_rate=1.0, max_depth=1, reg_lambda=0.0)
    >>> [round(float(v), 12) for v in r["predicted"]]
    [1.0, 1.0, 10.0, 10.0]
    >>> round(float(r["base_score"]), 12)
    5.5
    >>> round(float(r["trees"][0]["gain"]), 6)
    40.5
    >>> round(float(r["trees"][0]["left"]["weight"]), 12)
    -4.5

    Boosting drives the training loss down monotonically:

    >>> r2 = geron_xgboost([[0.0], [1.0], [2.0], [3.0]], [0.0, 1.0, 4.0, 9.0], n_estimators=5, max_depth=2)
    >>> bool(all(a >= b for a, b in zip(r2["loss_curve"], r2["loss_curve"][1:])))
    True

    References
    ----------
    Géron Ch 6
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError("geron_xgboost: X must be a non-empty (n, d) design matrix")
    t = np.asarray(y, dtype=float).ravel()
    if t.size != A.shape[0]:
        raise ValueError(f"geron_xgboost: X has {A.shape[0]} rows but y has {t.size} targets")
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(t))):
        raise ValueError("geron_xgboost: X and y must be finite")
    obj = str(objective).lower()
    if obj not in ("squared", "logistic"):
        raise ValueError(f"geron_xgboost: objective must be 'squared' or 'logistic', got {objective!r}")
    if obj == "logistic" and not np.all(np.isin(t, (0.0, 1.0))):
        raise ValueError("geron_xgboost: objective='logistic' requires y in {0, 1}")
    M = int(n_estimators)
    if M < 1:
        raise ValueError(f"geron_xgboost: n_estimators must be >= 1, got {M}")
    eta = float(learning_rate)
    if not (0.0 < eta <= 1.0):
        raise ValueError(f"geron_xgboost: learning_rate must lie in (0, 1], got {eta}")
    depth = int(max_depth)
    if depth < 1:
        raise ValueError(f"geron_xgboost: max_depth must be >= 1, got {depth}")
    lam, gam, mcw = float(reg_lambda), float(gamma), float(min_child_weight)
    for nm, v in (("reg_lambda", lam), ("gamma", gam), ("min_child_weight", mcw)):
        if not np.isfinite(v) or v < 0:
            raise ValueError(f"geron_xgboost: {nm} must be non-negative and finite, got {v}")

    if obj == "squared":
        base = float(np.mean(t))
        F = np.full(t.size, base)
    else:
        p = float(np.clip(np.mean(t), 1e-6, 1 - 1e-6))
        base = float(np.log(p / (1 - p)))
        F = np.full(t.size, base)

    def _loss(F):
        if obj == "squared":
            return float(np.mean(0.5 * (F - t) ** 2))
        z = np.clip(F, -50, 50)
        return float(np.mean(np.log1p(np.exp(z)) - t * z))

    trees = []
    losses = [_loss(F)]
    importance = np.zeros(A.shape[1])
    for _ in range(M):
        if obj == "squared":
            g = F - t
            h = np.ones_like(t)
        else:
            p = 1.0 / (1.0 + np.exp(-np.clip(F, -50, 50)))
            g = p - t
            h = np.maximum(p * (1.0 - p), 1e-12)
        tree = _build(A, g, h, 0, depth, lam, gam, mcw)
        F = F + eta * _predict(tree, A)
        trees.append(tree)
        losses.append(_loss(F))

        stack = [tree]
        while stack:
            nd = stack.pop()
            if not nd["leaf"]:
                importance[nd["feature"]] += nd["gain"]
                stack.extend([nd["left"], nd["right"]])

    pred = F if obj == "squared" else 1.0 / (1.0 + np.exp(-np.clip(F, -50, 50)))

    return RichResult(
        title="XGBoost (regularised gradient boosting)",
        summary_lines=[
            ("Rounds", M),
            ("Learning rate", eta),
            ("lambda", lam),
            ("Final training loss", losses[-1]),
        ],
        interpretation=(
            "The second-order expansion is what lets XGBoost score a split in closed form; lambda "
            "shrinks leaf weights and gamma refuses splits whose gain does not pay for the extra leaf."
        ),
        payload={
            "predicted": pred,
            "raw_score": F,
            "trees": trees,
            "base_score": base,
            "loss_curve": np.asarray(losses, dtype=float),
            "feature_importance": importance,
            "objective": obj,
            "estimate": float(losses[-1]),
            "n": int(t.size),
            "method": "XGBoost: exact greedy splits scored by the second-order gain, shrunk by eta",
        },
    )


def cheatsheet():
    return "hmxgb: XGBoost: regularized gradient boosting with second-order Taylor approximation"
