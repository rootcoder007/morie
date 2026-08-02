# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Decision trees exhibit high variance; small data changes yield very different trees."""

from . import _array_core as np

from ._richresult import RichResult
from .hmcart import geron_cart_algorithm, predict_tree

__all__ = ["geron_tree_high_variance"]


def geron_tree_high_variance(X, y, n_resamples=20, seed=0, criterion="gini", max_depth=None):
    """
    Decision trees exhibit high variance; small data changes yield very
    different trees.

    Formula: Var(f_tree(x)) large relative to Bias^2(f_tree)

    The claim is measured, not asserted. ``n_resamples`` bootstrap samples
    are drawn with a deterministic LCG, a tree is grown on each (via
    :func:`morie.fn.hmcart.geron_cart_algorithm`), and every tree predicts
    the full original ``X``. From that ensemble of predictions:

    * ``variance`` is the mean over points of the prediction variance
      across resamples (for labels, the probability of disagreeing with
      the majority vote);
    * ``bias2`` is the mean squared gap between the ensemble's average
      prediction and the truth;
    * ``structural_instability`` is the fraction of resamples whose root
      split differs from the root split on the full data -- the most
      visible symptom, since a different root means a completely
      different tree.

    High variance with low bias is the signature that motivates bagging:
    averaging many such trees cancels the variance and leaves the bias.

    Parameters
    ----------
    X : array-like, shape (m, n)
    y : array-like, shape (m,)
    n_resamples : int, default 20
        Number of bootstrap samples, >= 2.
    seed : int, default 0
    criterion : {"gini", "entropy", "mse"}, default "gini"
    max_depth : int, optional

    Returns
    -------
    result : RichResult
        Keys: variance, bias2, root_splits, structural_instability,
        ensemble_prediction, ensemble_score, single_tree_score,
        per_point_variance, estimate, n, method.

    Examples
    --------
    Even a cleanly separable problem is not stable: 3 of 10 bootstrap
    resamples miss a class entirely and collapse to a single leaf, which
    is enough to move the predictions.

    >>> X = [[1.0], [2.0], [8.0], [9.0]]
    >>> y = [0, 0, 1, 1]
    >>> r = geron_tree_high_variance(X, y, n_resamples=10, seed=1)
    >>> r["structural_instability"]
    0.3
    >>> round(r["variance"], 6)
    0.05
    >>> r["bias2"]
    0.0

    Overlapping classes destabilise the root split, and the variance is
    then strictly positive:

    >>> X2 = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
    >>> y2 = [0, 1, 0, 1, 1, 0]
    >>> r2 = geron_tree_high_variance(X2, y2, n_resamples=15, seed=3)
    >>> round(r2["structural_instability"], 6)
    0.733333
    >>> round(r2["variance"], 6)
    0.222222
    >>> len(r2["root_splits"])
    15

    References
    ----------
    Géron Ch 5
    """
    Xa = np.atleast_2d(np.asarray(X, dtype=float))
    ya = np.asarray(y).ravel()
    if ya.size != Xa.shape[0]:
        raise ValueError(f"geron_tree_high_variance: y has {ya.size} entries but X has {Xa.shape[0]} rows")
    B = int(n_resamples)
    if B < 2:
        raise ValueError(f"geron_tree_high_variance: n_resamples must be >= 2, got {n_resamples!r}")
    m = Xa.shape[0]
    if m < 2:
        raise ValueError(f"geron_tree_high_variance: need at least 2 instances, got {m}")

    full = geron_cart_algorithm(Xa, ya, criterion=criterion, max_depth=max_depth)
    root = full["tree"]
    root_key = None if root["leaf"] else (int(root["feature"]), float(root["threshold"]))

    s = int(seed) % 2**32
    preds = []
    roots = []
    for _ in range(B):
        idx = np.empty(m, dtype=int)
        for i in range(m):
            s = (1664525 * s + 1013904223) % 2**32
            idx[i] = int(((s + 0.5) / 2**32) * m)
        idx = np.minimum(idx, m - 1)
        yb = ya[idx]
        if criterion != "mse" and np.unique(yb).size < 2:
            # A degenerate resample: the tree is a single leaf, which is
            # itself part of the variance being measured.
            tree = {"leaf": True, "value": yb[0].item(), "n": int(m), "impurity": 0.0, "depth": 0}
            roots.append(None)
        else:
            res = geron_cart_algorithm(Xa[idx], yb, criterion=criterion, max_depth=max_depth)
            tree = res["tree"]
            roots.append(None if tree["leaf"] else (int(tree["feature"]), float(tree["threshold"])))
        preds.append(predict_tree(tree, Xa))

    P = np.asarray(preds)
    if criterion == "mse":
        Pf = P.astype(float)
        per_var = Pf.var(axis=0)
        mean_pred = Pf.mean(axis=0)
        bias2 = float(np.mean((mean_pred - ya.astype(float)) ** 2))
        ens = mean_pred
        ens_score = float(np.mean((ens - ya.astype(float)) ** 2))
        single = float(full["train_mse"])
    else:
        classes = np.unique(ya)
        counts = np.stack([(P == c).sum(axis=0) for c in classes], axis=1)
        maj = classes[counts.argmax(axis=1)]
        per_var = 1.0 - counts.max(axis=1) / B
        bias2 = float(np.mean((maj != ya).astype(float)))
        ens = maj
        ens_score = float(np.mean(maj == ya))
        single = float(full["train_accuracy"])

    instability = float(np.mean([rk != root_key for rk in roots]))

    return RichResult(
        title="Decision tree variance",
        summary_lines=[("Variance", float(np.mean(per_var))), ("Bias^2", bias2), ("Root instability", instability)],
        interpretation="High variance with low bias is exactly the profile that bagging fixes by averaging.",
        payload={
            "variance": float(np.mean(per_var)),
            "bias2": bias2,
            "per_point_variance": np.asarray(per_var, dtype=float).tolist(),
            "root_splits": roots,
            "reference_root": root_key,
            "structural_instability": instability,
            "ensemble_prediction": np.asarray(ens).tolist(),
            "ensemble_score": ens_score,
            "single_tree_score": single,
            "n_resamples": B,
            "criterion": criterion,
            "estimate": float(np.mean(per_var)),
            "n": int(m),
            "method": "bootstrap resampling of CART trees to measure prediction variance and root instability",
        },
    )


def cheatsheet():
    return "hmdthv: Decision trees exhibit high variance; small data changes yield very different trees"
