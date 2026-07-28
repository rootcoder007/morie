# morie.fn -- function file (rootcoder007/morie)
"""Causal forest for heterogeneous treatment effect estimation."""

import numpy as np

from ._cforest import CausalForest
from ._richresult import RichResult

__all__ = ["causal_forest"]


def causal_forest(Y, T, X, n_trees=200, min_node_size=10, max_depth=6,
                  mtry=None, subsample=0.5, seed=0):
    r"""Honest causal forest, over the shared engine in ``_cforest``.

    A causal forest is not a random forest fitted to the treatment
    effect, because the treatment effect is never observed. Athey and
    Imbens's change is to the SPLITTING RULE: a split is scored by how
    much the child nodes' estimated effects differ, so the trees hunt
    for heterogeneity rather than for outcome variance.

    The second change is HONESTY. Each tree's subsample is cut in
    half -- one half chooses the splits, the other fills the leaves.
    Without that, the observations that decided where a boundary went
    also supply the effect inside it, and the forest reports
    heterogeneity it manufactured: the leaf estimates are pushed away
    from the mean by construction. Wager and Athey's asymptotic
    normality result holds only for the honest version, so the split
    is part of the estimator, not a tuning choice.

    Out-of-bag predictions are returned next to the in-sample ones for
    the same reason. ``insample_spread`` much larger than
    ``oob_spread`` is the signal that the apparent heterogeneity is
    noise.

    Parameters
    ----------
    Y : array-like, shape (n,)
        Outcome.
    T : array-like of {0, 1}, shape (n,)
        Treatment.
    X : array-like, shape (n, p) or (n,)
        Covariates.
    n_trees, min_node_size, max_depth, mtry, subsample, seed :
        Forest controls.

    Returns
    -------
    RichResult
        ``cate``, ``cate_oob``, ``ate``, ``var_importance``
        (depth-weighted split share by feature), ``split_counts``
        (raw), ``insample_spread``, ``oob_spread``, ``forest`` (for
        :func:`~morie.fn.crfvar.causal_forest_variance`), ``honest``.

    References
    ----------
    Athey and Imbens (2016), *PNAS* 113:7353-7360 (honest splitting).
    Wager and Athey (2018), *JASA* 113:1228-1242 (asymptotic normality).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(800, 3))
    >>> T = (rng.uniform(size=800) < 0.5).astype(float)
    >>> Y = X[:, 0] + T * (X[:, 0] > 0) + rng.normal(scale=0.2, size=800)
    >>> out = causal_forest(Y, T, X, n_trees=60, seed=1)
    >>> hi, lo = X[:, 0] > 1, X[:, 0] < -1
    >>> bool(out["cate"][hi].mean() > out["cate"][lo].mean())
    True
    """
    Xa = np.asarray(X, dtype=float)
    if Xa.ndim == 1:
        Xa = Xa[:, None]
    y = np.asarray(Y, dtype=float).ravel()
    t = np.asarray(T, dtype=float).ravel()
    forest = CausalForest(
        n_trees=int(n_trees), min_leaf=int(min_node_size),
        max_depth=int(max_depth), mtry=mtry, subsample=float(subsample),
        seed=int(seed),
    ).fit(Xa, y, t)

    cate = forest.predict()
    oob = forest.predict(oob=True)
    ok = np.isfinite(oob)

    # Variable importance by DEPTH-WEIGHTED split frequency, weighting a
    # split at depth d by (d+1)^-2. Raw split counts do not work: deep
    # splits are made on tiny, noisy subsamples and there are
    # exponentially more of them, so counting every split equally buries
    # the real signal near the root under noise from the leaves. This is
    # grf's convention.
    counts = np.zeros(Xa.shape[1])
    raw = np.zeros(Xa.shape[1])
    stack = [(t, 0) for t in forest.trees_]
    while stack:
        node, depth = stack.pop()
        if node.feature is None:
            continue
        counts[node.feature] += 1.0 / (depth + 1.0) ** 2
        raw[node.feature] += 1.0
        stack.append((node.left, depth + 1))
        stack.append((node.right, depth + 1))
    total = counts.sum()

    return RichResult(
        payload={
            "cate": cate,
            "cate_oob": oob,
            "estimate": float(np.mean(cate)),
            "ate": float(np.mean(cate)),
            "ate_oob": float(np.mean(oob[ok])) if ok.any() else np.nan,
            "var_importance": counts / total if total > 0 else counts,
            "split_counts": raw,
            "importance_note": (
                "splits are weighted by (depth + 1)^-2: raw counts bury the "
                "signal near the root under the exponentially more "
                "numerous, noisier deep splits"
            ),
            "insample_spread": float(np.std(cate)),
            "oob_spread": float(np.std(oob[ok])) if ok.any() else np.nan,
            "spread_note": (
                "an in-sample spread far above the out-of-bag spread means "
                "the forest is fitting noise rather than finding "
                "heterogeneity"
            ),
            "n_oob_missing": int((~ok).sum()),
            "forest": forest,
            "n_trees": int(n_trees),
            "min_node_size": int(min_node_size),
            "honest": True,
            "honesty_note": (
                "each tree splits on one half of its subsample and fills "
                "leaves from the other; without it the same data both chose "
                "the boundary and estimated inside it, and the asymptotic "
                "normality result does not hold"
            ),
            "n": int(y.size),
            "method": "Honest causal forest (Athey-Imbens 2016, Wager-Athey 2018)",
        }
    )


def cheatsheet():
    return (
        "cfst: honest causal forest -- splits scored on child-effect "
        "heterogeneity, leaves filled from a held-out half, OOB spread "
        "reported next to in-sample"
    )
