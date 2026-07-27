# morie.fn -- function file (rootcoder007/morie)
"""Causal forest with a monotonicity constraint on the CATE."""

import numpy as np

from ._cforest import CausalForest
from ._richresult import RichResult

__all__ = ["hetero_causal_forest"]


def hetero_causal_forest(y, D, X, monotone_feature=None, direction=1, n_trees=200, min_leaf=10, seed=0):
    r"""Honest causal forest, then isotonic projection onto a monotone CATE.

    When theory says the effect can only increase (or only decrease)
    in one covariate, the unconstrained forest's non-monotone wiggles
    are noise. This fits the forest, then projects
    :math:`\hat\tau(x)` onto the set of monotone functions of that
    covariate by the pool-adjacent-violators algorithm -- the exact
    :math:`L_2` isotonic projection, so the constrained fit is the
    closest monotone function to the forest's own output and can only
    reduce mean squared error against a truly monotone target.

    Parameters
    ----------
    y, D, X :
        As in :func:`morie.fn.crfath.causal_forest_wager_athey`.
    monotone_feature : int, optional
        Column of X the effect must be monotone in. None returns the
        unconstrained forest.
    direction : {1, -1}, default 1
        1 = nondecreasing, -1 = nonincreasing.
    n_trees, min_leaf, seed :
        Forest hyperparameters.

    Returns
    -------
    RichResult
        keys: ``cate`` (constrained), ``cate_raw`` (unconstrained),
        ``monotone_feature``, ``direction``, ``violations_before``
        (adjacent decreases in the raw fit, sorted by the feature),
        ``violations_after``, ``n``, ``method``.

    References
    ----------
    Robertson, T., Wright, F. T. & Dykstra, R. L. (1988). *Order
    Restricted Statistical Inference*. Wiley. (PAVA is the L2 isotonic
    projection)

    Wager, S. & Athey, S. (2018). Estimation and inference of
    heterogeneous treatment effects using random forests. *JASA*,
    113(523), 1228-1242.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    f = CausalForest(n_trees=n_trees, min_leaf=min_leaf, seed=seed)
    f.fit(X, y, D)
    raw = f.predict()

    if monotone_feature is None:
        return RichResult(
            payload={
                "cate": raw,
                "cate_raw": raw,
                "monotone_feature": None,
                "direction": int(direction),
                "violations_before": 0,
                "violations_after": 0,
                "n": int(raw.size),
                "method": "Causal forest (no monotonicity constraint requested)",
            }
        )

    j = int(monotone_feature)
    if not 0 <= j < X.shape[1]:
        raise ValueError(f"monotone_feature must index a column of X (0..{X.shape[1] - 1}).")
    if direction not in (1, -1):
        raise ValueError(f"direction must be 1 or -1, got {direction}.")

    order = np.argsort(X[:, j])
    v = raw[order] * direction
    before = int(np.sum(np.diff(v) < -1e-12))

    # pool-adjacent-violators
    vals, weights = [], []
    for x in v:
        vals.append(float(x))
        weights.append(1.0)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = weights[-2] + weights[-1]
            m = (vals[-2] * weights[-2] + vals[-1] * weights[-1]) / w
            vals[-2:] = [m]
            weights[-2:] = [w]
    fitted = np.repeat(vals, [int(w) for w in weights])

    after = int(np.sum(np.diff(fitted) < -1e-9))
    cate = np.empty_like(raw)
    cate[order] = fitted * direction

    return RichResult(
        payload={
            "cate": cate,
            "cate_raw": raw,
            "monotone_feature": j,
            "direction": int(direction),
            "violations_before": before,
            "violations_after": after,
            "n": int(raw.size),
            "method": "Causal forest with an isotonic (PAVA) monotonicity constraint",
        }
    )


def cheatsheet():
    return "htgcrf: forest CATE projected onto monotone-in-X_j by PAVA"
