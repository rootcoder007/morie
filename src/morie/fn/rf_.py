# morie.fn -- function file (rootcoder007/morie)
"""Random forest (bagged CART) for regression."""

from . import _array_core as np

from ._containers import DescriptiveResult
from .cart import _build_tree, _predict_one


def random_forest_simple(
    X: np.ndarray, y: np.ndarray, n_trees: int = 50, max_depth: int = 5,
    seed: int = 42, max_features: int | None = None
) -> DescriptiveResult:
    """
    Random forest regression (pure numpy).

    Each tree is fit to a bootstrap sample, and at EVERY node a
    random subset of ``max_features`` columns is drawn and the best
    split taken among those only. That per-node draw is what
    distinguishes a random forest from plain bagging: averaging trees
    that are identically distributed with pairwise correlation rho
    leaves rho sigma^2 behind, so the gain is bounded by how
    decorrelated they are.

    The default is ``floor(p/3)``, which is the REGRESSION default.
    ``floor(sqrt(p))`` is the classification default; the two cross
    at p = 9 and above it floor(p/3) is the larger.

    (Until this was corrected, the subset never reached a split --
    ``_build_tree`` was called without ``max_features`` -- so every
    tree saw all p columns and the result was Breiman's 1996 bagging
    rather than his 2001 random forest, despite the docstring and
    citation saying otherwise.)

    :param X: (n, p) feature matrix.
    :param y: (n,) target.
    :param n_trees: Number of trees.
    :param max_depth: Maximum tree depth.
    :param seed: Random seed.
    :param max_features: Columns drawn per node; ``floor(p/3)`` when
        ``None``.
    :return: DescriptiveResult with predictions and R-squared.

    References
    ----------
    Breiman L (2001). Random forests. Machine Learning, 45(1), 5-32.
    Hastie, Tibshirani and Friedman (2009), Algorithm 15.1 and
    Sec. 15.3 for the floor(p/3) regression default.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n, p = X.shape
    rng = np.random.default_rng(seed)
    mf = max(1, p // 3) if max_features is None else int(max_features)
    if not 1 <= mf <= p:
        raise ValueError(f"max_features must lie in 1..{p}, got {mf}.")
    trees = []
    for _ in range(n_trees):
        idx = rng.choice(n, n, replace=True)
        tree = _build_tree(X[idx], y[idx], 0, max_depth, 2,
                           max_features=mf, rng=rng)
        trees.append(tree)
    preds = np.zeros(n)
    for tree in trees:
        preds += np.array([_predict_one(tree, x) for x in X])
    preds /= n_trees
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return DescriptiveResult(
        name="random_forest",
        value=r2,
        extra={"predictions": preds, "r_squared": r2, "n_trees": n_trees,
               "max_depth": max_depth, "n": n, "max_features": mf,
               "subset_drawn_per": "node",
               "mtry_rule": "floor(p/3), the regression default; "
                            "floor(sqrt(p)) is the classification one"},
    )


rf_ = random_forest_simple


def cheatsheet() -> str:
    return "random_forest_simple({}) -> Random forest (bagged CART) for regression."
