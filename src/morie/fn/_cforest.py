# morie.fn -- function file (rootcoder007/morie)
"""Shared honest causal-forest core (Wager-Athey style)."""

import numpy as np

__all__ = ["CausalForest"]


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "tau", "n")

    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.tau = 0.0
        self.n = 0


def _tau(y, d):
    """Difference in means; NaN when an arm is empty."""
    t, c = d == 1, d == 0
    if not t.any() or not c.any():
        return np.nan
    return float(y[t].mean() - y[c].mean())


class CausalForest:
    r"""Honest causal forest for heterogeneous treatment effects.

    Each tree is grown on one half of a subsample (the *splitting*
    half) by maximising the squared difference in child treatment
    effects -- the causal analogue of variance reduction -- and its
    leaf values are then re-estimated on the untouched *estimation*
    half. That honesty split is what makes the leaf effects
    approximately unbiased; a tree that both chooses the split and
    fills the leaf with the same data overfits the heterogeneity it
    claims to find.

    Predictions are averaged over trees; out-of-bag predictions
    average only the trees that never saw the row, which is the
    version to use for any downstream test of heterogeneity.

    Parameters
    ----------
    n_trees : int, default 200
    min_leaf : int, default 10
        Minimum units of *each* arm in a leaf.
    max_depth : int, default 6
    mtry : int, optional
        Features tried per split; default ceil(sqrt(p)).
    subsample : float, default 0.5
        Fraction of rows drawn (without replacement) per tree.
    imbalance_penalty : float, default 0.0
        GRF's regularizer on child imbalance. The raw heterogeneity
        criterion is happiest carving off a tiny extreme leaf, whose
        tau is then estimated from almost nothing; subtracting
        ``imbalance_penalty * (1/n_L + 1/n_R)`` prices that in. Zero
        recovers the plain Athey-Imbens criterion.
    seed : int, default 0

    References
    ----------
    Wager, S. & Athey, S. (2018). Estimation and inference of
    heterogeneous treatment effects using random forests. *Journal of
    the American Statistical Association*, 113(523), 1228-1242.

    Athey, S. & Imbens, G. (2016). Recursive partitioning for
    heterogeneous causal effects. *PNAS*, 113(27), 7353-7360. (honest
    estimation)
    """

    def __init__(self, n_trees=200, min_leaf=10, max_depth=6, mtry=None, subsample=0.5,
                 imbalance_penalty=0.0, seed=0):
        if n_trees < 1:
            raise ValueError(f"n_trees must be at least 1, got {n_trees}.")
        if min_leaf < 1:
            raise ValueError(f"min_leaf must be at least 1, got {min_leaf}.")
        if not 0 < subsample <= 1:
            raise ValueError(f"subsample must lie in (0, 1], got {subsample}.")
        if imbalance_penalty < 0:
            raise ValueError(f"imbalance_penalty must be non-negative, got {imbalance_penalty}.")
        self.n_trees = int(n_trees)
        self.min_leaf = int(min_leaf)
        self.max_depth = int(max_depth)
        self.mtry = mtry
        self.subsample = float(subsample)
        self.imbalance_penalty = float(imbalance_penalty)
        self.seed = int(seed)
        self.trees_ = []
        self.in_bag_ = []

    def _grow(self, X, y, d, rows_split, rows_est, depth, rng):
        node = _Node()
        node.n = rows_est.size
        node.tau = _tau(y[rows_est], d[rows_est])
        if np.isnan(node.tau):
            node.tau = _tau(y[rows_split], d[rows_split])
        if np.isnan(node.tau):
            node.tau = 0.0
        if depth >= self.max_depth or rows_split.size < 4 * self.min_leaf:
            return node

        p = X.shape[1]
        m = self.mtry or max(1, int(np.ceil(np.sqrt(p))))
        feats = rng.choice(p, size=min(m, p), replace=False)
        best = (0.0, None, None)
        for f in feats:
            vals = np.unique(np.quantile(X[rows_split, f], np.linspace(0.1, 0.9, 9)))
            for thr in vals:
                lm = X[rows_split, f] <= thr
                lsp, rsp = rows_split[lm], rows_split[~lm]
                if lsp.size < 2 * self.min_leaf or rsp.size < 2 * self.min_leaf:
                    continue
                tl, tr = _tau(y[lsp], d[lsp]), _tau(y[rsp], d[rsp])
                if np.isnan(tl) or np.isnan(tr):
                    continue
                # Athey-Imbens criterion: reward heterogeneity between children,
                # less GRF's imbalance regularizer.
                score = lsp.size * rsp.size / rows_split.size * (tl - tr) ** 2
                if self.imbalance_penalty:
                    score -= self.imbalance_penalty * (1.0 / lsp.size + 1.0 / rsp.size)
                if score > best[0]:
                    best = (score, f, thr)

        if best[1] is None:
            return node
        _, f, thr = best
        node.feature, node.threshold = int(f), float(thr)
        lsp = rows_split[X[rows_split, f] <= thr]
        rsp = rows_split[X[rows_split, f] > thr]
        les = rows_est[X[rows_est, f] <= thr]
        res = rows_est[X[rows_est, f] > thr]
        node.left = self._grow(X, y, d, lsp, les, depth + 1, rng)
        node.right = self._grow(X, y, d, rsp, res, depth + 1, rng)
        return node

    def fit(self, X, y, d):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        y = np.asarray(y, dtype=float).ravel()
        d = np.asarray(d, dtype=float).ravel()
        n = y.size
        if X.shape[0] != n or d.size != n:
            raise ValueError("X, y, d must share their first dimension.")
        if not np.all(np.isin(d, (0.0, 1.0))):
            raise ValueError("d must be binary 0/1.")
        if n < 8 * self.min_leaf:
            raise ValueError(f"need at least {8 * self.min_leaf} observations, got {n}.")

        rng = np.random.default_rng(self.seed)
        self.trees_, self.in_bag_ = [], []
        m = max(4 * self.min_leaf, int(self.subsample * n))
        for _ in range(self.n_trees):
            idx = rng.choice(n, size=min(m, n), replace=False)
            half = idx.size // 2
            self.trees_.append(self._grow(X, y, d, idx[:half], idx[half:], 0, rng))
            mask = np.zeros(n, dtype=bool)
            mask[idx] = True
            self.in_bag_.append(mask)
        self._X, self._n = X, n
        return self

    @staticmethod
    def _walk(node, x):
        while node.feature is not None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.tau

    def predict(self, X=None, oob=False):
        if not self.trees_:
            raise ValueError("fit the forest before predicting.")
        if oob:
            if X is not None:
                raise ValueError("out-of-bag predictions are only defined on the training rows.")
            out = np.full(self._n, np.nan)
            for i in range(self._n):
                vals = [
                    self._walk(t, self._X[i])
                    for t, bag in zip(self.trees_, self.in_bag_)
                    if not bag[i]
                ]
                if vals:
                    out[i] = float(np.mean(vals))
            return out
        Xq = self._X if X is None else np.asarray(X, dtype=float)
        if Xq.ndim == 1:
            Xq = Xq[:, None]
        return np.array([float(np.mean([self._walk(t, row) for t in self.trees_])) for row in Xq])


def cheatsheet():
    return "_cforest: honest causal forest -- split on child-tau heterogeneity, fill leaves on held-out half"
