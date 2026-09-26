# morie.fn -- function file (rootcoder007/morie)
"""Shared honest causal-forest core (Wager-Athey style)."""

from . import _array_core as np

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

    @staticmethod
    def _tau_rows(rows, y, d):
        """Difference in means over plain row indices; NaN when an arm is empty."""
        st = sc = 0.0
        nt = nc = 0
        for i in rows:
            if d[i] == 1.0:
                st += y[i]
                nt += 1
            else:
                sc += y[i]
                nc += 1
        if nt == 0 or nc == 0:
            return float("nan")
        return st / nt - sc / nc

    @staticmethod
    def _quantiles(vals, qs):
        """numpy's default (linear) quantiles of a list."""
        v = sorted(vals)
        n = len(v)
        out = []
        for q in qs:
            pos = (n - 1) * q
            lo = int(pos)
            hi = min(lo + 1, n - 1)
            out.append(v[lo] + (pos - lo) * (v[hi] - v[lo]))
        return out

    def _grow(self, X, y, d, rows_split, rows_est, depth, rng):
        # X is a list of row lists, y and d plain lists, rows lists of ints;
        # the arithmetic is the array version's, without per-element array
        # indexing (which dominated the run time)
        node = _Node()
        node.n = len(rows_est)
        node.tau = self._tau_rows(rows_est, y, d)
        if node.tau != node.tau:
            node.tau = self._tau_rows(rows_split, y, d)
        if node.tau != node.tau:
            node.tau = 0.0
        if depth >= self.max_depth or len(rows_split) < 4 * self.min_leaf:
            return node
        p = len(X[0])
        m = self.mtry or max(1, int(np.ceil(np.sqrt(p))))
        feats = [int(v) for v in rng.choice(p, size=min(m, p), replace=False).tolist()]
        best = (0.0, None, None)
        ns = len(rows_split)
        for f in feats:
            col = [X[i][f] for i in rows_split]
            vals = sorted(set(self._quantiles(col, [0.1 * k for k in range(1, 10)])))
            for thr in vals:
                lsp = [i for i, v in zip(rows_split, col) if v <= thr]
                rsp = [i for i, v in zip(rows_split, col) if v > thr]
                # Wager and Athey (2018): every leaf holds at least min_leaf
                # units of EACH arm, so both child effects are estimable
                ltr = sum(1 for i in lsp if d[i] == 1.0)
                rtr = sum(1 for i in rsp if d[i] == 1.0)
                if min(ltr, len(lsp) - ltr, rtr, len(rsp) - rtr) < self.min_leaf:
                    continue
                tl, tr = self._tau_rows(lsp, y, d), self._tau_rows(rsp, y, d)
                if tl != tl or tr != tr:
                    continue
                # Athey-Imbens criterion: reward heterogeneity between children,
                # less GRF's imbalance regularizer.
                score = len(lsp) * len(rsp) / ns * (tl - tr) ** 2
                if self.imbalance_penalty:
                    score -= self.imbalance_penalty * (1.0 / len(lsp) + 1.0 / len(rsp))
                if score > best[0]:
                    best = (score, f, thr)
        if best[1] is None:
            return node
        _, f, thr = best
        node.feature, node.threshold = int(f), float(thr)
        lsp = [i for i in rows_split if X[i][f] <= thr]
        rsp = [i for i in rows_split if X[i][f] > thr]
        les = [i for i in rows_est if X[i][f] <= thr]
        res = [i for i in rows_est if X[i][f] > thr]
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
        Xl = [[float(v) for v in row] for row in X.tolist()]
        yl = [float(v) for v in y.tolist()]
        dl = [float(v) for v in d.tolist()]
        for _ in range(self.n_trees):
            idx = [int(v) for v in rng.choice(n, size=min(m, n), replace=False).tolist()]
            half = len(idx) // 2
            self.trees_.append(self._grow(Xl, yl, dl, idx[:half], idx[half:], 0, rng))
            mask = [False] * n
            for i in idx:
                mask[i] = True
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
            Xl = [[float(v) for v in row] for row in self._X.tolist()]
            out = []
            for i in range(self._n):
                vals = [
                    self._walk(t, Xl[i])
                    for t, bag in zip(self.trees_, self.in_bag_)
                    if not bag[i]
                ]
                out.append(sum(vals) / len(vals) if vals else float("nan"))
            return np.array(out)
        Xq = self._X if X is None else np.asarray(X, dtype=float)
        if Xq.ndim == 1:
            Xq = Xq[:, None]
        rows = [[float(v) for v in row] for row in Xq.tolist()]
        return np.array([sum(self._walk(t, row) for t in self.trees_) / len(self.trees_)
                         for row in rows])


def cheatsheet():
    return "_cforest: honest causal forest -- split on child-tau heterogeneity, fill leaves on held-out half"
