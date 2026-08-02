# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Histogram-based gradient boosting (HistGB): bin features before split search."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_histogram_gradient_boosting"]

_METHOD = "Histogram-based gradient boosting (squared loss)"


def _best_split(binned, grad, hess, n_bins, min_samples_leaf):
    """Best (feature, bin) split by histogram scan.

    Returns ``(gain, feature, bin_threshold)``; gain is the squared-loss
    reduction ``GL^2/HL + GR^2/HR - G^2/H``.
    """
    m, n_feat = binned.shape
    G, H = float(grad.sum()), float(hess.sum())
    parent = G * G / H if H > 0 else 0.0
    best = (0.0, -1, -1)
    for j in range(n_feat):
        gh = np.bincount(binned[:, j], weights=grad, minlength=n_bins)
        hh = np.bincount(binned[:, j], weights=hess, minlength=n_bins)
        gl = np.cumsum(gh)[:-1]
        hl = np.cumsum(hh)[:-1]
        gr = G - gl
        hr = H - hl
        ok = (hl >= min_samples_leaf) & (hr >= min_samples_leaf)
        if not np.any(ok):
            continue
        gain = np.where(ok, gl**2 / np.where(hl > 0, hl, 1.0) + gr**2 / np.where(hr > 0, hr, 1.0) - parent, -np.inf)
        b = int(np.argmax(gain))
        if gain[b] > best[0]:
            best = (float(gain[b]), j, b)
    return best


def _grow(binned, grad, hess, rows, depth, max_depth, n_bins, min_samples_leaf):
    """Grow one regression tree over binned features; returns a node dict."""
    g, h = grad[rows], hess[rows]
    if depth >= max_depth or rows.size < 2 * min_samples_leaf:
        return {"leaf": float(g.sum() / h.sum()) if h.sum() > 0 else 0.0}
    gain, j, b = _best_split(binned[rows], g, h, n_bins, min_samples_leaf)
    if j < 0 or gain <= 1e-12:
        return {"leaf": float(g.sum() / h.sum()) if h.sum() > 0 else 0.0}
    left_mask = binned[rows, j] <= b
    left, right = rows[left_mask], rows[~left_mask]
    if left.size == 0 or right.size == 0:
        return {"leaf": float(g.sum() / h.sum()) if h.sum() > 0 else 0.0}
    return {
        "feature": j,
        "bin": b,
        "gain": gain,
        "left": _grow(binned, grad, hess, left, depth + 1, max_depth, n_bins, min_samples_leaf),
        "right": _grow(binned, grad, hess, right, depth + 1, max_depth, n_bins, min_samples_leaf),
    }


def _predict_tree(node, binned):
    out = np.empty(binned.shape[0])
    stack = [(node, np.arange(binned.shape[0]))]
    while stack:
        nd, rows = stack.pop()
        if "leaf" in nd:
            out[rows] = nd["leaf"]
            continue
        mask = binned[rows, nd["feature"]] <= nd["bin"]
        stack.append((nd["left"], rows[mask]))
        stack.append((nd["right"], rows[~mask]))
    return out


def geron_histogram_gradient_boosting(X, y, max_iter=100, learning_rate=0.1, max_bins=255, max_depth=3, min_samples_leaf=1):
    """
    Histogram-based gradient boosting (HistGB): bin features before split search.

    Formula: use histograms of size H to approximate best split

    Ordinary gradient boosting sorts every feature at every node to find
    the best threshold: ``O(m log m)`` per feature per node.  HistGB bins
    each feature *once*, up front, into at most ``max_bins`` quantile
    bins, and then a split search is a ``O(max_bins)`` scan over a
    histogram of accumulated gradients.  Once binned, the cost stops
    depending on ``m`` at all -- that is the entire trick, and it is why
    HistGB is the default choice on large data.

    The approximation is the binning and nothing else: a threshold
    strictly inside a bin can no longer be chosen.  With ``max_bins``
    at or above the number of distinct values the binning is exact and
    the result matches exhaustive boosting; ``bins_used`` reports which
    regime you are in.

    Squared loss, so gradients are residuals and Hessians are 1, and
    each leaf value is the mean residual of its rows.

    Parameters
    ----------
    X : array-like, shape (m, n)
        Features.
    y : array-like, shape (m,)
        Targets.
    max_iter : int
        Number of boosting rounds (trees).
    learning_rate : float
        Shrinkage applied to each tree, in (0, 1].
    max_bins : int
        Bins per feature, ``2 <= max_bins``.
    max_depth : int
        Depth of each tree.
    min_samples_leaf : int
        Minimum rows per leaf.

    Returns
    -------
    result : RichResult
        Keys: prediction, train_mse, mse_history, baseline, trees,
        bins_used, bin_edges, estimate, n, method.

    Examples
    --------
    A step function is learned exactly; boosting drives the training MSE
    to zero:

    >>> X = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
    >>> y = [0.0, 0.0, 5.0, 5.0, 10.0, 10.0]
    >>> r = geron_histogram_gradient_boosting(X, y, max_iter=50, learning_rate=0.3)
    >>> bool(r["train_mse"] < 1e-6)
    True

    The first prediction is the mean, and each round can only lower the
    training MSE:

    >>> round(r["baseline"], 6)
    5.0
    >>> bool(np.all(np.diff(r["mse_history"]) <= 1e-12))
    True

    Binning is exact when there are fewer distinct values than bins:

    >>> r["bins_used"]
    [6]

    Fewer bins than distinct values is where the approximation bites --
    with two bins the six x values collapse to two groups:

    >>> c = geron_histogram_gradient_boosting(X, y, max_iter=50, learning_rate=0.3, max_bins=2)
    >>> c["bins_used"]
    [2]
    >>> bool(c["train_mse"] > r["train_mse"])
    True

    References
    ----------
    Géron Ch 6
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_histogram_gradient_boosting: X must be a non-empty 2-D array, got shape {A.shape}")
    yy = np.asarray(y, dtype=float).ravel()
    if yy.size != A.shape[0]:
        raise ValueError(
            f"geron_histogram_gradient_boosting: X has {A.shape[0]} rows but y has {yy.size} entries"
        )
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(yy)):
        raise ValueError("geron_histogram_gradient_boosting: X and y must be finite")
    rounds = int(max_iter)
    if rounds < 1:
        raise ValueError(f"geron_histogram_gradient_boosting: max_iter must be at least 1, got {max_iter!r}")
    lr = float(learning_rate)
    if not (0.0 < lr <= 1.0):
        raise ValueError(f"geron_histogram_gradient_boosting: learning_rate must lie in (0, 1], got {learning_rate!r}")
    bins = int(max_bins)
    if bins < 2:
        raise ValueError(f"geron_histogram_gradient_boosting: max_bins must be at least 2, got {max_bins!r}")
    depth = int(max_depth)
    if depth < 1:
        raise ValueError(f"geron_histogram_gradient_boosting: max_depth must be at least 1, got {max_depth!r}")
    msl = int(min_samples_leaf)
    if msl < 1:
        raise ValueError(f"geron_histogram_gradient_boosting: min_samples_leaf must be at least 1, got {min_samples_leaf!r}")

    m, n_feat = A.shape
    binned = np.empty((m, n_feat), dtype=np.int64)
    edges = []
    used = []
    for j in range(n_feat):
        col = A[:, j]
        distinct = np.unique(col)
        if distinct.size <= bins:
            e = distinct[:-1] if distinct.size > 1 else np.asarray([])
        else:
            qs = np.linspace(0.0, 1.0, bins + 1)[1:-1]
            e = np.unique(np.quantile(col, qs))
        edges.append(e)
        binned[:, j] = np.searchsorted(e, col, side="left") if e.size else 0
        used.append(int(binned[:, j].max()) + 1)
    n_bin_slots = max(max(used), 2)

    baseline = float(np.mean(yy))
    pred = np.full(m, baseline)
    hess = np.ones(m)
    trees = []
    history = [float(np.mean((pred - yy) ** 2))]
    rows_all = np.arange(m)
    for _ in range(rounds):
        grad = yy - pred  # negative gradient of 0.5*(y - F)^2
        tree = _grow(binned, grad, hess, rows_all, 0, depth, n_bin_slots, msl)
        step = _predict_tree(tree, binned)
        pred = pred + lr * step
        trees.append(tree)
        history.append(float(np.mean((pred - yy) ** 2)))

    mse = history[-1]

    return RichResult(
        title="Histogram gradient boosting",
        summary_lines=[
            ("Rounds", rounds),
            ("Bins per feature", used),
            ("Learning rate", lr),
            ("Training MSE", mse),
        ],
        warnings=(
            [
                f"max_bins={bins} is below the number of distinct values in some feature, so thresholds "
                f"inside a bin are unreachable; that is the only approximation HistGB makes."
            ]
            if any(u >= bins for u in used)
            else []
        ),
        interpretation=(
            "After binning, a split search costs O(max_bins) per feature instead of O(m log m); "
            "the cost stops scaling with the number of rows."
        ),
        payload={
            "prediction": pred,
            "train_mse": mse,
            "mse_history": np.asarray(history),
            "baseline": baseline,
            "trees": trees,
            "bins_used": used,
            "bin_edges": edges,
            "learning_rate": lr,
            "estimate": mse,
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmhgb: histogram gradient boosting -- quantile-bin once, then O(bins) histogram split scans"
