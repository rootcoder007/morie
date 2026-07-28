# morie.fn -- internal helpers (rootcoder007/morie)
"""Shared machinery for the Hastie, Tibshirani and Friedman shelf.

Spec: Hastie, T., Tibshirani, R. and Friedman, J., *The Elements of
Statistical Learning*, 2nd ed., Springer. Equation numbers below are
the book's and were read off the PDF.

The recurring theme across the modules that use this file is that a
resampling estimate of prediction error is only honest if the point
being predicted was NOT used to fit the predictor. Eq. (7.54)'s
Err_boot violates that -- the bootstrap samples and the training set
overlap -- and the book shows the damage concretely: for a
1-nearest-neighbour rule on independent labels the true error rate is
0.5 and Err_boot expects 0.5 x 0.368 = 0.184. Everything else here
exists to repair that.
"""

import numpy as np

__all__ = ["inclusion_probability", "bootstrap_indices", "default_mtry",
           "grow_tree", "predict_tree", "gaussian_product_kernel_density",
           "squared_error"]

# 1 - e^{-1} to double precision: the limit of (7.55).
BOOTSTRAP_INCLUSION_LIMIT = 1.0 - np.exp(-1.0)


def inclusion_probability(n):
    r"""Eq. (7.55): :math:`\Pr\{i \in \text{bootstrap sample } b\} =
    1 - (1 - 1/n)^n \approx 1 - e^{-1} = 0.632`.

    The exact finite-n value, not the limit -- they differ by about
    1% at n = 20, which is exactly the regime where someone reaches
    for the bootstrap. The 0.632 in the name of the .632 estimator is
    this number.
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"need at least one observation, got {n}.")
    return float(1.0 - (1.0 - 1.0 / n) ** n)


def bootstrap_indices(n, B, seed=0):
    """``B`` bootstrap index vectors of length ``n``, drawn with
    replacement (Figure 7.12)."""
    n, B = int(n), int(B)
    if n < 2:
        raise ValueError(f"need at least 2 observations, got {n}.")
    if B < 1:
        raise ValueError(f"need at least one bootstrap replicate, got {B}.")
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(B, n))


def default_mtry(p, task="regression"):
    r"""The book's defaults for the number of variables sampled at
    each split: :math:`\lfloor \sqrt p \rfloor` for classification
    and :math:`\lfloor p/3 \rfloor` for regression, each with a floor
    of 1.

    These are NOT interchangeable, and which is larger depends on
    p: they cross at p = 9 (where both are 3), the classification
    rule is larger below it, and the REGRESSION rule is larger above
    -- 33 against 10 at p = 100. The book calls both tuning
    parameters, so they are a starting point rather than a rule.
    """
    p = int(p)
    if p < 1:
        raise ValueError(f"need at least one predictor, got {p}.")
    if task == "classification":
        return max(1, int(np.floor(np.sqrt(p))))
    if task == "regression":
        return max(1, int(np.floor(p / 3.0)))
    raise ValueError("task must be 'regression' or 'classification'.")


def grow_tree(X, y, rng, mtry, max_depth=8, min_node=5):
    """Algorithm 15.1 step 1(b): grow one random-forest tree.

    At EVERY node, ``mtry`` of the ``p`` variables are drawn at
    random and the best split is taken among those only. Drawing the
    subset once per tree instead of once per node is a different (and
    weaker) estimator -- the decorrelation between trees is what buys
    the variance reduction, and it comes from the per-node draw.

    Splitting minimises the within-node sum of squares, which for a
    fixed parent is equivalent to maximising the reduction in squared
    error.
    """
    return _grow(X, y, rng, int(mtry), 0, int(max_depth), int(min_node))


def _stop(y, depth, max_depth, min_node):
    return (depth >= max_depth or y.size <= min_node
            or float(np.var(y)) < 1e-12)


def _grow(X, y, rng, mtry, depth, max_depth, min_node):
    if _stop(y, depth, max_depth, min_node):
        return {"leaf": True, "value": float(np.mean(y))}
    p = X.shape[1]
    feats = rng.choice(p, size=min(mtry, p), replace=False)
    best = None
    for j in feats:
        col = X[:, j]
        cuts = np.unique(col)
        if cuts.size < 2:
            continue
        # midpoints, so the threshold does not sit on a data value
        cuts = (cuts[:-1] + cuts[1:]) / 2.0
        for t in cuts:
            left = col <= t
            nl = int(left.sum())
            if nl == 0 or nl == y.size:
                continue
            sse = (nl * np.var(y[left])
                   + (y.size - nl) * np.var(y[~left]))
            if best is None or sse < best[0]:
                best = (float(sse), int(j), float(t))
    if best is None:
        return {"leaf": True, "value": float(np.mean(y))}
    _, j, t = best
    left = X[:, j] <= t
    return {"leaf": False, "feature": j, "threshold": t,
            "left": _grow(X[left], y[left], rng, mtry, depth + 1,
                          max_depth, min_node),
            "right": _grow(X[~left], y[~left], rng, mtry, depth + 1,
                           max_depth, min_node)}


def predict_tree(node, X):
    """Vectorised descent of one tree."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    out = np.empty(X.shape[0])
    for i in range(X.shape[0]):
        nd = node
        while not nd["leaf"]:
            nd = nd["left"] if X[i, nd["feature"]] <= nd["threshold"] \
                else nd["right"]
        out[i] = nd["value"]
    return out


def gaussian_product_kernel_density(x0, data, lam):
    r"""Eq. (6.24), the p-dimensional Gaussian product kernel:

    .. math:: \hat f_X(x_0) = \frac1{N(2\lambda^2\pi)^{p/2}}
              \sum_{i=1}^N e^{-\frac12(\|x_i - x_0\|/\lambda)^2}.

    In one dimension this is (6.23), the Parzen estimate
    :math:`\frac1N\sum_i \phi_\lambda(x - x_i)` -- the convolution of
    the empirical distribution with a Gaussian of standard deviation
    :math:`\lambda`. The exponent on the normalising constant is
    ``p/2``, not ``p``; with ``p`` the estimate is off by a factor of
    :math:`(2\lambda^2\pi)^{p/2}`, which for :math:`\lambda` near 1
    is close enough to 1 in low dimension to pass a smell test and
    then grows with every added covariate. The integral check in the
    tests is what catches it.
    """
    D = np.atleast_2d(np.asarray(data, dtype=float))
    if D.shape[0] == 1 and D.shape[1] > 1:
        D = D.T
    N, p = D.shape
    lam = float(lam)
    if lam <= 0:
        raise ValueError(f"lambda must be positive, got {lam}.")
    Q = np.atleast_2d(np.asarray(x0, dtype=float))
    if Q.shape[1] != p:
        Q = Q.reshape(-1, p)
    d2 = ((Q[:, None, :] - D[None, :, :]) ** 2).sum(axis=2)
    norm = N * (2.0 * lam ** 2 * np.pi) ** (p / 2.0)
    return np.exp(-0.5 * d2 / lam ** 2).sum(axis=1) / norm


def squared_error(y, yhat):
    """The default loss ``L(y, f(x)) = (y - f(x))^2``."""
    y = np.asarray(y, dtype=float).ravel()
    yhat = np.asarray(yhat, dtype=float).ravel()
    if y.size != yhat.size:
        raise ValueError(
            f"y has {y.size} entries and the prediction has {yhat.size}.")
    return (y - yhat) ** 2


def cheatsheet():
    return ("_esl: Err_boot overlaps train and test and is biased LOW; "
            "Err^(1) fixes that and .632 corrects what it overcorrects")
