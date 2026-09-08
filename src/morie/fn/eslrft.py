# morie.fn -- function file (rootcoder007/morie)
"""Random forest for regression, ESL Algorithm 15.1."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["esl_random_forest"]


def esl_random_forest(X, y, B=100, mtry=None, max_depth=8, min_node=5,
                      newdata=None, seed=0):
    r"""Random forest, ESL Algorithm 15.1.

    For ``b = 1..B``: draw a bootstrap sample :math:`Z^*` of size N,
    and grow a tree by repeating, **at each terminal node**, select
    ``m`` of the ``p`` variables at random, pick the best
    variable/split-point among those ``m``, and split. Predict by

    .. math:: \hat f_{rf}^B(x) = \frac1B \sum_{b=1}^B T_b(x).

    Two details in that algorithm carry the whole method:

    The variable subset is drawn **per node**, not per tree. Bagging
    alone already averages B identically distributed trees, so --
    as the book puts it -- the bias of bagged trees is the same as
    that of the individual trees and the only hope of improvement is
    variance reduction. Averaging B i.d. variables with pairwise
    correlation :math:`\rho` leaves :math:`\rho\sigma^2` behind, so
    the gain is bounded by how DECORRELATED the trees are. Choosing
    the subset once per tree decorrelates far less than choosing it
    at every node.

    The default ``m`` differs by task: :math:`\lfloor\sqrt p\rfloor`
    for classification and :math:`\lfloor p/3\rfloor` for regression.
    This module is the regression case and uses :math:`\lfloor
    p/3\rfloor`. The two rules cross at ``p = 9``, where both give
    3; below that the classification rule is larger and above it the
    regression rule is, by a widening margin -- 33 against 10 at
    ``p = 100``. Swapping them is therefore not a small perturbation
    in either direction. The book calls both tuning parameters.

    Out-of-bag error comes free: each observation is left out of
    about 36.8% of the trees (7.55), and averaging only those trees
    gives a nearly-cross-validated error at no extra fitting cost.

    Parameters
    ----------
    x : array-like, shape (N, p)
        Predictors.
    y : array-like, shape (N,)
        Numeric response.
    B : int, default 100
        Number of trees.
    mtry : int, optional
        Variables sampled per node; ``floor(p/3)`` when omitted.
    max_depth, min_node : int
        Stopping rules standing in for the book's ``n_min``.
    newdata : array-like, optional
        Points to predict; the training rows when omitted.
    seed : int, default 0
        Seed for the bootstrap draws and the per-node variable draws.

    Returns
    -------
    RichResult
        keys: ``prediction``, ``oob_prediction``, ``oob_mse``,
        ``train_mse``, ``mtry``, ``mtry_rule``, ``B``, ``n``, ``p``,
        ``n_oob_missing``, ``subset_drawn_per``, ``method``.

    References
    ----------
    Hastie, Tibshirani and Friedman, *The Elements of Statistical
    Learning*, 2nd ed., Ch. 15, Algorithm 15.1 and Sec. 15.2-15.3.
    Read from the PDF. Breiman (2001).
    """
    from ._esl import default_mtry, grow_tree, predict_tree

    A = np.atleast_2d(np.asarray(X, dtype=float))
    yv = np.asarray(y, dtype=float).ravel()
    if A.shape[0] != yv.size:
        A = A.T
    if A.shape[0] != yv.size:
        raise ValueError(f"X has {A.shape[0]} rows for {yv.size} responses.")
    n, p = A.shape
    if n < 4:
        raise ValueError(f"need at least 4 observations, got {n}.")
    Bn = int(B)
    if Bn < 1:
        raise ValueError(f"need at least one tree, got {Bn}.")
    m = default_mtry(p, "regression") if mtry is None else int(mtry)
    if not 1 <= m <= p:
        raise ValueError(f"mtry must lie in 1..{p}, got {m}.")
    Q = A if newdata is None else np.atleast_2d(
        np.asarray(newdata, dtype=float))
    if Q.shape[1] != p:
        raise ValueError(f"newdata has {Q.shape[1]} columns, expected {p}.")

    rng = np.random.default_rng(seed)
    pred = np.zeros(Q.shape[0])
    oob_sum = np.zeros(n)
    oob_cnt = np.zeros(n)
    for _ in range(Bn):
        rows = rng.integers(0, n, size=n)
        tree = grow_tree(A[rows], yv[rows], rng, m, max_depth, min_node)
        pred += predict_tree(tree, Q)
        out = np.setdiff1d(np.arange(n), rows, assume_unique=False)
        if out.size:
            oob_sum[out] += predict_tree(tree, A[out])
            oob_cnt[out] += 1
    pred /= Bn
    has_oob = oob_cnt > 0
    oob_pred = np.where(has_oob, oob_sum / np.maximum(oob_cnt, 1), np.nan)
    oob_mse = (float(np.mean((yv[has_oob] - oob_pred[has_oob]) ** 2))
               if has_oob.any() else np.nan)
    train_pred = pred if newdata is None else None
    return RichResult(payload={
        "prediction": pred,
        "oob_prediction": oob_pred, "oob_mse": oob_mse,
        "train_mse": (float(np.mean((yv - train_pred) ** 2))
                      if train_pred is not None else None),
        "mtry": int(m),
        "mtry_rule": "floor(p/3) for regression; floor(sqrt(p)) is the "
                     "CLASSIFICATION default. They cross at p = 9; above "
                     "it the regression rule is the larger of the two",
        "subset_drawn_per": "node",
        "why_per_node": "averaging B identically distributed trees leaves "
                        "rho sigma^2 behind, so the gain is bounded by how "
                        "decorrelated they are; a per-tree subset "
                        "decorrelates far less than a per-node one",
        "n_oob_missing": int((~has_oob).sum()),
        "B": Bn, "n": int(n), "p": int(p),
        "method": "ESL Algorithm 15.1 random forest for regression"})


def cheatsheet():
    return "eslrft: the m-of-p draw is PER NODE -- that is where the decorrelation comes from"
