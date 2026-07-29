# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Out-of-bag evaluation for a bagged ensemble."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_oob_score"]


def geron_oob_score(X, y, models, task="auto"):
    """
    Out-of-bag (OOB) evaluation using the unsampled observations per bootstrap.

    Formula: OOB score = avg_i accuracy(f_m(x_i)) over m not sampling x_i

    A bootstrap of size m leaves out (1 - 1/m)^m -> 1/e ~ 37 % of the
    rows, so every row has its own held-out sub-ensemble and NO
    validation split is needed -- every instance trains some estimators
    and scores the rest. The estimate is pessimistic for the full
    ensemble, because each row is judged by roughly a third of the
    estimators; ``mean_oob_votes`` says how many that was, and a small
    number means the score is both noisy and biased low.

    Each entry of ``models`` is ``(predict, in_bag)``, or a mapping with
    those keys, where ``in_bag`` is a boolean mask over the rows or the
    array of sampled row indices.

    Parameters
    ----------
    X : array-like, shape (m, n)
    y : array-like, shape (m,)
    models : sequence
        ``(predict, in_bag)`` pairs as described.
    task : {"auto", "classification", "regression"}, default "auto"

    Returns
    -------
    result : RichResult
        Keys: oob_score, oob_predictions, covered, mean_oob_votes,
        estimate, n, method.

    Examples
    --------
    Two estimators that each read the feature straight off, each trained
    on one row: every row is scored by the estimator that did not see it.

    >>> f = lambda A: np.asarray(A, dtype=float)[:, 0]
    >>> r = geron_oob_score([[0.0], [1.0]], [0, 1], [(f, [True, False]), (f, [False, True])])
    >>> float(r["oob_score"]), float(r["mean_oob_votes"])
    (1.0, 1.0)

    An estimator that always says 1 is wrong on the row it never saw:

    >>> g = lambda A: np.ones(len(np.atleast_2d(A)))
    >>> float(geron_oob_score([[0.0], [1.0]], [0, 1],
    ...                       [(g, [True, False]), (g, [False, True])])["oob_score"])
    0.5

    References
    ----------
    Geron Ch 6
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_oob_score: X must be a non-empty 2-D array, got shape {A.shape}")
    yv = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    n = A.shape[0]
    if yv.size != n:
        raise ValueError(f"geron_oob_score: X has {n} rows but y has {yv.size} entries")
    ms = list(models)
    if not ms:
        raise ValueError("geron_oob_score: models is empty")
    if task not in ("auto", "classification", "regression"):
        raise ValueError(f"geron_oob_score: task must be auto, classification or regression, got {task!r}")
    classify = task == "classification" or (task == "auto" and set(np.unique(yv).tolist()) <= {0.0, 1.0})

    total = np.zeros(n)
    votes = np.zeros(n)
    for k, entry in enumerate(ms):
        if hasattr(entry, "get"):
            pred_fn, bag = entry.get("predict"), entry.get("in_bag")
        else:
            try:
                pred_fn, bag = entry
            except (TypeError, ValueError):
                raise ValueError(f"geron_oob_score: model {k} must be a (predict, in_bag) pair or a mapping") from None
        if not callable(pred_fn):
            raise ValueError(f"geron_oob_score: model {k} has no callable predict")
        b = np.asarray(bag)
        if b.dtype == bool:
            if b.size != n:
                raise ValueError(f"geron_oob_score: model {k} has an in_bag mask of length {b.size}, expected {n}")
            mask = b
        else:
            idx = b.astype(int).ravel()
            if idx.size and (idx.min() < 0 or idx.max() >= n):
                raise ValueError(f"geron_oob_score: model {k} has in_bag indices outside [0, {n})")
            mask = np.zeros(n, dtype=bool)
            mask[idx] = True
        oob = ~mask
        if not oob.any():
            continue
        p = np.asarray(pred_fn(A[oob]), dtype=float).ravel()
        if p.size != int(oob.sum()):
            raise ValueError(f"geron_oob_score: model {k} returned {p.size} predictions for {int(oob.sum())} OOB rows")
        total[oob] += p
        votes[oob] += 1

    covered = votes > 0
    if not covered.any():
        raise ValueError("geron_oob_score: every row was in every bag, so no OOB estimate exists")
    oob_pred = np.full(n, np.nan)
    oob_pred[covered] = total[covered] / votes[covered]
    if classify:
        hard = (oob_pred[covered] >= 0.5).astype(float)
        score = float(np.mean(hard == yv[covered]))
        label = "OOB accuracy"
    else:
        score = float(np.mean((oob_pred[covered] - yv[covered]) ** 2))
        label = "OOB MSE"

    return RichResult(
        title="Out-of-bag evaluation",
        summary_lines=[(label, score), ("Rows covered", float(np.mean(covered))), ("Mean OOB votes", float(votes[covered].mean()))],
        interpretation="Each row is judged by the ~37 % of estimators that excluded it, so the score runs pessimistic.",
        payload={
            "oob_score": score,
            "oob_predictions": oob_pred,
            "covered": covered,
            "votes": votes,
            "mean_oob_votes": float(votes[covered].mean()),
            "task": "classification" if classify else "regression",
            "estimate": score,
            "n": int(n),
            "method": f"Out-of-bag {'accuracy' if classify else 'MSE'} over per-estimator bag masks",
        },
    )


def cheatsheet():
    return "hmoob: Out-of-bag evaluation from per-estimator bag masks"
