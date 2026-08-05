# morie.fn -- wave 2 slice w2_00 (rootcoder007/morie)
"""Bootstrap aggregating (bagging) ensemble prediction.

Source: Breiman, L. (1996), "Bagging predictors", *Machine Learning*
24(2), 123-140, doi:10.1007/BF00058655.  Breiman gives two aggregation
rules and they are not interchangeable:

  * numerical prediction -- average the ensemble,
    ghat_bag(x) = (1/B) sum_b ghat_b(x);
  * classification -- vote, ghat_bag(x) = argmax_k #{b : ghat_b(x) = k}.

Averaging class labels is the classic bug: it returns 1.5 for a two-one
split between classes 1 and 2, a label that does not exist.  This module
therefore refuses to guess and takes ``kind`` explicitly.  Ties in the
vote go to the smaller label, so the rule is a function.

``models`` is a B-by-m matrix of predictions, one row per bootstrap
replicate, one column per new case: the fitted learners are supplied
already evaluated, which is what lets the Python and R arms be compared
at all.  ``X_new`` is used only to check the column count.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core  # noqa: F401

from ._richresult import RichResult

__all__ = ["boot_bagging_predict"]


def boot_bagging_predict(models, X_new=None, kind="regression"):
    """Aggregate the ensemble.

    Parameters
    ----------
    models : array-like
        B-by-m matrix of per-replicate predictions.
    X_new : array-like, optional
        The new cases; only its length is used, as a check.
    kind : {"regression", "classification"}
        Average or majority vote.

    Returns
    -------
    y_pred : the m aggregated predictions
    vote_share : for classification, the winning share; for regression,
        the ensemble standard deviation
    """
    rows = [[float(v) for v in r] for r in models]
    B = len(rows)
    if B == 0:
        raise ValueError("boot_bagging_predict: no models")
    m = len(rows[0])
    if m == 0:
        raise ValueError("boot_bagging_predict: no cases to predict")
    for r in rows:
        if len(r) != m:
            raise ValueError("boot_bagging_predict: the model matrix is ragged")
    if X_new is not None:
        try:
            nx = len(X_new)
        except TypeError:
            nx = m
        if nx != m:
            raise ValueError("boot_bagging_predict: X_new and the model matrix disagree on the case count")
    kd = str(kind).lower()
    if kd not in ("regression", "classification"):
        raise ValueError("boot_bagging_predict: kind must be regression or classification")
    yp = []
    share = []
    for j in range(m):
        col = [rows[b][j] for b in range(B)]
        if kd == "regression":
            s = 0.0
            for v in col:
                s += v
            mu = s / B
            yp.append(mu)
            ss = 0.0
            for v in col:
                d = v - mu
                ss += d * d
            share.append((ss / (B - 1.0)) ** 0.5 if B > 1 else 0.0)
        else:
            cnt = {}
            for v in col:
                cnt[v] = cnt.get(v, 0) + 1
            best = None
            for g in sorted(cnt):
                if best is None or cnt[g] > cnt[best]:
                    best = g
            yp.append(best)
            share.append(cnt[best] / (B + 0.0))
    return RichResult(
        title="Bagged ensemble prediction",
        summary_lines=[("B", B), ("cases", m)],
        payload={
            "y_pred": yp,
            "estimate": yp[0],
            "vote_share": share,
            "B": B,
            "m": m,
            "kind": kd,
            "method": "Breiman (1996) bagging: average for regression, plurality vote for classification",
        },
    )


def cheatsheet():
    return "btbg: Bootstrap aggregating (bagging) ensemble prediction"


# compact alias per ledger/NAMING.md
bootbaggingpredict = boot_bagging_predict
