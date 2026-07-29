# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bagging aggregator -- mean of bootstrap-trained predictors."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_bagging_predictor"]

_METHOD = "Bagging ensemble aggregation"


def geron_bagging_predictor(predictions, aggregate="mean"):
    r"""Aggregate the predictions of ``B`` bootstrap-trained predictors.

    .. math::
        h_{\text{bag}}(x) = \frac{1}{B}\sum_{b=1}^{B} h_b(x)

    Bagging leaves the bias of a single predictor roughly where it was
    and cuts the variance, by a factor approaching :math:`B` when the
    predictors are uncorrelated.  The reported ``mean_disagreement`` is
    the across-predictor variance -- if it is near zero the ensemble is
    buying nothing, because the bootstrap samples produced the same tree
    every time.

    Parameters
    ----------
    predictions : array-like, shape (B, m)
        Row ``b`` holds predictor ``b``'s output on all ``m`` instances.
    aggregate : {"mean", "median", "vote"}, optional
        ``"mean"`` for regression (soft voting), ``"median"`` for a
        robust variant, ``"vote"`` for hard majority voting.

    Returns
    -------
    RichResult
        Payload keys ``prediction``, ``per_instance_variance``,
        ``mean_disagreement``, ``n_predictors``, ``se``, ``estimate``
        (mean of the aggregated predictions), ``n``, ``method``.

    References
    ----------
    Géron Ch 6, Bagging and Pasting section.

    Examples
    --------
    >>> r = geron_bagging_predictor([[1.0, 2.0], [3.0, 4.0]])
    >>> r["prediction"]
    [2.0, 3.0]
    >>> r["per_instance_variance"]   # ddof=1 across the two predictors
    [2.0, 2.0]

    Hard voting takes the modal label:

    >>> v = geron_bagging_predictor([[0, 1], [1, 1], [1, 0]], aggregate="vote")
    >>> v["prediction"]
    [1.0, 1.0]
    """
    P = np.atleast_2d(np.asarray(predictions, dtype=float))
    if P.size == 0:
        raise ValueError("predictions is empty.")
    if P.ndim != 2:
        raise ValueError(f"predictions must be 2-D (B, m), got ndim={P.ndim}.")
    if not np.all(np.isfinite(P)):
        raise ValueError("predictions contains non-finite values.")
    B, m = P.shape

    if aggregate == "mean":
        agg = P.mean(axis=0)
    elif aggregate == "median":
        agg = np.median(P, axis=0)
    elif aggregate == "vote":
        agg = np.empty(m, dtype=float)
        for j in range(m):
            vals, counts = np.unique(P[:, j], return_counts=True)
            agg[j] = vals[np.argmax(counts)]
    else:
        raise ValueError(
            f"aggregate must be one of 'mean', 'median', 'vote', got {aggregate!r}."
        )

    var = P.var(axis=0, ddof=1) if B > 1 else np.zeros(m)
    se = float(np.sqrt(var.mean() / B)) if B > 1 else float("nan")

    return RichResult(
        title="Bagging ensemble",
        summary_lines=[("Predictors", B), ("Instances", m)],
        payload={
            "prediction": agg.tolist(),
            "per_instance_variance": var.tolist(),
            "mean_disagreement": float(var.mean()),
            "n_predictors": int(B),
            "aggregate": aggregate,
            "se": se,
            "estimate": float(agg.mean()),
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grbag: bagging aggregator -- h_bag(x) = mean_b h_b(x), with disagreement diagnostics"
