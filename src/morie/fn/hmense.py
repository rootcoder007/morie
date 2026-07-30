# morie.fn -- function file (rootcoder007/morie)
"""Ensemble combination of several models' predictions."""

import numpy as np

from ._richresult import RichResult

__all__ = ["ensemble_evaluate", "geron_ensemble_eval"]


def ensemble_evaluate(predictions, y=None, weights=None, task="regression"):
    r"""Combine model predictions and show what the ensembling bought.

    For regression the combination is a weighted mean; for
    classification, a majority vote on labels.

    The reason an ensemble helps is decomposable, and both parts are
    reported. For a simple average of :math:`M` models the squared
    error obeys

    .. math::
       \mathrm{err}_{ens} = \overline{\mathrm{err}} - \overline{\mathrm{amb}},

    where :math:`\overline{\mathrm{amb}}` is the average AMBIGUITY, the
    mean squared spread of the members about the ensemble. So the
    ensemble can never be worse than its average member, and the margin
    is exactly how much the members DISAGREE. Averaging near-identical
    models buys nothing; ``diversity`` measures whether there was any
    to collect.

    Parameters
    ----------
    predictions : array-like, shape (M, n)
    y : array-like, shape (n,), optional
        Enables the error decomposition.
    weights : array-like, shape (M,), optional
    task : {'regression', 'classification'}

    Returns
    -------
    RichResult
        ``prediction``, ``diversity``, and with ``y``:
        ``ensemble_error``, ``mean_member_error``, ``ambiguity``,
        ``decomposition_residual``, ``best_member_error``.

    References
    ----------
    Geron (2022), *Hands-On Machine Learning*, 3rd ed., chapter 7.
    Krogh and Vedelsby (1995) for the ambiguity decomposition.

    Examples
    --------
    >>> out = ensemble_evaluate([[1.0, 2.0], [3.0, 4.0]])
    >>> out["prediction"].tolist()
    [2.0, 3.0]
    """
    P = np.atleast_2d(np.asarray(predictions, dtype=float))
    M, n = P.shape
    if M < 2:
        raise ValueError("need at least 2 members, got %d." % M)
    if task not in ("regression", "classification"):
        raise ValueError(
            "task must be 'regression' or 'classification', got %r." % task
        )
    w = np.ones(M) / M if weights is None else np.asarray(
        weights, dtype=float
    ).ravel()
    if w.size != M:
        raise ValueError(
            "weights has %d entries for %d members." % (w.size, M)
        )
    if np.any(w < 0):
        raise ValueError("weights must be non-negative.")
    s = w.sum()
    if s <= 0:
        raise ValueError("weights must not all be zero.")
    w = w / s

    if task == "regression":
        pred = w @ P
        diversity = float(np.mean(np.var(P, axis=0)))
    else:
        labs = np.unique(P)
        counts = np.stack([
            (P == lab).T.astype(float) @ w for lab in labs
        ], axis=1)
        pred = labs[np.argmax(counts, axis=1)]
        diversity = float(np.mean(1.0 - counts.max(axis=1)))

    payload = {
        "estimate": pred,
        "prediction": pred,
        "diversity": diversity,
        "diversity_note": (
            "the ensemble's advantage over its average member IS the "
            "members' disagreement; averaging near-identical models buys "
            "nothing"
        ),
        "weights": w,
        "n_members": int(M),
        "n": int(n),
        "task": task,
        "method": "Ensemble combination (%s)" % (
            "weighted mean" if task == "regression" else "majority vote"),
    }
    if y is not None:
        yv = np.asarray(y, dtype=float).ravel()
        if yv.size != n:
            raise ValueError(
                "y has %d entries for %d predictions." % (yv.size, n)
            )
        if task == "regression":
            memb = np.array([float(np.mean((P[m] - yv) ** 2))
                             for m in range(M)])
            ens = float(np.mean((pred - yv) ** 2))
            amb = float(np.mean(w @ (P - pred[None, :]) ** 2))
            payload.update({
                "ensemble_error": ens,
                "mean_member_error": float(w @ memb),
                "ambiguity": amb,
                "decomposition_residual": float(
                    abs((w @ memb) - amb - ens)
                ),
                "decomposition_note": (
                    "err_ensemble = mean member error - ambiguity, exactly, "
                    "for a weighted mean; the residual is the check"
                ),
                "best_member_error": float(memb.min()),
                "member_errors": memb,
                "beats_average_member": bool(ens <= float(w @ memb) + 1e-12),
            })
        else:
            memb = np.array([float(np.mean(P[m] != yv)) for m in range(M)])
            payload.update({
                "ensemble_error": float(np.mean(pred != yv)),
                "mean_member_error": float(w @ memb),
                "best_member_error": float(memb.min()),
                "member_errors": memb,
            })
    return RichResult(payload=payload)


def cheatsheet():
    return (
        "hmense: ensemble mean or vote with the Krogh-Vedelsby "
        "error-minus-ambiguity decomposition"
    )


#: Catalogue alias for :func:`ensemble_evaluate`.
geron_ensemble_eval = ensemble_evaluate
