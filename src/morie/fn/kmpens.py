# morie.fn -- function file (rootcoder007/morie)
"""Prompt ensembling."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["prompt_ensemble", "kamath_prompt_ensemble"]


def prompt_ensemble(probabilities, y=None, weights=None, method="mean"):
    r"""Average class probabilities over several prompt phrasings.

    Few-shot prompting is unstable: the same task, reworded, can move
    accuracy by tens of points, and the ordering of the examples alone
    is enough to do it. Prompt ensembling treats the phrasing as a
    NUISANCE to be averaged out rather than a hyperparameter to be
    tuned on a validation set -- which matters, because tuning the
    phrasing on held-out data is how few-shot results end up
    irreproducible.

    ``method='mean'`` averages probabilities; ``'logmean'`` averages
    log-probabilities (the geometric mean), which is the more
    conservative of the two because a single prompt assigning near-zero
    probability vetoes the class. Which is right depends on whether a
    confident dissenter should be able to veto.

    ``prompt_variance`` is what the ensemble is buying: the spread of
    per-prompt predictions. Small variance means the phrasings agreed
    and the ensemble was unnecessary; large variance means single-prompt
    results from this setup should not be reported without it.

    Parameters
    ----------
    probabilities : array-like, shape (P, n, C)
        Class probabilities under each of ``P`` prompts.
    y : array-like, shape (n,), optional
    weights : array-like, shape (P,), optional
    method : {'mean', 'logmean'}

    Returns
    -------
    RichResult
        ``prediction``, ``proba``, ``prompt_variance``,
        ``per_prompt_accuracy`` and ``spread`` when ``y`` is given.

    References
    ----------
    Kamath, Keenan, Somers and Sorenson (2024), *Large Language
    Models: A Deep Dive*, Springer, chapter 3, prompt ensembling.
    Zhao et al. (2021), "Calibrate before use", ICML, on prompt
    instability.

    Examples
    --------
    >>> P = [[[0.8, 0.2]], [[0.4, 0.6]]]
    >>> [round(float(v), 2) for v in prompt_ensemble(P)["proba"][0]]
    [0.6, 0.4]
    """
    Q = np.asarray(probabilities, dtype=float)
    if Q.ndim != 3:
        raise ValueError(
            "probabilities must be (P, n, C), got %d dimensions." % Q.ndim
        )
    P, n, C = Q.shape
    if P < 2:
        raise ValueError("need at least 2 prompts, got %d." % P)
    if np.any(Q < -1e-9) or np.any(Q > 1 + 1e-9):
        raise ValueError("probabilities must lie in [0, 1].")
    if not np.allclose(Q.sum(axis=2), 1.0, atol=1e-6):
        raise ValueError("each prompt's probabilities must sum to 1.")
    if method not in ("mean", "logmean"):
        raise ValueError("method must be 'mean' or 'logmean', got %r." % method)
    w = np.ones(P) / P if weights is None else np.asarray(
        weights, dtype=float
    ).ravel()
    if w.size != P:
        raise ValueError("weights has %d entries for %d prompts." % (w.size, P))
    if np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be non-negative and not all zero.")
    w = w / w.sum()

    if method == "mean":
        proba = np.tensordot(w, Q, axes=(0, 0))
    else:
        lg = np.tensordot(w, np.log(np.clip(Q, 1e-300, None)), axes=(0, 0))
        ex = np.exp(lg - lg.max(axis=1, keepdims=True))
        proba = ex / ex.sum(axis=1, keepdims=True)
    pred = np.argmax(proba, axis=1)
    per_pred = np.argmax(Q, axis=2)

    payload = {
        "estimate": pred,
        "prediction": pred,
        "proba": proba,
        "prompt_variance": float(np.mean(Q.var(axis=0))),
        "variance_note": (
            "the spread across phrasings is what the ensemble buys; near "
            "zero means the prompts agreed and the ensembling was idle"
        ),
        "unanimity": float(np.mean(
            (per_pred == per_pred[0][None, :]).all(axis=0)
        )),
        "method_used": method,
        "logmean_note": (
            "logmean is the geometric mean, so one prompt assigning near-zero "
            "probability vetoes the class; mean lets the majority carry it"
        ),
        "n_prompts": int(P),
        "n": int(n),
        "method": "Prompt ensemble (%s)" % method,
    }
    if y is not None:
        yv = np.asarray(y).ravel()
        if yv.size != n:
            raise ValueError("y has %d entries for %d rows." % (yv.size, n))
        acc = np.array([float(np.mean(per_pred[i] == yv)) for i in range(P)])
        payload.update({
            "accuracy": float(np.mean(pred == yv)),
            "per_prompt_accuracy": acc,
            "best_prompt_accuracy": float(acc.max()),
            "worst_prompt_accuracy": float(acc.min()),
            "spread": float(acc.max() - acc.min()),
            "spread_note": (
                "the gap between the best and worst phrasing of the same "
                "task; tuning the prompt on held-out data to close it is how "
                "few-shot numbers stop reproducing"
            ),
        })
    return RichResult(payload=payload)


def cheatsheet():
    return (
        "kmpens: prompt ensembling by arithmetic or geometric mean, "
        "reporting the phrasing spread it is averaging out"
    )


#: Catalogue alias for :func:`prompt_ensemble`.
kamath_prompt_ensemble = prompt_ensemble
