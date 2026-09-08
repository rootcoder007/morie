# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""BERT next-sentence-prediction binary cross-entropy loss."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_bert_nsp_loss"]

_METHOD = "BERT next-sentence-prediction loss"


def geron_bert_nsp_loss(logits, labels):
    r"""Binary cross-entropy of the IsNext head.

    .. math::
        L_{\text{NSP}} = -\log p\bigl(\text{is\_next} \mid
            \text{segment}_A, \text{segment}_B\bigr)

    BERT reads the two-class head off the ``[CLS]`` token, which is the
    only position with no word of its own to predict, so NSP is what
    gives ``[CLS]`` a sequence-level meaning at all.  Half the training
    pairs are constructed with a random second segment, so a model that
    always answers "not next" scores ``log 2 = 0.693`` -- the number to
    beat, and reported here as ``baseline_loss``.  (RoBERTa later showed
    the task adds little; the mechanics are still worth being exact
    about.)

    Computed via log-softmax rather than ``log(softmax(...))``, so a
    confident correct answer does not underflow to ``-inf``.

    Parameters
    ----------
    logits : array-like, shape (n, 2) or (n,)
        Two-class logits, or a single logit for "is next".
    labels : array-like of {0, 1}, shape (n,)
        1 = IsNext.

    Returns
    -------
    RichResult
        Payload keys ``loss``, ``per_pair``, ``probabilities``,
        ``accuracy``, ``baseline_loss``, ``estimate``, ``n``,
        ``method``.

    References
    ----------
    Géron Ch 15, BERT pretraining (NSP) section.

    Examples
    --------
    Undecided logits cost ``log 2`` per pair:

    >>> r = geron_bert_nsp_loss([[0.0, 0.0], [0.0, 0.0]], [1, 0])
    >>> round(r["loss"], 6)
    0.693147
    >>> round(r["baseline_loss"], 6)
    0.693147

    A confident correct call is nearly free; a confident wrong one is
    expensive:

    >>> right = geron_bert_nsp_loss([[0.0, 5.0]], [1])["loss"]
    >>> wrong = geron_bert_nsp_loss([[0.0, 5.0]], [0])["loss"]
    >>> round(right, 6), round(wrong, 6)
    (0.006715, 5.006715)
    """
    Z = np.asarray(logits, dtype=float)
    if Z.ndim == 1:
        Z = np.stack([np.zeros_like(Z), Z], axis=1)
    Z = np.atleast_2d(Z)
    if Z.ndim != 2 or Z.shape[1] != 2 or Z.size == 0:
        raise ValueError(f"logits must be (n, 2) or (n,), got shape {Z.shape}.")
    if not np.all(np.isfinite(Z)):
        raise ValueError("logits contains non-finite values.")
    y = np.asarray(labels).ravel()
    if y.size != Z.shape[0]:
        raise ValueError(f"labels has {y.size} entries but logits has {Z.shape[0]} pairs.")
    uniq = set(np.unique(y).tolist())
    if not uniq <= {0, 1}:
        raise ValueError(f"NSP labels must be 0 or 1, got {sorted(uniq)}.")
    y = y.astype(int)

    M = Z.max(axis=1, keepdims=True)
    logp = (Z - M) - np.log(np.exp(Z - M).sum(axis=1, keepdims=True))
    per = -logp[np.arange(y.size), y]
    p = np.exp(logp)

    return RichResult(
        title="BERT NSP loss",
        summary_lines=[("Loss", float(per.mean())),
                       ("Accuracy", float(np.mean(np.argmax(p, axis=1) == y)))],
        payload={
            "loss": float(per.mean()),
            "per_pair": per.tolist(),
            "probabilities": p.tolist(),
            "accuracy": float(np.mean(np.argmax(p, axis=1) == y)),
            "baseline_loss": float(np.log(2.0)),
            "estimate": float(per.mean()),
            "n": int(y.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grnsp: -log p(IsNext) from the [CLS] head; log-softmax, and log 2 is the coin-flip baseline"
