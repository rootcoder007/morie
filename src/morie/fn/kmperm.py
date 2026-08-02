# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Permutation language-model loss and two-stream masks (XLNet).

Yang, Dai, Yang, Carbonell, Salakhutdinov and Le (2019),
*XLNet: Generalized Autoregressive Pretraining for Language
Understanding*, arXiv:1906.08237. Equation numbers below are that
paper's.
"""

from . import _array_core as np

from ._richresult import RichResult

__all__ = [
    "kamath_permutation_lm_loss",
    "permutation_attention_masks",
]

_METHOD = "Permutation language-model loss (XLNet, eq 3 and 5)"


def _log_softmax(z):
    m = np.max(z, axis=-1, keepdims=True)
    d = z - m
    return d - np.log(np.sum(np.exp(d), axis=-1, keepdims=True))


def permutation_attention_masks(permutation):
    """Two-stream attention masks induced by a factorization order.

    Returns the pair of boolean matrices that make the permutation
    objective computable with a single forward pass over an unpermuted
    sequence. ``content[i, j]`` is True when position ``i`` may attend
    to position ``j``.

    The content stream sees ``z_{<=t}`` -- itself included -- and the
    query stream sees only ``z_{<t}``. The single difference is the
    diagonal, and that diagonal is the entire reason two streams are
    needed: a query that could see its own token would make predicting
    that token trivial, and a content stream that could not see it
    would have no way to encode it for later positions.

    Parameters
    ----------
    permutation : array-like of int
        A permutation ``z`` of ``0 .. T-1``. ``permutation[t]`` is the
        sequence position that comes ``t``-th in factorization order.

    Returns
    -------
    dict with ``content``, ``query``, ``rank``.
    """
    z = np.asarray(permutation, dtype=int).ravel()
    T = z.size
    if T == 0:
        raise ValueError("permutation must not be empty.")
    if not np.array_equal(np.sort(z), np.arange(T)):
        raise ValueError(
            "permutation must be a permutation of 0 .. T-1; got values "
            f"{np.sort(z).tolist()}."
        )
    # rank[p] = the step at which position p is generated
    rank = np.empty(T, dtype=int)
    rank[z] = np.arange(T)
    content = rank[:, None] >= rank[None, :]
    query = rank[:, None] > rank[None, :]
    return {"content": content, "query": query, "rank": rank}


def kamath_permutation_lm_loss(logits, targets, permutation,
                               num_predict=None, reduction="mean"):
    """Permutation language-model loss under one factorization order.

    Equation (3) of the XLNet paper is

    .. math::
        \\max_\\theta\\ \\mathbb{E}_{z \\sim \\mathcal{Z}_T}
        \\left[ \\sum_{t=1}^{T} \\log p_\\theta(x_{z_t} \\mid x_{z_{<t}}) \\right]

    and this returns the negated inner sum for a single sampled ``z``.

    A property worth stating plainly, because it is the most common
    misreading of the objective: **with the logits held fixed the total
    loss does not depend on the permutation at all.** Reordering the
    terms of a sum does not change the sum. The permutation enters only
    through what the network was allowed to condition on when producing
    those logits -- that is, through the attention masks returned by
    :func:`permutation_attention_masks` -- and never through the
    arithmetic here. ``order_invariant`` in the payload records this,
    and it is True by construction whenever the full sequence is
    scored.

    Equation (5) is partial prediction: split ``z`` at a cutting point
    ``c`` and score only the trailing ``z_{>c}``, whose contexts are
    the longest available. ``num_predict`` sets ``|z| - c``. Because
    the scored subset then depends on the permutation, the loss under
    partial prediction *is* order dependent, and ``order_invariant``
    goes False.

    Parameters
    ----------
    logits : array-like, shape (T, V)
        Unnormalised scores for each of ``T`` positions over a
        vocabulary of size ``V``.
    targets : array-like of int, shape (T,)
        The true token id at each sequence position.
    permutation : array-like of int, shape (T,)
        Factorization order ``z``.
    num_predict : int, optional
        Number of trailing positions in factorization order to score,
        as in eq (5). ``None`` scores all ``T``.
    reduction : {"mean", "sum", "none"}
        How to reduce the per-token negative log-likelihoods.

    Returns
    -------
    RichResult
        ``loss``, ``estimate`` (an alias), ``token_nll``,
        ``scored_positions``, ``perplexity``, ``order_invariant``,
        ``mean_context_length``.

    References
    ----------
    Yang et al (2019) arXiv:1906.08237, eq (3) and (5).

    Examples
    --------
    >>> import numpy as np
    >>> lg = np.log([[0.5, 0.5], [0.25, 0.75]])
    >>> out = kamath_permutation_lm_loss(lg, [0, 1], [1, 0])
    >>> round(float(out["loss"]), 6)  # (-log 0.5 - log 0.75) / 2
    0.490415
    """
    L = np.asarray(logits, dtype=float)
    if L.ndim == 1:
        L = L[:, None]
    if L.ndim != 2:
        raise ValueError(f"logits must be 2-D (T, V); got shape {L.shape}.")
    T, V = L.shape
    y = np.asarray(targets, dtype=int).ravel()
    if y.size != T:
        raise ValueError(
            f"targets has length {y.size} but logits has {T} positions."
        )
    if np.any(y < 0) or np.any(y >= V):
        raise ValueError(f"targets must lie in 0 .. {V - 1}.")
    masks = permutation_attention_masks(permutation)
    z = np.asarray(permutation, dtype=int).ravel()
    if z.size != T:
        raise ValueError(
            f"permutation has length {z.size} but logits has {T} positions."
        )
    if reduction not in ("mean", "sum", "none"):
        raise ValueError('reduction must be "mean", "sum" or "none".')

    logp = _log_softmax(L)
    nll = -logp[np.arange(T), y]

    if num_predict is None:
        scored = z
        partial = False
    else:
        k = int(num_predict)
        if k < 1 or k > T:
            raise ValueError(f"num_predict must lie in 1 .. {T}; got {k}.")
        scored = z[T - k:]
        partial = k < T

    sel = nll[scored]
    if reduction == "sum":
        loss = float(np.sum(sel))
    elif reduction == "mean":
        loss = float(np.mean(sel))
    else:
        loss = sel.copy()

    # context length available to each scored position under this order
    ctx = masks["rank"][scored]
    out = RichResult(
        title="Permutation language-model loss",
        summary_lines=[
            ("Loss", loss if np.isscalar(loss) else "(per token)"),
            ("Scored positions", int(scored.size)),
            ("Reduction", reduction),
        ],
        payload={
            "loss": loss,
            "estimate": loss,
            "token_nll": nll,
            "scored_positions": scored,
            "mean_context_length": float(np.mean(ctx)),
            "perplexity": (float(np.exp(np.mean(sel)))
                           if sel.size else float("nan")),
            "order_invariant": not partial,
            "partial_prediction": partial,
            "content_mask": masks["content"],
            "query_mask": masks["query"],
            "rank": masks["rank"],
            "n": T,
            "vocab_size": V,
            "method": _METHOD,
        },
    )
    if partial:
        out.warnings.append(
            "Partial prediction is on, so the loss depends on the sampled "
            "factorization order. Comparing losses across permutations is "
            "only meaningful once averaged over many draws."
        )
    return out


def cheatsheet():
    return (
        "kmperm: permutation LM loss (XLNet eq 3/5) plus the two-stream "
        "content and query attention masks a factorization order induces"
    )
