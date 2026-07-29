# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""BERT masked-language-modeling loss."""

import numpy as np

from ._richresult import RichResult
from .grgptl import geron_gpt_autoregressive_loss

__all__ = ["geron_bert_mlm_loss"]

_METHOD = "Masked language modeling loss (BERT)"


def geron_bert_mlm_loss(logits, targets, mask):
    r"""Cross-entropy over the masked positions only.

    .. math::
        L_{\text{MLM}} = -\sum_{i \in \text{masked}}
        \log p(x_i \mid \text{context without } x_i)

    The unmasked positions contribute nothing.  That is the whole
    difference from the autoregressive loss
    (:mod:`morie.fn.grgptl`), and it is why BERT is the more expensive
    pre-training objective per token seen: only the ~15% of positions
    that were masked produce any gradient, whereas GPT gets a signal at
    every position.  What BERT buys for that price is *bidirectional*
    context -- the prediction at ``i`` sees both sides.

    The arithmetic is the same cross-entropy, so the masked rows are
    selected here and handed to
    :func:`morie.fn.grgptl.geron_gpt_autoregressive_loss`.

    Parameters
    ----------
    logits : array-like, shape (T, V)
    targets : array-like of int, shape (T,)
        True tokens; entries at unmasked positions are ignored.
    mask : array-like of bool, shape (T,)
        True where the token was masked. At least one must be True.

    Returns
    -------
    RichResult
        Payload keys ``loss``, ``mean_loss``, ``per_token_loss``
        (masked positions only), ``masked_positions``, ``n_masked``,
        ``mask_rate``, ``perplexity``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 15, BERT pretraining (MLM) section (Devlin et al. 2019).

    Examples
    --------
    Only the masked position is scored -- the confident, wrong second
    position costs nothing because it was never masked:

    >>> logits = [[0.0, 0.0], [10.0, 0.0]]
    >>> r = geron_bert_mlm_loss(logits, [0, 1], [True, False])
    >>> round(r["loss"], 10)
    0.6931471806
    >>> r["n_masked"], r["mask_rate"]
    (1, 0.5)

    Mask that second position too and the ignored mistake shows up:

    >>> r2 = geron_bert_mlm_loss(logits, [0, 1], [True, True])
    >>> round(r2["loss"], 6)
    10.693193
    """
    Z = np.atleast_2d(np.asarray(logits, dtype=float))
    t = np.asarray(targets).ravel()
    m = np.asarray(mask)
    if m.dtype != bool:
        if not np.all(np.isin(m, (0, 1))):
            raise ValueError("mask must be boolean (or 0/1).")
        m = m.astype(bool)
    m = m.ravel()
    if Z.ndim != 2:
        raise ValueError(f"logits must be 2-D (T, V), got shape {Z.shape}.")
    T = Z.shape[0]
    if t.size != T or m.size != T:
        raise ValueError(
            f"logits has {T} positions but targets has {t.size} and mask has {m.size}."
        )
    n_masked = int(m.sum())
    if n_masked == 0:
        raise ValueError(
            "no position is masked, so the MLM loss has nothing to score; "
            "BERT masks about 15% of tokens."
        )

    inner = geron_gpt_autoregressive_loss(Z[m], t[m])

    return RichResult(
        title="BERT MLM loss",
        summary_lines=[("Loss", inner["loss"]), ("Masked", n_masked),
                       ("Mask rate", n_masked / T)],
        payload={
            "loss": inner["loss"],
            "mean_loss": inner["mean_loss"],
            "perplexity": inner["perplexity"],
            "per_token_loss": inner["per_token_loss"],
            "masked_positions": np.flatnonzero(m).tolist(),
            "n_masked": n_masked,
            "mask_rate": float(n_masked) / float(T),
            "estimate": inner["loss"],
            "n": int(T),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grmlm: cross-entropy on masked positions only (rows handed to grgptl)"
