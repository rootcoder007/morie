# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""GPT decoder-only autoregressive next-token loss."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_gpt_autoregressive_loss"]

_METHOD = "Autoregressive next-token cross-entropy"


def _log_softmax_rows(Z):
    """Row-wise log-softmax, max-shifted so no exp overflows."""
    M = Z.max(axis=1, keepdims=True)
    Z = Z - M
    return Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))


def geron_gpt_autoregressive_loss(logits, targets, reduction="sum"):
    r"""Negative log-likelihood of the next token at every position.

    .. math::
        L = -\sum_{t=1}^{T} \log p(x_t \mid x_{<t})

    Computed as ``logit - logsumexp(logits)`` rather than
    ``log(softmax(...))``: the two are algebraically identical, but the
    second exponentiates first and a confident model routinely produces
    a probability that rounds to 0, whose log is ``-inf``.  The
    log-domain form has no such failure.

    ``perplexity`` is ``exp(mean loss)`` -- the effective number of
    tokens the model is choosing between.  Uniform over ``V`` tokens
    gives exactly ``V``, which is the sanity check to run first on any
    language model.

    Parameters
    ----------
    logits : array-like, shape (T, V)
        Unnormalised scores; position ``t`` predicts ``targets[t]``.
    targets : array-like of int, shape (T,)
    reduction : {"sum", "mean"}, optional
        What ``estimate`` reports. Both are in the payload regardless.

    Returns
    -------
    RichResult
        Payload keys ``loss`` (sum), ``mean_loss``, ``perplexity``,
        ``per_token_loss``, ``target_logprob``, ``estimate``, ``n``,
        ``method``.

    References
    ----------
    Géron Ch 15, GPT / Decoder-only Transformer section.

    Examples
    --------
    A model with no opinion between two tokens pays ``log 2`` per
    position, and its perplexity is exactly 2:

    >>> r = geron_gpt_autoregressive_loss([[0.0, 0.0], [0.0, 0.0]], [0, 1])
    >>> round(r["loss"], 10)
    1.3862943611
    >>> round(r["perplexity"], 10)
    2.0

    Getting one position confidently right and the other confidently
    wrong is far worse than sitting at chance on both:

    >>> r2 = geron_gpt_autoregressive_loss([[10.0, 0.0], [10.0, 0.0]], [0, 1])
    >>> round(r2["loss"], 6)
    10.000091
    """
    Z = np.atleast_2d(np.asarray(logits, dtype=float))
    t = np.asarray(targets).ravel()
    if Z.ndim != 2 or Z.size == 0:
        raise ValueError(f"logits must be a non-empty 2-D array (T, V), got {Z.shape}.")
    if not np.all(np.isfinite(Z)):
        raise ValueError("logits must be finite.")
    T, V = Z.shape
    if t.size != T:
        raise ValueError(f"targets has {t.size} entries but logits has {T} positions.")
    if not np.issubdtype(t.dtype, np.integer):
        t = t.astype(int)
    if t.min() < 0 or t.max() >= V:
        raise ValueError(f"targets must lie in [0, {V - 1}], got range [{t.min()}, {t.max()}].")
    if reduction not in ("sum", "mean"):
        raise ValueError(f"reduction must be 'sum' or 'mean', got {reduction!r}.")

    logp = _log_softmax_rows(Z)
    chosen = logp[np.arange(T), t]
    per = -chosen
    total = float(per.sum())
    mean = float(per.mean())

    return RichResult(
        title="Autoregressive next-token loss",
        summary_lines=[("Loss (sum)", total), ("Mean", mean),
                       ("Perplexity", float(np.exp(mean)))],
        payload={
            "loss": total,
            "mean_loss": mean,
            "perplexity": float(np.exp(mean)),
            "per_token_loss": per.tolist(),
            "target_logprob": chosen.tolist(),
            "estimate": total if reduction == "sum" else mean,
            "n": int(T),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grgptl: L = -sum_t log p(x_t | x_<t) in the log domain; perplexity = exp(mean loss)"
