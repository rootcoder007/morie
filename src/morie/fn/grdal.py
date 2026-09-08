# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""DALL-E autoregressive text-to-image token modeling."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_dalle_autoregressive_token"]

_METHOD = "Autoregressive image-token likelihood"


def _log_softmax(z):
    z = z - z.max()
    return z - np.log(np.exp(z).sum())


def geron_dalle_autoregressive_token(text_tokens, image_tokens_prefix, logits_fn,
                                     temperature=1.0, top_k=None):
    r"""Score an image-token prefix and produce the next-token distribution.

    .. math::
        p(\text{img} \mid \text{text}) =
        \prod_t p(\text{img}_t \mid \text{text}, \text{img}_{<t})

    DALL-E treats an image as a sentence of discrete VQ codebook tokens
    and models it with an ordinary autoregressive transformer -- the
    modality change is entirely in the tokeniser.  This routine supplies
    the surrounding machinery (teacher-forced scoring, temperature,
    top-k truncation) and defers the model itself to ``logits_fn``.

    Parameters
    ----------
    text_tokens : sequence of int
        Conditioning prompt tokens.
    image_tokens_prefix : sequence of int
        Image tokens generated so far; may be empty.
    logits_fn : callable
        ``logits_fn(context) -> array of shape (V,)``, where ``context``
        is the concatenated ``list(text_tokens) + prefix_so_far``.
        Must return finite logits of a constant length.
    temperature : float, optional
        Sampling temperature applied to the *next-token* distribution
        only (the likelihood is always scored at temperature 1).
    top_k : int, optional
        Truncate the next-token distribution to its ``k`` largest
        logits before renormalising.

    Returns
    -------
    RichResult
        Payload keys ``log_likelihood`` (of the prefix given the text),
        ``token_logprobs``, ``perplexity``, ``next_token_probs``,
        ``next_token``, ``vocab_size``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 16, DALL-E section.

    Examples
    --------
    A model that is uniform over two tokens assigns each prefix token
    ``log 1/2``:

    >>> import numpy as np, math
    >>> uniform = lambda ctx: np.zeros(2)
    >>> r = geron_dalle_autoregressive_token([0], [1, 0], uniform)
    >>> round(r["log_likelihood"], 6) == round(2 * math.log(0.5), 6)
    True
    >>> [round(p, 6) for p in r["next_token_probs"]]
    [0.5, 0.5]
    >>> round(r["perplexity"], 6)
    2.0

    ``top_k=1`` collapses the next-token distribution onto the argmax:

    >>> skewed = lambda ctx: np.array([0.0, 5.0])
    >>> geron_dalle_autoregressive_token([0], [], skewed, top_k=1)["next_token_probs"]
    [0.0, 1.0]
    """
    text = [int(t) for t in np.asarray(text_tokens).ravel().tolist()]
    prefix = [int(t) for t in np.asarray(image_tokens_prefix).ravel().tolist()]
    if not callable(logits_fn):
        raise ValueError(
            f"logits_fn must be callable(context) -> logits, got {type(logits_fn).__name__}."
        )
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"temperature must be a positive finite float, got {temperature}.")

    V = None
    logprobs = []
    context = list(text)
    for step in range(len(prefix) + 1):
        z = np.asarray(logits_fn(list(context)), dtype=float).ravel()
        if z.size == 0:
            raise ValueError(f"logits_fn returned an empty array at step {step}.")
        if not np.all(np.isfinite(z)):
            raise ValueError(f"logits_fn returned non-finite logits at step {step}.")
        if V is None:
            V = z.size
        elif z.size != V:
            raise ValueError(
                f"logits_fn changed vocabulary size from {V} to {z.size} at step {step}."
            )
        if step < len(prefix):
            tok = prefix[step]
            if not (0 <= tok < V):
                raise ValueError(
                    f"image token {tok} at position {step} is outside the "
                    f"vocabulary of size {V}."
                )
            logprobs.append(float(_log_softmax(z)[tok]))
            context.append(tok)
        else:
            next_logits = z

    if top_k is not None:
        k = int(top_k)
        if not (1 <= k <= V):
            raise ValueError(f"top_k must lie in [1, {V}], got {k}.")
        cut = np.sort(next_logits)[-k]
        next_logits = np.where(next_logits >= cut, next_logits, -np.inf)

    scaled = next_logits / temperature
    finite = np.isfinite(scaled)
    probs = np.zeros(V)
    lp = _log_softmax(scaled[finite])
    probs[finite] = np.exp(lp)

    ll = float(np.sum(logprobs))
    ppl = float(np.exp(-ll / len(prefix))) if prefix else float("nan")

    return RichResult(
        title="DALL-E autoregressive tokens",
        summary_lines=[("Log-likelihood", ll), ("Perplexity", ppl),
                       ("Vocabulary", int(V))],
        payload={
            "log_likelihood": ll,
            "token_logprobs": logprobs,
            "perplexity": ppl,
            "next_token_probs": probs.tolist(),
            "next_token": int(np.argmax(probs)),
            "vocab_size": int(V),
            "context_length": len(text) + len(prefix),
            "estimate": ll,
            "n": int(len(prefix)),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grdal: p(img|text) = prod_t p(img_t | text, img_<t); teacher-forced scoring + next-token"
