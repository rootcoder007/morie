# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""DALL-E: text-to-image generation via discrete VAE + autoregressive transformer."""

import numpy as np

from ._richresult import RichResult
from .grdal import geron_dalle_autoregressive_token

__all__ = ["geron_dalle"]


def geron_dalle(text, model, n_image_tokens=4, temperature=1.0, top_k=None, image_vocab=None, grid=None):
    """
    DALL-E: text-to-image generation via discrete VAE + autoregressive
    transformer.

    Formula: text tokens -> image tokens via autoregressive LM

    The generation loop is real: starting from the text prompt, image
    tokens are produced one at a time, each conditioned on everything
    generated so far, with the per-step scoring DELEGATED to
    :func:`morie.fn.grdal.geron_dalle_autoregressive_token`.

    The point of the discrete VAE is that an image *is* a short sequence
    of codebook indices, so image generation reduces to language
    modelling with no change to the transformer. That is made concrete
    here: ``n_image_tokens`` tokens are drawn from a codebook of
    ``image_vocab`` entries and reshaped to ``grid``, giving the token
    map the decoder would expand back to pixels.

    Sampling is deterministic given the model: ``temperature`` and
    ``top_k`` reshape the distribution and the argmax of the reshaped
    distribution is taken, so the same prompt and model always give the
    same image tokens. ``log_likelihood`` accumulates the score of the
    generated sequence, and ``perplexity`` is its exponential per token.

    ``model`` must be a callable ``model(context) -> logits`` over a
    constant vocabulary; the contract is enforced on every step.

    Parameters
    ----------
    text : sequence of int
        Prompt tokens.
    model : callable
        ``model(context) -> array of shape (V,)``.
    n_image_tokens : int, default 4
        Image tokens to generate.
    temperature : float, default 1.0
    top_k : int, optional
    image_vocab : int, optional
        Codebook size; default the model's logit width.
    grid : tuple, optional
        Reshape of the token sequence; default a square if it fits.

    Returns
    -------
    result : RichResult
        Keys: image_tokens, token_grid, log_likelihood, token_logprobs,
        perplexity, context, n_steps, estimate, n, method.

    Examples
    --------
    A model that is uniform over two codebook entries: every step is a
    coin flip, so each token costs ``log 2`` and the perplexity is 2.

    >>> import numpy as np, math
    >>> uniform = lambda ctx: np.zeros(2)
    >>> r = geron_dalle([0, 1], uniform, n_image_tokens=4)
    >>> len(r["image_tokens"]), r["n_steps"]
    (4, 4)
    >>> round(r["log_likelihood"], 9) == round(4 * math.log(0.5), 9)
    True
    >>> round(r["perplexity"], 6)
    2.0

    A model that always prefers token 1 generates only ones, and the
    tokens reshape into the image grid:

    >>> skewed = lambda ctx: np.array([0.0, 5.0])
    >>> r2 = geron_dalle([0], skewed, n_image_tokens=4)
    >>> r2["image_tokens"]
    [1, 1, 1, 1]
    >>> r2["token_grid"]
    [[1, 1], [1, 1]]

    A model whose vocabulary changes between calls breaks the contract:

    >>> bad = lambda ctx: np.zeros(len(ctx))
    >>> geron_dalle([0, 1], bad, n_image_tokens=2)
    Traceback (most recent call last):
      ...
    ValueError: geron_dalle: model returned 3 logits at step 1 but 2 at step 0; the vocabulary must be constant

    References
    ----------
    Géron Ch 16
    """
    if not callable(model):
        raise ValueError("geron_dalle: model must be a callable model(context) -> logits")
    prompt = [int(t) for t in np.asarray(text).ravel().tolist()]
    if not prompt:
        raise ValueError("geron_dalle: text prompt is empty")
    N = int(n_image_tokens)
    if N < 1:
        raise ValueError(f"geron_dalle: n_image_tokens must be >= 1, got {n_image_tokens!r}")

    tokens = []
    logprobs = []
    V0 = None
    for step in range(N):
        ctx = prompt + tokens
        logits = np.asarray(model(ctx), dtype=float).ravel()
        if V0 is None:
            V0 = logits.size
        elif logits.size != V0:
            raise ValueError(
                f"geron_dalle: model returned {logits.size} logits at step {step} but {V0} at step 0; "
                "the vocabulary must be constant"
            )
        if not np.all(np.isfinite(logits)):
            raise ValueError(f"geron_dalle: model returned non-finite logits at step {step}")
        step_res = geron_dalle_autoregressive_token(
            prompt, tokens, model, temperature=temperature, top_k=top_k
        )
        probs = np.asarray(step_res["next_token_probs"], dtype=float)
        nxt = int(np.argmax(probs))
        # Score at temperature 1, as the likelihood always is.
        shift = logits - logits.max()
        lp = float(shift[nxt] - np.log(np.exp(shift).sum()))
        logprobs.append(lp)
        tokens.append(nxt)

    V = int(V0)
    vocab = V if image_vocab is None else int(image_vocab)
    if vocab < 1:
        raise ValueError(f"geron_dalle: image_vocab must be >= 1, got {image_vocab!r}")
    if max(tokens) >= vocab:
        raise ValueError(f"geron_dalle: generated token {max(tokens)} lies outside the codebook of {vocab}")

    if grid is None:
        side = int(round(np.sqrt(N)))
        gr = (side, side) if side * side == N else (1, N)
    else:
        gr = tuple(int(v) for v in grid)
        if gr[0] * gr[1] != N:
            raise ValueError(f"geron_dalle: grid {gr} does not hold {N} tokens")

    ll = float(np.sum(logprobs))

    return RichResult(
        title="DALL-E generation",
        summary_lines=[("Image tokens", N), ("Log-likelihood", ll), ("Codebook", vocab)],
        interpretation="A discrete VAE turns an image into a token sequence, so image generation is just language modelling.",
        payload={
            "image_tokens": tokens,
            "token_grid": np.asarray(tokens).reshape(gr).tolist(),
            "log_likelihood": ll,
            "token_logprobs": logprobs,
            "perplexity": float(np.exp(-ll / N)),
            "context": prompt + tokens,
            "prompt": prompt,
            "n_steps": N,
            "vocab_size": vocab,
            "grid": gr,
            "temperature": float(temperature),
            "estimate": ll,
            "n": int(N),
            "method": "autoregressive image-token generation; per-step scoring delegated to grdal",
        },
    )


def cheatsheet():
    return "hmdale: DALL-E: text-to-image generation via discrete VAE + autoregressive transformer"
