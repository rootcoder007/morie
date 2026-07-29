# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Ch 3: AutoPrompt discrete trigger-token search."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_autoprompt_gradient_search"]


def kamath_autoprompt_gradient_search(template, dataset, model,
                                      vocab=None, grad_fn=None,
                                      passes=1):
    r"""Fill each trigger slot with argmax_v of the candidate score.

    ``template`` is the prompt as a token list with ``None`` at every
    trigger position. Two selection rules are supported, both the
    caller's:

    * ``grad_fn(filled, dataset, position) -> one score per vocabulary
      entry`` -- AutoPrompt's first-order gradient score, and the
      candidate with the LARGEST score is taken, per the equation;
    * no ``grad_fn`` -- each candidate is instead scored exactly by
      ``model(filled, dataset) -> loss`` and the smallest loss wins.

    ``model`` is required either way (it reports the loss of the
    filled template), and each pass sweeps the trigger positions left
    to right, so ``passes > 1`` is the usual iterated search.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 3, AutoPrompt; Shin et
    al. (2020).

    Examples
    --------
    >>> loss = lambda tpl, d: 1.0 if tpl[1] == "a" else 0.5
    >>> out = kamath_autoprompt_gradient_search(
    ...     ["x", None], [1], loss, vocab=["a", "b"])
    >>> out["trigger_tokens"], out["estimate"]
    (['b'], 0.5)
    """
    if not callable(model):
        raise ValueError("model must be callable model(filled_template,"
                         " dataset) -> loss.")
    if vocab is None or len(list(vocab)) == 0:
        raise ValueError("vocab= is required and must be non-empty: "
                         "AutoPrompt searches over a candidate set.")
    V = list(vocab)
    filled = list(template)
    slots = [i for i, tok in enumerate(filled) if tok is None]
    if not slots:
        raise ValueError("the template has no trigger slots; mark them "
                         "with None.")
    for i in slots:                      # a starting value to score
        filled[i] = V[0]
    if grad_fn is not None and not callable(grad_fn):
        raise ValueError("grad_fn must be callable or None.")
    n_pass = int(passes)
    if n_pass < 1:
        raise ValueError("passes must be at least 1.")
    history = []
    for _p in range(n_pass):
        for i in slots:
            if grad_fn is None:
                scores = []
                for v in V:
                    trial = list(filled)
                    trial[i] = v
                    scores.append(float(model(trial, dataset)))
                pick = int(np.argmin(scores))
            else:
                g = np.atleast_1d(np.asarray(
                    grad_fn(list(filled), dataset, i), dtype=float))
                if g.size != len(V):
                    raise ValueError(
                        f"grad_fn returned {g.size} scores for a "
                        f"vocabulary of {len(V)}.")
                pick = int(np.argmax(g))
            filled[i] = V[pick]
            history.append((i, V[pick]))
    final = float(model(list(filled), dataset))
    return RichResult(payload={
        "estimate": final, "loss": final,
        "trigger_tokens": [filled[i] for i in slots],
        "prompt": list(filled), "positions": slots,
        "history": history, "n": len(slots),
        "method": "AutoPrompt trigger search (Kamath Ch 3)"})


def cheatsheet():
    return "kmap: coordinate-wise trigger-token search over a vocabulary"
