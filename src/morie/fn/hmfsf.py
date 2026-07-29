# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Few-shot learning: small number of in-context examples."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_few_shot"]


def geron_few_shot(model, examples, query, k=None, separator="\n", template="{x} -> {y}", max_context=None):
    """
    Few-shot learning: small number of in-context examples.

    Formula: prompt contains k (x, y) pairs

    The prompt is built and the model is called, with the contract
    enforced: ``model(prompt) -> prediction``, one call per prompt, and a
    model that raises or returns ``None`` is an error rather than a
    silently dropped result.

    What makes this few-shot rather than fine-tuning is that nothing is
    updated -- the ``k`` demonstrations exist only inside the prompt, and
    the same model object answers the zero-shot prompt too. Both are run
    here, and ``changed_by_context`` reports whether the demonstrations
    changed the answer at all. If they did not, the model either already
    knew the task or ignored the examples, and both are worth knowing
    before adding more shots.

    ``k`` selects the first ``k`` demonstrations, which is the honest
    default: sampling them would make the result irreproducible.
    ``max_context`` rejects a prompt that would overflow the model's
    window instead of letting the caller discover it at inference time.

    Parameters
    ----------
    model : callable
        ``model(prompt) -> prediction``.
    examples : sequence of (x, y)
        Demonstrations.
    query : object
        The input to predict.
    k : int, optional
        Number of demonstrations; default all of them.
    separator : str, default "\\n"
    template : str, default "{x} -> {y}"
        Must contain both ``{x}`` and ``{y}``.
    max_context : int, optional
        Maximum prompt length in characters.

    Returns
    -------
    result : RichResult
        Keys: prediction, zero_shot_prediction, prompt, zero_shot_prompt,
        k, changed_by_context, prompt_length, n_available, estimate, n,
        method.

    Examples
    --------
    A model that copies the last demonstration's answer: with shots it
    returns that answer, with none it falls back.

    >>> def copycat(prompt):
    ...     lines = [l for l in prompt.split("\\n") if "->" in l and not l.endswith("-> ")]
    ...     return lines[-1].split("-> ")[1] if lines else "?"
    >>> r = geron_few_shot(copycat, [("a", "1"), ("b", "2")], "c")
    >>> r["prediction"]
    '2'
    >>> r["zero_shot_prediction"]
    '?'
    >>> r["changed_by_context"]
    True
    >>> r["k"], r["n_available"]
    (2, 2)

    Restricting to one shot changes the prompt and the answer:

    >>> r2 = geron_few_shot(copycat, [("a", "1"), ("b", "2")], "c", k=1)
    >>> r2["prediction"]
    '1'
    >>> r2["prompt"]
    'a -> 1\\nc -> '

    A prompt that would overflow the window is rejected up front:

    >>> geron_few_shot(copycat, [("a", "1")], "c", max_context=3)
    Traceback (most recent call last):
      ...
    ValueError: geron_few_shot: the prompt is 12 characters but max_context is 3

    References
    ----------
    Géron Ch 15
    """
    if not callable(model):
        raise ValueError("geron_few_shot: model must be a callable model(prompt) -> prediction")
    ex = list(examples or [])
    if any(not (isinstance(e, (tuple, list)) and len(e) == 2) for e in ex):
        raise ValueError("geron_few_shot: every example must be an (x, y) pair")
    if "{x}" not in template or "{y}" not in template:
        raise ValueError(f"geron_few_shot: template must contain both {{x}} and {{y}}, got {template!r}")
    n_avail = len(ex)
    kk = n_avail if k is None else int(k)
    if kk < 0:
        raise ValueError(f"geron_few_shot: k must be non-negative, got {k!r}")
    if kk > n_avail:
        raise ValueError(f"geron_few_shot: k={kk} exceeds the {n_avail} examples supplied")

    shots = ex[:kk]
    prefix = separator.join(template.format(x=a, y=b) for a, b in shots)
    tail = template.format(x=query, y="").rstrip() + " "
    prompt = (prefix + separator + tail) if shots else tail
    zero_prompt = tail

    if max_context is not None:
        mc = int(max_context)
        if mc < 1:
            raise ValueError(f"geron_few_shot: max_context must be >= 1, got {max_context!r}")
        if len(prompt) > mc:
            raise ValueError(f"geron_few_shot: the prompt is {len(prompt)} characters but max_context is {mc}")

    pred = model(prompt)
    if pred is None:
        raise ValueError("geron_few_shot: model returned None for the few-shot prompt")
    zero = model(zero_prompt)
    if zero is None:
        raise ValueError("geron_few_shot: model returned None for the zero-shot prompt")

    return RichResult(
        title="Few-shot prompting",
        summary_lines=[("Shots", kk), ("Prediction", pred), ("Zero-shot", zero)],
        interpretation="Nothing is trained: the demonstrations live only in the prompt, so k is bounded by the context window.",
        payload={
            "prediction": pred,
            "zero_shot_prediction": zero,
            "prompt": prompt,
            "zero_shot_prompt": zero_prompt,
            "shots": shots,
            "k": int(kk),
            "n_available": int(n_avail),
            "changed_by_context": bool(pred != zero),
            "prompt_length": int(len(prompt)),
            "template": template,
            "estimate": float(kk),
            "n": int(kk),
            "method": "in-context few-shot prompt construction with a zero-shot control",
        },
    )


def cheatsheet():
    return "hmfsf: Few-shot learning: small number of in-context examples"
