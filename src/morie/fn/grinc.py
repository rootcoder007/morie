# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""In-context (few-shot) learning: prompt assembly and scoring."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_in_context_learning"]

_METHOD = "In-context / few-shot prompt assembly"


def geron_in_context_learning(examples, query, predict=None, separator="\n",
                              template="{input} -> {output}"):
    r"""Prepend ``K`` demonstrations to the query.

    .. math::
        p(y \mid x) \approx p\bigl(y \mid [\,e_1, \dots, e_K, x\,]\bigr)

    No weights are touched.  The demonstrations are just tokens in the
    context window, which is why the effect appears at inference time
    and disappears the moment the context is cleared -- and why the
    *order* of the examples matters: they are read left to right like
    any other text, and the last one sits closest to the query.

    Assembly is done here; the model is the caller's. If ``predict`` is
    given it must be ``predict(prompt) -> answer`` and must return
    something (``None`` is rejected), so a silently failing model
    surfaces as an error rather than an empty prediction.

    Parameters
    ----------
    examples : sequence of (input, output) pairs
        The ``K`` demonstrations, in the order they should appear.
    query : object
        The input to answer.
    predict : callable, optional
        ``predict(prompt) -> answer``.
    separator : str, optional
        Text between blocks, default newline.
    template : str, optional
        Format string with ``{input}`` and ``{output}`` fields.

    Returns
    -------
    RichResult
        Payload keys ``prompt``, ``k_shot``, ``answer`` (``None``
        without a ``predict``), ``prompt_chars``, ``prompt_words``,
        ``example_order``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 15, In-context / few-shot learning section (Brown et al.
    2020).

    Examples
    --------
    Two demonstrations then the query, left with an open slot:

    >>> ex = [("cat", "chat"), ("dog", "chien")]
    >>> r = geron_in_context_learning(ex, "bird")
    >>> print(r["prompt"])
    cat -> chat
    dog -> chien
    bird ->
    >>> r["k_shot"]
    2

    Zero-shot is the same machinery with an empty list -- the query
    alone:

    >>> geron_in_context_learning([], "bird")["prompt"]
    'bird ->'

    With a model attached, the prompt is what the model sees:

    >>> r2 = geron_in_context_learning(ex, "bird",
    ...                                predict=lambda p: p.count("->"))
    >>> r2["answer"]
    3
    """
    if template.find("{input}") < 0 or template.find("{output}") < 0:
        raise ValueError("template must contain both {input} and {output} fields.")
    ex = list(examples)
    blocks = []
    for i, pair in enumerate(ex):
        try:
            a, b = pair
        except (TypeError, ValueError):
            raise ValueError(
                f"examples[{i}] must be an (input, output) pair, got {pair!r}."
            ) from None
        blocks.append(template.format(input=a, output=b))
    tail = template.format(input=query, output="").rstrip()
    prompt = separator.join(blocks + [tail])

    answer = None
    if predict is not None:
        if not callable(predict):
            raise ValueError(f"predict must be callable, got {type(predict).__name__}.")
        answer = predict(prompt)
        if answer is None:
            raise ValueError("predict returned None; it must return an answer.")
        if isinstance(answer, float) and not np.isfinite(answer):
            raise ValueError(f"predict returned a non-finite value: {answer}.")

    return RichResult(
        title="In-context learning prompt",
        summary_lines=[("Shots", len(ex)), ("Prompt chars", len(prompt))],
        payload={
            "prompt": prompt,
            "k_shot": len(ex),
            "answer": answer,
            "prompt_chars": len(prompt),
            "prompt_words": len(prompt.split()),
            "example_order": [str(a) for a, _ in ex],
            "estimate": answer if answer is not None else prompt,
            "n": len(ex),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grinc: build [e_1 ... e_K, x] prompt; no weight update, order matters, model is the caller's"
