# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""One-shot learning: a single labelled example in the prompt."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_one_shot"]


def geron_one_shot(model, example, query, verbalizer=None):
    """
    One-shot learning: a single example in the prompt.

    Formula: prompt = (x_1, y_1, x_query) -> y_query

    Nothing is trained. The single demonstration goes into the CONTEXT
    and the model conditions on it, which is why the technique is called
    in-context learning and why it works at all only for models large
    enough to have learned the task during pretraining -- the example
    selects a behaviour, it does not teach one.

    The demonstration therefore does a lot of work per token, and its
    weaknesses are structural: the label of the single example biases the
    answer toward that class (majority-label bias), and the format of the
    example, not just its content, sets what the model thinks it is being
    asked. Both are visible in ``prompt``, which is returned assembled so
    it can be inspected rather than guessed at.

    ``model`` must be callable on the assembled prompt. There is no
    default: a one-shot predictor without a model would just echo the
    example's label, which is a stub, not a baseline.

    Parameters
    ----------
    model : callable
        ``model(prompt) -> prediction``, where ``prompt`` is a list of
        ``(input, label)`` pairs ending with ``(query, None)``.
    example : tuple or mapping
        The single demonstration ``(x_1, y_1)``.
    query : object
        The input to label.
    verbalizer : callable, optional
        ``verbalizer(label) -> str`` used only for ``prompt_text``.

    Returns
    -------
    result : RichResult
        Keys: prediction, prompt, prompt_text, shots, demo_label,
        estimate, n, method.

    Examples
    --------
    A model that copies the demonstration's label -- the majority-label
    bias in its purest form:

    >>> copy = lambda prompt: prompt[0][1]
    >>> r = geron_one_shot(copy, ("hello", "greeting"), "goodbye")
    >>> r["prediction"], int(r["shots"])
    ('greeting', 1)
    >>> r["prompt"]
    [('hello', 'greeting'), ('goodbye', None)]

    A model that actually reads the query:

    >>> rule = lambda p: "greeting" if "hello" in p[-1][0] else "farewell"
    >>> geron_one_shot(rule, ("hello there", "greeting"), "goodbye now")["prediction"]
    'farewell'

    The assembled prompt is inspectable:

    >>> geron_one_shot(copy, ("hello", "greeting"), "goodbye")["prompt_text"]
    'hello -> greeting\\ngoodbye ->'

    References
    ----------
    Geron Ch 15
    """
    if not callable(model):
        raise ValueError(
            "geron_one_shot: model must be callable; there is no default, since a one-shot predictor "
            "without a model can only echo the demonstration's label"
        )
    if hasattr(example, "get") and not isinstance(example, (list, tuple)):
        if "input" not in example or "label" not in example:
            raise ValueError("geron_one_shot: an example mapping needs 'input' and 'label'")
        x1, y1 = example["input"], example["label"]
    else:
        try:
            x1, y1 = example
        except (TypeError, ValueError):
            raise ValueError("geron_one_shot: example must be an (input, label) pair or a mapping") from None
    if y1 is None:
        raise ValueError("geron_one_shot: the demonstration needs a label; that is what makes it one-shot")

    prompt = [(x1, y1), (query, None)]
    verb = (lambda v: str(v)) if verbalizer is None else verbalizer
    if not callable(verb):
        raise ValueError("geron_one_shot: verbalizer must be callable")
    text = f"{x1} -> {verb(y1)}\n{query} ->"

    pred = model(prompt)
    if pred is None:
        raise ValueError("geron_one_shot: the model returned None; a prediction is required")

    return RichResult(
        title="One-shot prompt",
        summary_lines=[("Shots", 1), ("Demonstration label", y1), ("Prediction", pred)],
        interpretation="Nothing is trained: the demonstration selects a behaviour the model already had.",
        payload={
            "prediction": pred,
            "prompt": prompt,
            "prompt_text": text,
            "shots": 1,
            "demo_label": y1,
            "query": query,
            "estimate": pred,
            "n": 1,
            "method": "One-shot in-context prompt assembly and model call",
        },
    )


def cheatsheet():
    return "hmosf: One-shot in-context prompting"
