# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""In-context learning: LLM adapts via examples in prompt."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_in_context_learning"]

_METHOD = "In-context learning by prompt-conditioned scoring"


def geron_in_context_learning(model, examples, query, candidates=None, template="{x} -> {y}", separator="\n"):
    """
    In-context learning: LLM adapts via examples in prompt.

    Formula: P(y | prompt_examples, x_query)

    No weights change.  The demonstrations enter through the prompt, the
    model conditions on them, and the "learning" is entirely a
    conditioning effect -- which is why it vanishes the moment the
    context ends.

    The prompt is assembled here and the scoring is delegated to a
    caller-supplied model, whose contract is enforced:
    ``model(prompt, candidate) -> log-probability`` (a finite float),
    called once per candidate label.  The prediction is the argmax over
    candidates, with the full normalised posterior returned so a
    confident wrong answer is distinguishable from a coin flip.

    Candidates default to the distinct labels appearing in ``examples``,
    which is the honest default: an in-context classifier can only
    produce labels it has been shown.

    ``n_shot`` is reported because the effect is strongly shot-dependent;
    zero examples is a legitimate call (zero-shot) and is not an error.

    Parameters
    ----------
    model : callable
        ``model(prompt, candidate) -> float`` log-probability.
    examples : sequence of (x, y)
        Demonstrations.
    query : object
        The input to classify.
    candidates : sequence, optional
        Label set; defaults to the distinct labels in ``examples``.
    template : str
        Format string with ``{x}`` and ``{y}`` placeholders.
    separator : str
        Joined between demonstrations.

    Returns
    -------
    result : RichResult
        Keys: prediction, prompt, log_probs, posterior, n_shot,
        candidates, estimate, n, method.

    Examples
    --------
    A scorer that prefers whichever label appeared most often in the
    prompt -- crude, but deterministic and checkable:

    >>> def scorer(prompt, cand):
    ...     return float(prompt.count(str(cand)))
    >>> ex = [("a", "pos"), ("b", "pos"), ("c", "neg")]
    >>> r = geron_in_context_learning(scorer, ex, "d")
    >>> r["prediction"]
    'pos'
    >>> r["n_shot"]
    3

    The prompt really does contain the demonstrations and ends with the
    query:

    >>> r["prompt"].splitlines()
    ['a -> pos', 'b -> pos', 'c -> neg', 'd ->']

    The posterior is a distribution over the candidate labels:

    >>> round(float(np.sum(r["posterior"])), 12)
    1.0
    >>> sorted(r["candidates"])
    ['neg', 'pos']

    Zero-shot is allowed and the prompt is then just the query:

    >>> z = geron_in_context_learning(scorer, [], "d", candidates=["pos", "neg"])
    >>> z["n_shot"], z["prompt"]
    (0, 'd ->')

    A scorer that returns something non-finite is refused, with the
    candidate named:

    >>> geron_in_context_learning(lambda p, c: float("inf"), ex, "d")
    Traceback (most recent call last):
        ...
    ValueError: geron_in_context_learning: model returned a non-finite log-probability (inf) for candidate 'pos'

    References
    ----------
    Géron Ch 15
    """
    if not callable(model):
        raise ValueError(f"geron_in_context_learning: model must be callable, got {type(model).__name__}")
    try:
        shots = list(examples)
    except TypeError:
        raise ValueError("geron_in_context_learning: examples must be a sequence of (x, y) pairs") from None
    pairs = []
    for i, ex in enumerate(shots):
        try:
            x, y = ex
        except (TypeError, ValueError):
            raise ValueError(f"geron_in_context_learning: example {i} is not an (x, y) pair") from None
        pairs.append((x, y))

    if candidates is None:
        seen = []
        for _, y in pairs:
            if y not in seen:
                seen.append(y)
        cands = seen
    else:
        cands = list(candidates)
    if not cands:
        raise ValueError(
            "geron_in_context_learning: no candidate labels -- pass candidates explicitly when examples is empty"
        )

    if "{x}" not in template:
        raise ValueError(f"geron_in_context_learning: template must contain an {{x}} placeholder, got {template!r}")
    lines = [template.format(x=x, y=y) for x, y in pairs]
    query_line = template.format(x=query, y="").rstrip()
    prompt = separator.join(lines + [query_line])

    scores = []
    for c in cands:
        s = model(prompt, c)
        try:
            s = float(s)
        except (TypeError, ValueError):
            raise ValueError(
                f"geron_in_context_learning: model returned {type(s).__name__} for candidate {c!r}, expected a float"
            ) from None
        if not np.isfinite(s):
            raise ValueError(
                f"geron_in_context_learning: model returned a non-finite log-probability ({s}) for candidate {c!r}"
            )
        scores.append(s)

    lp = np.asarray(scores)
    shifted = lp - lp.max()
    e = np.exp(shifted)
    post = e / e.sum()
    best = int(np.argmax(lp))

    return RichResult(
        title="In-context learning",
        summary_lines=[
            ("Shots", len(pairs)),
            ("Candidates", len(cands)),
            ("Prediction", cands[best]),
            ("Confidence", float(post[best])),
        ],
        interpretation=(
            "No parameter is updated: the demonstrations condition the model through the prompt, "
            "and the adaptation ends with the context."
        ),
        payload={
            "prediction": cands[best],
            "prompt": prompt,
            "log_probs": lp,
            "posterior": post,
            "n_shot": len(pairs),
            "candidates": cands,
            "estimate": float(post[best]),
            "n": len(pairs),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmicl: in-context learning -- assemble the k-shot prompt, score candidates with a supplied model"
