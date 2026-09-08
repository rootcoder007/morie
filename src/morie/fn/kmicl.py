# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""In-context learning conditional probability with K demonstrations."""

import math

from ._richresult import RichResult

__all__ = ["kamath_in_context_learning_prob"]


def kamath_in_context_learning_prob(demonstrations, query, model,
                                    answer=None, sep="\n"):
    """P(y | x, D_K) = P_LLM(y | [ex_1, ..., ex_K, x]).

    The prompt is assembled by concatenating the K demonstrations
    ahead of the query -- that IS in-context learning, no weights
    move. ``model(prompt, answer) -> probability in [0, 1]`` is the
    caller's LM; the contract is enforced, so a model returning a
    log-probability or a logit is rejected instead of being reported
    as a probability.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 3, in-context
    learning.

    Examples
    --------
    >>> demos = ["1 -> odd", "2 -> even"]
    >>> out = kamath_in_context_learning_prob(
    ...     demos, "3 ->", lambda p, y: 0.25 if "2 -> even" in p else 0.0,
    ...     answer="odd")
    >>> out["estimate"]
    0.25
    >>> out["K"]
    2
    >>> abs(out["log_prob"] + 2 * math.log(2)) < 1e-12
    True
    """
    demos = [str(d) for d in demonstrations]
    if not callable(model):
        raise ValueError(
            "model must be callable (prompt, answer) -> probability.")
    parts = demos + [str(query)]
    prompt = sep.join(parts)
    p = model(prompt, answer)
    try:
        p = float(p)
    except (TypeError, ValueError):
        raise ValueError(
            "the model must return a numeric probability.") from None
    if not 0.0 <= p <= 1.0:
        raise ValueError(
            f"the model returned {p}, which is not a probability; "
            "P(y | x, D_K) must lie in [0, 1].")
    with_log = -math.inf if p == 0.0 else math.log(p)
    return RichResult(payload={
        "estimate": p, "probability": p, "log_prob": with_log,
        "prompt": prompt, "K": len(demos), "n": len(parts),
        "method": "In-context learning conditional probability"})


def cheatsheet():
    return "kmicl: P(y | [ex_1..ex_K, x]) from a caller-supplied LM"
